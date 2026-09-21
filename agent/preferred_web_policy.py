"""GXTD-603 explicit request capability. No process-global permission state."""
from __future__ import annotations

import ipaddress
import json
import re
import threading
import time
from urllib.parse import urlsplit, urlunsplit

TOOLS = frozenset({'web_search', 'web_extract'})
PROFILE = 'homelab-coordinator'


class PolicyError(ValueError):
    pass


def canonical(value):
    if not isinstance(value, str) or len(value) > 4096 or not re.match(r'^https?://', value) or re.search(r'[\s\\\x00-\x1f]', value):
        raise PolicyError('invalid_source_url')
    try:
        parts = urlsplit(value)
        host = parts.hostname
        if not host or parts.username or parts.password or parts.port not in (None, 80 if parts.scheme == 'http' else 443):
            raise PolicyError('unsafe_source_url')
        host = host.encode('idna').decode().lower()
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if '.' not in host or host.endswith(('.local', '.localhost', '.internal', '.invalid', '.test', '.onion')):
                raise PolicyError('unsafe_source_url')
        else:
            if not address.is_global or address.is_multicast:
                raise PolicyError('unsafe_source_url')
            if address.version == 6:
                host = '[' + host + ']'
        return urlunsplit((parts.scheme, host, parts.path or '/', parts.query, ''))
    except (ValueError, UnicodeError) as error:
        raise PolicyError('unsafe_source_url') from error


def explicit(text):
    return isinstance(text, str) and bool(re.match(r'^/research(?:\s|$)', text.strip()))


def admit(text, requested_mode, profile, configured):
    if requested_mode is None and not explicit(text):
        return None
    if requested_mode not in (None, 'research') or profile != PROFILE or configured is not True:
        raise PolicyError('research_mode_unavailable')
    if not isinstance(text, str):
        raise PolicyError('research_multimodal_unsupported')
    return ResearchPolicy(text)


def unsupported_request(body, path):
    if not isinstance(body, dict):
        return False
    text = body.get('message', body.get('input', ''))
    messages = body.get('messages')
    if isinstance(messages, list) and messages:
        last = messages[-1]
        text = last.get('content', '') if isinstance(last, dict) and last.get('role') == 'user' else ''
    requested = 'preferred_web_mode' in body or explicit(text)
    return requested and not path.endswith('/v1/chat/completions')


def bind_agent(agent, policy):
    if not isinstance(policy, ResearchPolicy):
        raise PolicyError('research_policy_missing')
    agent._preferred_web_required = True
    agent._preferred_web_policy = policy
    agent.tools = [tool for tool in agent.tools if tool.get('function', {}).get('name') in TOOLS]
    agent.valid_tool_names = {tool['function']['name'] for tool in agent.tools}
    if agent.valid_tool_names != TOOLS:
        raise PolicyError('research_tools_unavailable')


def dispatch_for_agent(agent, name, args):
    if not getattr(agent, '_preferred_web_required', False):
        return None
    policy = getattr(agent, '_preferred_web_policy', None)
    if not isinstance(policy, ResearchPolicy):
        return ResearchPolicy.error('research_policy_missing')
    try:
        return policy.execute(name, args)
    except Exception:
        return ResearchPolicy.error('research_policy_failed')


class ResearchPolicy:
    def __init__(self, user_text):
        self.allowed = {}
        self.lock = threading.RLock()
        self.closed = False
        self.failed = set()
        self.attempts = {}
        self.not_before = {}
        for raw in re.findall(r'https?://[^\s<>"\x00-\x1f]+', user_text):
            self._add(raw.rstrip(').,;!?]'), 0)

    def _add(self, value, depth):
        url = canonical(value)
        with self.lock:
            if len(self.allowed) < 256:
                self.allowed[url] = min(depth, self.allowed.get(url, depth))
        return url

    def _guard(self, capability):
        with self.lock:
            if capability in self.failed:
                raise PolicyError('web_backend_unavailable')
        from tools.web_backend_policy import native_backend_guard
        from hermes_cli.config import load_config_readonly
        web = load_config_readonly().get('web') or {}
        expected = {'search': 'http://chatty:8891/searxng', 'extract': 'https://chatty.tail5f50dc.ts.net:16443'}
        if web.get('preferred_research_mode') is not True or web.get('required_endpoints') != expected or native_backend_guard(capability):
            with self.lock:
                self.failed.add(capability)
            raise PolicyError('web_backend_unavailable')

    def _budget(self, capability, key):
        with self.lock:
            if self.closed:
                raise PolicyError('research_request_closed')
            if capability in self.failed:
                raise PolicyError('web_backend_unavailable')
            if time.monotonic() < self.not_before.get(capability, 0):
                raise PolicyError('backend_retry_after')
            slot = (capability, key)
            if self.attempts.get(slot, 0) >= 2:
                raise PolicyError('backend_retry_exhausted')
            self.attempts[slot] = self.attempts.get(slot, 0) + 1

    @staticmethod
    def _native_search(query, limit):
        from tools.web_tools import web_search_tool
        return web_search_tool(query, limit)

    def _search(self, query, limit):
        self._budget('search', query)
        try:
            result = json.loads(self._native_search(query, limit))
            if result.get('error') or result.get('success') is False:
                with self.lock:
                    self.failed.add('search')
                return self.error('web_backend_unavailable')
            rows = result['data']['web']
            if not isinstance(rows, list):
                raise ValueError('invalid search envelope')
            for row in rows:
                try:
                    if not row.get('error'):
                        self._add(row['url'], 0)
                except (PolicyError, KeyError, TypeError, AttributeError):
                    continue
            result['policy'] = 'gxtd603-v1'
            result['backend'] = 'searxng'
            return json.dumps(result)
        except Exception:
            with self.lock:
                self.failed.add('search')
            return self.error('web_backend_unavailable')

    @staticmethod
    def _post(body):
        import httpx
        from hermes_cli.config import get_env_value
        # The effective endpoint was checked immediately before this operation.
        endpoint = get_env_value('FIRECRAWL_API_URL')
        if endpoint.rstrip('/') != 'https://chatty.tail5f50dc.ts.net:16443':
            raise PolicyError('web_backend_unavailable')
        key = get_env_value('FIRECRAWL_API_KEY')
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) as client:
            response = client.post(endpoint.rstrip('/') + '/v2/research/scrape', json=body, headers=headers)
            return response.status_code, response.json(), response.headers

    def _extract(self, url, allowed):
        self._budget('extract', url)
        try:
            status, response, headers = self._post({'url': url, 'allowedUrls': allowed})
            if status == 429:
                wait = headers.get('retry-after', '')
                with self.lock:
                    # Unknown/date Retry-After stops recovery, never guesses a shorter delay.
                    if not str(wait).isdigit() or int(wait) > 30:
                        self.failed.add('extract')
                    else:
                        self.not_before['extract'] = time.monotonic() + int(wait)
                raise PolicyError('backend_rate_limited')
            if status != 200 or response.get('success') is not True:
                if status in (401, 403) or 300 <= status < 400:
                    with self.lock:
                        self.failed.add('extract')
                return {'url': url, 'content': '', 'error': response.get('code') or 'source_fetch_failed',
                        'metadata': response.get('metadata', {}), 'backend': 'firecrawl', 'status': status}
            data = response['data']
            metadata = data['metadata']
            if metadata.get('policy') != 'gxtd603-v1' or metadata.get('statusCode') != 200 or canonical(metadata['url']) not in allowed:
                raise PolicyError('extraction_policy_missing')
            content = data.get('markdown')
            if not isinstance(content, str) or not content.strip():
                raise PolicyError('source_unusable')
            with self.lock:
                depth = self.allowed[url]
            if depth < 2:
                origin = urlsplit(metadata['url']).netloc
                for link in data.get('links', [])[:128]:
                    try:
                        accepted = canonical(link)
                        if urlsplit(accepted).netloc == origin:
                            self._add(accepted, depth + 1)
                    except PolicyError:
                        continue
            return {'url': url, 'title': metadata.get('title', ''), 'content': content,
                    'error': None, 'metadata': metadata, 'backend': 'firecrawl'}
        except PolicyError:
            raise
        except Exception:
            return {'url': url, 'content': '', 'error': 'source_fetch_failed', 'backend': 'firecrawl'}

    @staticmethod
    def error(code):
        return json.dumps({'success': False, 'code': code, 'error': code,
                           'policy': 'gxtd603-v1'})

    def execute(self, name, args):
        if name not in TOOLS:
            return self.error('research_tool_denied')
        if self.closed:
            return self.error('research_request_closed')
        if not isinstance(args, dict):
            return self.error('invalid_research_arguments')
        capability = 'search' if name == 'web_search' else 'extract'
        urls, allowed = [], []
        try:
            if capability == 'extract':
                if set(args) - {'urls', 'char_limit'} or not isinstance(args.get('urls'), list) or not 1 <= len(args['urls']) <= 5:
                    raise PolicyError('invalid_research_arguments')
                urls = [canonical(url) for url in args['urls']]
                with self.lock:
                    if any(url not in self.allowed for url in urls):
                        raise PolicyError('source_not_approved')
                    allowed = list(self.allowed)
                limit = args.get('char_limit') or 15000
                if type(limit) is not int or not 2000 <= limit <= 50000:
                    raise PolicyError('invalid_research_arguments')
            else:
                if set(args) - {'query', 'limit'} or not isinstance(args.get('query'), str) or not 1 <= len(args['query']) <= 2000:
                    raise PolicyError('invalid_research_arguments')
                limit = args.get('limit', 5)
                if type(limit) is not int or not 1 <= limit <= 10:
                    raise PolicyError('invalid_research_arguments')
            self._guard(capability)
            if capability == 'search':
                return self._search(args['query'], limit)
            results = []
            for url in urls:
                try:
                    row = self._extract(url, allowed)
                    if isinstance(row.get('content'), str):
                        row['content'] = row['content'][:limit]
                    results.append(row)
                except PolicyError as error:
                    results.append({'url': url, 'content': '', 'error': str(error), 'backend': 'firecrawl'})
            return json.dumps({'results': results, 'policy': 'gxtd603-v1'})
        except PolicyError as error:
            return self.error(str(error))
