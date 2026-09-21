"""GXTD-624: native OWUI location, authenticated chat binding, existing Hermes tools.

OWUI owns acquisition and consent. Its native value has no accuracy/sample time;
never invent either, read saved user-info location, or acquire on the server.
"""
import hashlib
import hmac
import json
import math
import re
import time
from dataclasses import dataclass, field

CONTEXT = 'hermes_chat.v1'
UUID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
NATIVE = re.compile(r'^\s*(-?\d{1,3}(?:\.\d{1,8})?),\s*(-?\d{1,3}(?:\.\d{1,8})?)\s*\(lat, long\)\s*$')
POLICY = ('This turn only: location describes the browser device, not the Hermes server. '
          'An explicit requested place takes precedence. Never infer current location from '
          'past turns, home memory, identity, hostnames, IP or timezone. '
          'For unknown device location, ask for a city/postcode rather than guessing. '
          'Use existing web tools to retrieve current weather for the supplied coordinates/place; '
          'name the source, place, units and local forecast date. No weather_here or chat_location '
          'tool is provided or needed. Do not save coordinates as a durable memory. ')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('invalid_location')
    return value


def identifier(value):
    if not isinstance(value, str) or not UUID.fullmatch(value):
        raise ValueError('invalid_chat_binding')
    return value


def native_context(*, user_id, metadata, now=None):
    now = time.time() if now is None else now
    if not isinstance(user_id, str) or not user_id or len(user_id) > 128:
        raise ValueError('location_identity_required')
    ids = {name: identifier(metadata.get(name)) for name in ('chat_id', 'message_id', 'user_message_id')}
    variables = metadata.get('variables') or {}
    raw = variables.get('{{USER_LOCATION}}') if isinstance(variables, dict) else None
    match = NATIVE.fullmatch(raw) if isinstance(raw, str) and len(raw) < 100 else None
    location = {'status': 'unknown'}
    if match:
        lat, lon = map(float, match.groups())
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            location = {'status': 'available', 'source': 'owui_native', 'latitude': lat, 'longitude': lon}
    return {'schema': CONTEXT, 'subject_id': user_id, **ids, 'received_at': now, 'location': location}


def body_digest(body):
    return hashlib.sha256(canonical({k: v for k, v in body.items() if k != 'hermes_chat_context'})).hexdigest()


def sign_context(context, *, body, user_id, profile, key, now=None):
    now = time.time() if now is None else now
    if not key or not user_id or context.get('subject_id') != user_id or context.get('schema') != CONTEXT:
        raise ValueError('location_identity_mismatch')
    if set(context) != {'schema', 'subject_id', 'chat_id', 'message_id', 'user_message_id', 'received_at', 'location'}:
        raise ValueError('invalid_chat_context')
    for name in ('chat_id', 'message_id', 'user_message_id'):
        identifier(context[name])
    if not 0 <= now - context['received_at'] <= 60:
        raise ValueError('expired_chat_context')
    signed = {**context, 'profile': profile, 'issued_at': now, 'body_sha256': body_digest(body)}
    signed['signature'] = hmac.new(key.encode(), canonical(signed), hashlib.sha256).hexdigest()
    return signed


def accept_context(value, *, body, profile, key, now=None):
    now = time.time() if now is None else now
    if not isinstance(value, dict) or len(canonical(value)) > 8192 or not key:
        raise ValueError('invalid_chat_context')
    data = {k: v for k, v in value.items() if k != 'signature'}
    signature = hmac.new(key.encode(), canonical(data), hashlib.sha256).hexdigest()
    if not isinstance(value.get('signature'), str) or not hmac.compare_digest(signature, value['signature']):
        raise ValueError('invalid_chat_signature')
    if data.get('profile') != profile or data.get('schema') != CONTEXT or data.get('body_sha256') != body_digest(body):
        raise ValueError('chat_context_mismatch')
    if not 0 <= now - number(data.get('issued_at'), 0, now) <= 60 or not 0 <= now - number(data.get('received_at'), 0, now) <= 60:
        raise ValueError('expired_chat_context')
    for name in ('chat_id', 'message_id', 'user_message_id'):
        identifier(data.get(name))
    if not isinstance(data.get('subject_id'), str) or not data['subject_id']:
        raise ValueError('location_identity_required')
    loc = data.get('location')
    if not isinstance(loc, dict):
        raise ValueError('invalid_location')
    if loc != {'status': 'unknown'}:
        if not isinstance(loc, dict) or set(loc) != {'status', 'source', 'latitude', 'longitude'} or loc.get('status') != 'available' or loc.get('source') != 'owui_native':
            raise ValueError('invalid_location')
        number(loc['latitude'], -90, 90)
        number(loc['longitude'], -180, 180)
    identity = canonical([data['subject_id'], profile, data['chat_id']])
    return TurnLocation('owui-' + hashlib.sha256(identity).hexdigest()[:32], dict(loc), data['received_at'] + 60)


@dataclass
class TurnLocation:
    session_id: str
    _location: dict = field(repr=False)
    expires_at: float

    def prompt_context(self, *, now=None):
        now = time.time() if now is None else now
        if now > self.expires_at or self._location.get('status') != 'available':
            return POLICY + 'Current device location: UNKNOWN. Ask for city/postcode if needed.'
        return POLICY + ('Native OWUI supplied this request with latitude={latitude}, longitude={longitude}. '
                         'Source: browser geolocation via OWUI. Sensor accuracy and sample timestamp '
                         'are not supplied; do not claim precise GPS or measured freshness.').format(**self._location)

    def clarification(self, text, *, now=None):
        now = time.time() if now is None else now
        if self._location.get('status') == 'available' and now <= self.expires_at:
            return None
        if not isinstance(text, str):
            return None
        normalized = ' '.join(text.lower().strip(' ?.!').split())
        pattern = r"(?:(?:what is|what's|how is|how's|show me|tell me) )?(?:the )?(?:current |local )?(?:weather|forecast)(?: here| near me| in my area| where i am)?(?: today| right now| now)?"
        if re.fullmatch(pattern, normalized):
            return "I don't have your current device location. Please allow location access in your browser, or give me a city or postcode for the weather."
        return None

    def close(self):
        self._location.clear()
