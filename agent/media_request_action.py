"""Trusted API request -> native scripted tools; no backend or rendering owner here."""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3


def eligible(prompt):
    if not isinstance(prompt, str) or len(prompt) > 4096:
        return False
    if not re.match(r'(?i)^\s*(?:please\s+)?(?:generate|draw|render|create)\s+(?:a|an|one)\s+(?:(?:new|anime|photorealistic|realistic)\s+)?(?:image|picture|photo|illustration)\b', prompt):
        return False
    return not re.search(r'(?i)\b(?:edit|modify|upscale|video|animation|attached|reference|previous|latest|original|workflow|install|download|batch)\b|\n|;', prompt)


def build_prelude(agent, *, profile, turn, prompt):
    """turn is ONLY the result of accept_media_turn, never raw request JSON.

    Returns None for ordinary non-eligible work. Native run_conversation retains
    its session lease, persistence, tool grants, hooks and progress callbacks.
    """
    if profile != 'media' or not isinstance(turn, dict) or turn.get('profile') != 'media' or not eligible(prompt):
        return None
    from hermes_constants import get_hermes_home
    from agent.turn_scripted_prelude import PreludeFinal
    key = hashlib.sha256(json.dumps([turn['subject'], turn['chat_id'], turn['user_message_id']], separators=(',', ':')).encode()).hexdigest()
    path = Path(get_hermes_home()) / 'state' / 'media-request-actions.sqlite3'
    # This is a request-effect hold, not a backend job/status store. An existing
    # key (including interrupted/unknown work) is NEVER automatically replayed.
    def hold():
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS media_request_holds (request_key TEXT PRIMARY KEY)')
            try:
                db.execute('INSERT INTO media_request_holds VALUES (?)', (key,))
            except sqlite3.IntegrityError:
                return False
        return True

    def terminal(text):
        return PreludeFinal(text)

    def steps():
        names = {t.get('function', {}).get('name') for t in getattr(agent, 'tools', []) if isinstance(t, dict)}
        if not {'media_match_workflow', 'media_workflows', 'media_generate'} <= names:
            return terminal('The governed Media tools are unavailable; no image was submitted.')
        if os.getenv('HERMES_MEDIA_MATCH_WORKFLOW_AUTO_SUBMIT', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
            return terminal('Media request execution is held because matcher auto-submit is enabled; no retry was issued.')
        # A durable hold precedes even matching, because plugin policy can change
        # across restarts. Storage failure must stop, not fall through to the model.
        try:
            acquired = hold()
        except (OSError, sqlite3.Error):
            return terminal('Media request ownership could not be retained; no automatic submission was attempted.')
        if not acquired:
            return terminal('This image request was already handled or interrupted. Its outcome must be reconciled before another submission.')
        raw = yield ('', 'media_match_workflow', {'prompt': prompt})
        try:
            matched = json.loads(raw)
            match, life, args = matched['match'], matched['job_lifecycle'], matched['media_generate_args']
            if (matched.get('success') is not True or match.get('requires_image') is not False
                    or match.get('needs_input') is not False or match.get('modality') != 'image-generation'
                    or match.get('approval_gate') or life.get('approval_gate')
                    or life.get('resource_class') != 'gpu0-media-image'
                    or life.get('artifact_finalization') != 'async_status_then_artifact_poll'
                    or args.get('wait_for_completion') is not False):
                return terminal('This request is outside the automatic new-image action; no further submission was attempted.')
            # Ordinary-image bound; do not accept model-selected graph overrides.
            inputs = args.get('inputs') or {}
            if any(k in inputs and (type(inputs[k]) is not int or not 64 <= inputs[k] <= 1024) for k in ('width', 'height')):
                return terminal('The requested image dimensions require separate approval; no image was submitted.')
            allowed = {'workflow_id', 'prompt', 'negative_prompt', 'wait_for_completion', 'poll_interval_seconds', 'timeout_seconds'}
            args = {k: v for k, v in args.items() if k in allowed}
            args['governed_defaults'] = True
        except (ValueError, TypeError, KeyError, AttributeError):
            return terminal('Media workflow selection did not return an actionable result; no further submission was attempted.')
        raw = yield ('', 'media_workflows', {'include_unavailable': False})
        try:
            inventory = json.loads(raw)
            selected = next((row for row in inventory['workflows'] if row.get('id') == args['workflow_id']), None)
            if (inventory.get('success') is not True or not selected or selected.get('submit_enabled') is not True
                    or (selected.get('metadata') or {}).get('bounded_standard_image') is not True):
                return terminal('This workflow is not an admitted bounded standard image; automatic submission is held.')
        except (ValueError, TypeError, KeyError, AttributeError):
            return terminal('The admitted image boundary could not be established; no image was submitted.')
        raw = yield ('', 'media_generate', args)
        try:
            generated = json.loads(raw)
            receipt = generated.get('submission', {})
            if (generated.get('success') is True and isinstance(receipt, dict)
                    and isinstance(receipt.get('prompt_id'), str) and receipt['prompt_id']
                    and isinstance(receipt.get('status_url'), str) and receipt['status_url']):
                return terminal('Image generation submitted; the governed result will attach when ready.')
        except (ValueError, TypeError, AttributeError):
            pass
        return terminal('No successful generation receipt was returned. No automatic retry was issued; reconcile the request before submitting again.')
    return steps()
