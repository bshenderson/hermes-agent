"""Runtime status payload initialization; mutations remain under the facade lock."""
import copy


def prepare_payload(previous, identity, clear_profiles, drop_profile, keep_platforms):
    from gateway import status

    payload = copy.deepcopy(previous)
    payload.setdefault('platforms', {})
    if not isinstance(payload['platforms'], dict):
        payload['platforms'] = {}
    if clear_profiles or drop_profile:
        drop_prefix = f'{drop_profile}:' if drop_profile else None
        payload['platforms'] = {
            k: v for k, v in payload['platforms'].items()
            if not isinstance(k, str) or ':' not in k
            or (drop_prefix is not None and not k.startswith(drop_prefix)) or k in keep_platforms
        }
    payload.update({key: identity[key] for key in ('kind', 'pid', 'argv', 'start_time')})
    payload['updated_at'] = status._utc_now_iso()
    payload.update(status._get_code_identity_fields())
    return payload
