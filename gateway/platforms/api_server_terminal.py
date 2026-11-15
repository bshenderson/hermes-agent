"""Exact-origin producer dispositions; never evidence of native/global drain."""
import logging

from gateway.platforms.api_server_artifacts import CompletedReceiptStore


class FailureDispositionStore(CompletedReceiptStore):
    """Separate database from successful artifact receipts, same bounded retention."""
    SCHEMA = 'hermes.producer-disposition.v1'
    MAX_RECEIPT_BYTES = 4096
    MAX_TOTAL_BYTES = 1024 * 1024

    def save(self, origin, receipt, *, now=None):
        if (not isinstance(receipt, dict) or receipt.get('status') != 'failed'
                or receipt.get('producer_settled') is not True
                or receipt.get('reason') != 'dispatch_refused'
                or receipt.get('native_retirement') != 'unproven'):
            raise ValueError('Invalid producer failure disposition')
        return super().save(origin, receipt, now=now)


def run_with_disposition(adapter, run, profile, origin):
    """Run in the executor, outside the turn cleanup, not an asyncio done callback.

    Cancellation of the waiting coroutine cannot publish a terminal result. A
    raising worker or unknown result remains unavailable. This narrow successor
    recognizes only the core's explicit non-replay refusal, never error prose.
    """
    outcome = run()
    result, _usage = outcome
    try:
        with adapter._profile_scope(profile):
            retain_failure(result, origin)
    except Exception:
        logging.getLogger(__name__).exception('Producer disposition scope unavailable')
    return outcome


def retain_failure(result, origin):
    """Publish only the core's narrow refusal after the owning lifecycle settles."""
    if (origin and isinstance(result, dict) and result.get('failed') is True
            and result.get('completed') is False and not result.get('partial')
            and result.get('failure_reason') == 'dispatch_refused'
            and result.get('failure_retryable') is False):
        try:
            from hermes_constants import get_hermes_home
            FailureDispositionStore(get_hermes_home() / 'authored-failures.sqlite3').save(origin, {
                'schema': FailureDispositionStore.SCHEMA, 'origin': dict(origin),
                'status': 'failed', 'reason': 'dispatch_refused',
                'producer_settled': True, 'native_retirement': 'unproven'})
        except Exception:
            # Preserve the actual failure; missing durable observation is pending.
            logging.getLogger(__name__).exception('Producer disposition retention unavailable')


def retain_authority_failure(authority, row):
    """Tracked settlement owns publication, not the departing HTTP observer.

    Read the committed result, never delivery/UI text. Unknown or discarded work
    and all other terminal reasons remain unavailable.
    """
    origin = row['payload'].get('api_turn_v1', {}).get('authored_origin')
    if not origin:
        return
    from gateway.session_authorities import owner_scope
    from gateway.session_results import admission_result
    try:
        with owner_scope(authority):
            saved = admission_result(authority.db, row['admission_id'])
            if saved is not None:
                retain_failure(saved['result'], origin)
    except Exception:
        # Retention failure must not strand the canonical result's observers.
        logging.getLogger(__name__).exception('Producer disposition observation unavailable')
