"""Internal verified-chat provenance for native history read after turn admission.

Everything here goes through the profile's storage interface handle: the rooted tip, the owner
row and the lease are read from the store, never from a handle the caller happens to hold.
"""
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class OwnedChatHistory:
    root_session_id: str
    profile: str
    subject: str
    chat: str

    def load_after_admission(self, store, session_id, holder):
        """Never move to another tip while holding a lease for the old session."""
        from persistence.conversation.messages import ConversationRequest
        from persistence.conversation.sessions import SessionId

        expected = (self.profile, self.subject, self.chat)
        sid = SessionId(session_id)

        def validate():
            if store.lineage.tip(self.root_session_id) != session_id:
                raise ValueError('Owned archive chat rotated before admission')
            row = store.sessions.get(sid)
            if row is None or (row.profile_name, row.user_id, row.chat_id) != expected or row.ended_at is not None:
                raise ValueError('Owned archive chat owner or live state changed')
            expiry = store.coordination.turn_lease_expires_at(session_id=sid, holder=holder)
            if expiry is None or expiry <= time.time():
                raise ValueError('Owned archive chat lease no longer held')

        validate()
        # Same replay projection the ordinary history loader uses: repaired alternation with the
        # compaction summary markers kept, so a rotated handoff stays typed.
        history = [message.as_openai_message()
                   for message in store.messages.read_conversation(sid, ConversationRequest(repair_alternation=True))]
        validate()
        return history
