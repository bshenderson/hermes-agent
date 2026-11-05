"""Internal verified-chat provenance for native history read after turn admission."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OwnedChatHistory:
    root_session_id: str
    profile: str
    subject: str
    chat: str

    def load_after_admission(self, db, session_id, holder):
        """Never move to another tip while holding a lease for the old session."""
        expected = (self.profile, self.subject, self.chat)

        def validate():
            if db.get_compression_tip(self.root_session_id) != session_id:
                raise ValueError('Owned archive chat rotated before admission')
            with db._read_ctx() as conn:
                row = conn.execute(
                    'SELECT profile_name,user_id,chat_id,ended_at FROM sessions WHERE id=?',
                    (session_id,)).fetchone()
                if row is None or tuple(row[:3]) != expected or row[3] is not None:
                    raise ValueError('Owned archive chat owner or live state changed')
            expiry = db.session_turn_lease_expires_at(session_id, holder)
            import time
            if expiry is None or expiry <= time.time():
                raise ValueError('Owned archive chat lease no longer held')

        validate()
        history = db.get_messages_as_conversation(session_id)
        validate()
        return history
