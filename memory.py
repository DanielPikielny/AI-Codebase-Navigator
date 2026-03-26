from __future__ import annotations
import textwrap
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Turn:
    user:      str
    assistant: str
    sources:   list[dict] = field(default_factory=list)

    def summary(self, max_chars: int = 300) -> str:
        """Compact single-string representation for context injection."""
        user_short = textwrap.shorten(self.user,      width=max_chars // 2, placeholder="…")
        asst_short = textwrap.shorten(self.assistant, width=max_chars,      placeholder="…")
        return f"Q: {user_short}\nA: {asst_short}"

class ChatMemory:
    """
    Sliding-window conversation memory.

    Parameters
    ----------
    max_turns : int
        How many past turns to keep in the context window.
        Older turns are dropped (FIFO).  Default 6 is ~3k tokens of context.
    """

    SESSION_KEY = "chat_memory"

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns:  list[Turn] = []

    def add(
        self,
        user:      str,
        assistant: str,
        sources:   Optional[list[dict]] = None,
    ) -> None:
        self.turns.append(Turn(user=user, assistant=assistant, sources=sources or []))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def clear(self) -> None:
        self.turns.clear()

    def as_context(self, include_sources: bool = False) -> str:
        """
        Produce a markdown-formatted conversation history block for prompt injection.
        Recent turns (last 2) are included in full; older turns are summarised.
        FIX-⑨: strips leading/trailing blank lines so the injected block is clean.
        """
        if not self.turns:
            return ""

        n      = len(self.turns)
        blocks = []

        for i, turn in enumerate(self.turns):
            is_recent = i >= n - 2 
            if is_recent:
                parts = [
                    f"**User:** {turn.user}",
                    f"**Assistant:** {turn.assistant}",
                ]
                if include_sources and turn.sources:
                    files = list({s["source"] for s in turn.sources})
                    suffix = "…" if len(files) > 3 else ""
                    parts.append(f"*(Sources: {', '.join(files[:3])}{suffix})*")
                blocks.append("\n".join(parts))
            else:
                blocks.append(turn.summary())

        body = "\n\n".join(blocks).strip() 
        return f"### Conversation history (most recent last)\n\n{body}"

    def referenced_files(self) -> list[str]:
        """Return deduplicated list of source files mentioned in recent turns."""
        seen: dict[str, None] = {}
        for turn in reversed(self.turns):
            for s in turn.sources:
                seen.setdefault(s["source"], None)
        return list(seen.keys())

    def last_user_message(self) -> str:
        return self.turns[-1].user if self.turns else ""

    def is_follow_up(self, query: str) -> bool:
        """
        Heuristic: is this query a follow-up to the previous turn?
        Signals: pronouns, "it"/"that"/"this"/"they", short length, question
        words that imply prior context.
        """
        if not self.turns:
            return False
        FOLLOW_UP_TOKENS = {
            "it", "that", "this", "they", "them", "those", "these",
            "there", "its", "their", "above", "mentioned", "previous",
            "also", "too", "more", "else", "other", "another",
        }
        tokens = set(query.lower().split())
        if tokens & FOLLOW_UP_TOKENS:
            return True
        if len(query.split()) <= 6:
            return True
        return False

    def to_session(self, st_session) -> None:
        """Save this memory object into st.session_state."""
        st_session[self.SESSION_KEY] = self

    @classmethod
    def from_session(cls, st_session, max_turns: int = 6) -> "ChatMemory":
        """
        Load from st.session_state, creating a fresh instance if none exists.
        """
        obj = st_session.get(cls.SESSION_KEY)
        if isinstance(obj, cls):
            obj.max_turns = max_turns
            return obj
        new = cls(max_turns=max_turns)
        st_session[cls.SESSION_KEY] = new
        return new

    def render_history(self, st) -> None:
        """
        Render the full chat history in Streamlit using st.chat_message bubbles.
        Call this before the input widget so history appears above the prompt box.
        """
        for turn in self.turns:
            with st.chat_message("user"):
                st.markdown(turn.user)
            with st.chat_message("assistant"):
                st.markdown(turn.assistant)
                if turn.sources:
                    files = list({s["source"] for s in turn.sources})
                    caption = "Sources: " + ", ".join(
                        f"`{s}`" for s in files[:3]
                    )
                    if len(files) > 3:
                        caption += f" +{len(files) - 3} more"
                    st.caption(caption)