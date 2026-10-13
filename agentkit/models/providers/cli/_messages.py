"""Message boundaries shared by the CLI stream folds."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class MessageLog:
    messages: list[str] = field(default_factory=list)
    ids: list[str | None] = field(default_factory=list)
    seen: set[tuple[str, str]] = field(default_factory=set)

    def record(self, text: str, message_id: str | None, *, append: bool = False) -> str:
        if message_id is not None:
            key = (message_id, text)
            if key in self.seen:
                return self.messages[-1] if self.messages else ""
            self.seen.add(key)
        if append and message_id is not None and self.ids and self.ids[-1] == message_id:
            self.messages[-1] += text
        else:
            self.ids.append(message_id)
            self.messages.append(text)
        return self.messages[-1]
