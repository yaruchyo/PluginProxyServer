from enum import Enum


class SSEConstants(Enum):
    """Constants related to Server-Sent Events (SSE) formatting."""
    DATA_PREFIX = "data: "
    DONE_MESSAGE_SUFFIX = "[DONE]"
    CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"
    TEXT_COMPLETION_OBJECT = "text_completion"  # Assuming this is the identifier for non-chat format

    @property
    def DONE_MESSAGE(self) -> str:
        """Returns the full SSE DONE message string."""
        # We define DONE_MESSAGE as a property to combine prefix and suffix
        # This avoids having the prefix hardcoded within the DONE message itself
        if self == SSEConstants.DONE_MESSAGE_SUFFIX:
             return f"{SSEConstants.DATA_PREFIX.value}{self.value}\n\n"
        return self.value # Should not happen, but prevents errors

    @classmethod
    def get_done_message(cls) -> str:
        """Class method to get the formatted DONE message."""
        return f"{cls.DATA_PREFIX.value}{cls.DONE_MESSAGE_SUFFIX.value}\n\n"