from pathlib import Path
from typing import Optional
from context_retriever import ContextRetriever

class ContextIntegration:
    def __init__(self):
        self._retriever: Optional[ContextRetriever] = None

    async def get_retriever(self) -> ContextRetriever:
        """Lazy initialization of the context retriever"""
        if self._retriever is None:
            self._retriever = await ContextRetriever.create()
        return self._retriever

    async def get_dynamic_header(self, message_history: str) -> str:
        """Get dynamic context header based on message history"""
        retriever = await self.get_retriever()
        return await retriever.retrieve_context_for_message(message_history)

# Create a singleton instance
context_integration = ContextIntegration()