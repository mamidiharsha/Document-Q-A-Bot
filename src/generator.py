"""
Answer Generation module for the RAG pipeline.

Uses Google Gemini LLM to generate grounded answers from retrieved
context, with strict instructions to cite sources and avoid
hallucination.
"""

from dataclasses import dataclass, field
from typing import Optional

from google import genai

import os

from src.config import GOOGLE_API_KEY, GEMINI_MODEL


# ---------------------------------------------------------------------------
# System Prompt — The core of grounded answer generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable and precise document Q&A assistant. Your role is to answer questions based ONLY on the provided context from the document collection.

## Rules you MUST follow:

1. **Answer ONLY from the provided context.** Do not use any information from your training data or general knowledge. If the answer is not contained in the context, say so clearly.

2. **Cite your sources.** For every claim or fact in your answer, include a citation in the format [Source N] where N corresponds to the source number in the context. Place citations inline, immediately after the relevant statement.

3. **Be specific and accurate.** Quote or closely paraphrase the relevant text from the context. Do not generalize or add information that is not present.

4. **If the context does not contain enough information to answer the question**, respond with:
   "I don't have enough information in the provided documents to answer this question. The documents in my knowledge base cover topics including [list the topics you can see in the sources], but they don't contain information about [the user's topic]."

5. **Format your response clearly.** Use paragraphs for readability. Start with a direct answer, then provide supporting details with citations.

6. **At the end of your response**, include a "Sources" section listing each source you cited with its filename and page/section number."""


@dataclass
class GeneratedAnswer:
    """Represents a generated answer with its sources."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    model: str = ""


class AnswerGenerator:
    """Generates grounded answers using Google Gemini LLM.

    The generator constructs a strict prompt that forces the LLM to:
    - Only answer from provided context
    - Include source citations
    - Gracefully handle unanswerable questions

    Args:
        api_key: Google API key. If None, uses the key from config.
        model: Gemini model name. Defaults to config value.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
    ):
        self.model = model
        # Allow a local generator fallback for offline testing
        self.use_local = os.getenv("USE_LOCAL_GENERATOR", "0") == "1" or not (
            api_key or GOOGLE_API_KEY
        )

        if not self.use_local:
            self.client = genai.Client(api_key=api_key or GOOGLE_API_KEY)
        else:
            self.client = None

    def generate(
        self,
        query: str,
        context: str,
        sources: list[dict],
    ) -> GeneratedAnswer:
        """Generate a grounded answer from retrieved context.

        Args:
            query: The user's question.
            context: Formatted context string from the retriever.
            sources: List of source metadata dicts from the retriever.

        Returns:
            GeneratedAnswer with the answer text and source references.
        """
        # Build the user prompt with context
        user_prompt = self._build_prompt(query, context)

        # Call the Gemini API
        if self.use_local:
            # Simple local-generation fallback: return the retrieved context
            # with explicit source citations for offline testing.
            answer_lines = []
            answer_lines.append("Local offline answer (no LLM):")
            answer_lines.append("")
            # Provide a short synthesized response by quoting the top sources
            for s in sources[:3]:
                idx = s.get("index")
                src = s.get("source")
                excerpt = s.get("excerpt", "")
                answer_lines.append(f"[Source {idx}] {src}: {excerpt}")

            answer_lines.append("")
            answer_lines.append(
                "I don't have a cloud LLM available, so this offline response shows the retrieved context and sources."
            )

            return GeneratedAnswer(
                answer="\n\n".join(answer_lines),
                sources=sources,
                query=query,
                model="local-mock",
            )

        # Call the Gemini API
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.2,  # Low temperature for factual accuracy
                "max_output_tokens": 2048,
            },
        )

        answer_text = response.text if response.text else (
            "I was unable to generate a response. Please try rephrasing your question."
        )

        return GeneratedAnswer(
            answer=answer_text,
            sources=sources,
            query=query,
            model=self.model,
        )

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the user prompt combining the query and context.

        Args:
            query: The user's question.
            context: The formatted context from retrieved chunks.

        Returns:
            Complete prompt string for the LLM.
        """
        return f"""## Retrieved Context

{context}

## Question

{query}

## Instructions

Answer the question above using ONLY the information provided in the Retrieved Context. Remember to cite your sources using [Source N] format and include a Sources section at the end."""
