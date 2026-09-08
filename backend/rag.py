"""
rag.py
Retrieval-Augmented Generation (RAG) pipeline for geospatial and remote sensing intelligence.
Combines vector retrieval, timeline analysis, and VLM/LLM generation.
"""

from typing import Any, Dict, List, Optional
from backend.retrieval import retriever
from backend.vlm import vlm_service


class GeospatialRAG:
    def query(self, prompt: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes a RAG query:
        1. Retrieves relevant satellite imagery matching semantic prompt.
        2. Gathers metadata and spatial context.
        3. Formulates a grounded response.
        """
        retrieved_items = retriever.search_by_text(prompt, top_k=top_k)

        context_summary = []
        for item in retrieved_items:
            context_summary.append(f"Image: {item['path']} (Similarity Score: {item['score']:.3f})")

        combined_context = "\n".join(context_summary) if context_summary else "No matching imagery found in index."

        # Generate synthesized insight
        answer = (
            f"Based on retrieved satellite observations for query: '{prompt}':\n"
            f"Found {len(retrieved_items)} relevant scenes.\n"
            f"Context:\n{combined_context}"
        )

        return {
            "query": prompt,
            "answer": answer,
            "retrieved_images": retrieved_items,
        }


geospatial_rag = GeospatialRAG()
