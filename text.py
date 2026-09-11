import os
from typing import Any, Dict, List
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.documents import Document

load_dotenv()

# Load DistilGPT2 for multi-query generation
distilgpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
distilgpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
distilgpt2_tokenizer.pad_token = distilgpt2_tokenizer.eos_token

# Create text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=distilgpt2_model,
    tokenizer=distilgpt2_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)


def generate_intent_analysis(user_query: str) -> str:
    """Generate detailed intent analysis using DistilGPT-2."""
    
    prompt = f"""Analyze this satellite image search query and explain the user's intent.
What are they looking for? What features, patterns, or changes might they want to identify?

User query: {user_query}

Analysis: This search query is looking for satellite imagery that shows"""
    
    try:
        result = text_generator(
            prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=distilgpt2_tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        # Extract the analysis part after "Analysis:"
        if "Analysis:" in generated_text:
            analysis = generated_text.split("Analysis:", 1)[1].strip()
            # Clean up and format
            analysis = "This search query is looking for satellite imagery that shows" + analysis
            return analysis
        
        return generated_text.replace(prompt, "").strip()
    except Exception as e:
        return f"Searching for satellite imagery matching: {user_query}. The query aims to identify relevant geographic features, land use patterns, or structures that match the described criteria."


def generate_three_queries_with_distilgpt2(user_query: str) -> List[str]:
    """Generate exactly 3 optimized search queries using DistilGPT2."""
    
    # Enhanced prompt engineering for better query generation
    prompt = f"""Satellite Image Search - Query Expansion

Original: {user_query}

Generate 3 diverse search variations:

1. """
    
    try:
        result = text_generator(
            prompt,
            max_new_tokens=120,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=distilgpt2_tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        
        # Parse queries
        queries = []
        lines = generated_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered items or list markers
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '•', '-', '*']):
                # Extract the query part
                query = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                if query and len(query) > 10 and query not in queries:
                    queries.append(query)
        
        # Ensure we have exactly 3 quality queries
        if len(queries) < 3:
            # Fallback with smart variations
            base_terms = user_query.lower().split()
            queries = [
                user_query,
                f"satellite view {user_query}",
                f"aerial imagery {' '.join(base_terms[:3])}"
            ]
        
        return queries[:3]
        
    except Exception as e:
        # Robust fallback
        return [
            user_query,
            f"satellite: {user_query}",
            f"aerial: {user_query}"
        ]


def normalize_score(score: Any) -> float:
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def document_to_result(document: Document, score: float, source_query: str) -> Dict[str, Any]:
    return {
        "content": document.page_content,
        "score": normalize_score(score),
        "metadata": document.metadata,
        "source_query": source_query
    }


def retrieve_top_three(query: str, retriever: Any) -> List[Dict[str, Any]]:
    try:
        documents_with_scores = retriever.similarity_search_with_score(query, k=3)
        return [
            document_to_result(document, score, query)
            for document, score in documents_with_scores[:3]
        ]
    except AttributeError:
        documents = retriever.invoke(query)
        return [
            {
                "content": document.page_content,
                "score": 0.0,
                "metadata": document.metadata,
                "source_query": query
            }
            for document in documents[:3]
        ]


def retrieve_all_queries(queries: List[str], retriever: Any) -> List[Dict[str, Any]]:
    all_results = []
    for query in queries:
        all_results.extend(retrieve_top_three(query, retriever))
    return all_results


def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_results = []
    seen = set()

    for result in results:
        metadata = result.get("metadata", {})
        content = result.get("content", "")
        identifier = metadata.get("id") or metadata.get("image_name") or metadata.get("source") or content

        if identifier in seen:
            continue

        seen.add(identifier)
        unique_results.append(result)

    return unique_results


def select_top_seven(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select top 7 results after deduplication."""
    unique_results = deduplicate_results(results)
    return sorted(unique_results, key=lambda x: normalize_score(x.get("score")), reverse=True)[:7]


def run_text_pipeline(user_query: str, retriever: Any) -> Dict[str, Any]:
    """Run complete text pipeline and return final results."""
    
    # Generate intent analysis with DistilGPT2
    intent = generate_intent_analysis(user_query)
    
    # Generate 3 queries with DistilGPT2
    queries = generate_three_queries_with_distilgpt2(user_query)
    
    # Retrieve 3 images per query
    all_results = []
    for query in queries:
        results = retrieve_top_three(query, retriever)
        all_results.extend(results)
    
    # Select top 7
    final_results = select_top_seven(all_results)
    
    # Format results
    formatted_results = []
    for r in final_results:
        metadata = r.get("metadata", {})
        formatted_results.append({
            "image_name": metadata.get("image_name", ""),
            "score": r.get("score", 0.0)
        })
    
    return {
        "user_query": user_query,
        "intent": intent,
        "queries": queries,
        "results": formatted_results
    }


def stream_text_pipeline(user_query: str, retriever: Any):
    """Stream text search pipeline with DistilGPT-2 for both intent and queries."""
    
    # Stage 1: Understanding
    yield {"type": "stage", "data": "Understanding your search query..."}
    
    # Stage 2: Generate intent analysis with DistilGPT-2
    yield {"type": "stage", "data": "AI is analyzing your intent..."}
    yield {"type": "thinking_start", "data": "Starting analysis..."}
    
    try:
        intent_text = generate_intent_analysis(user_query)
        yield {"type": "thinking_content", "data": intent_text}
    except Exception as e:
        yield {"type": "thinking_content", "data": f"Analyzing: {user_query}"}
    
    # Stage 3: Multi-query generation with DistilGPT2
    yield {"type": "stage", "data": "Generating 3 search variations..."}
    
    try:
        queries = generate_three_queries_with_distilgpt2(user_query)
        yield {"type": "queries", "data": queries}
    except Exception as e:
        yield {"type": "error", "data": f"Query generation failed: {str(e)}"}
        queries = [user_query] * 3
    
    # Stage 4: Retrieve 3 images per query
    yield {"type": "stage", "data": "Retrieving 3 images per query..."}
    
    all_results = []
    for i, query in enumerate(queries, 1):
        yield {"type": "query_variation", "data": {"index": i, "query": query}}
        try:
            results = retrieve_top_three(query, retriever)
            all_results.extend(results)
            yield {"type": "partial_results", "data": {"query_num": i, "count": len(results)}}
        except Exception as e:
            yield {"type": "error", "data": f"Retrieval failed for query {i}: {str(e)}"}
    
    # Stage 5: Select top 7
    yield {"type": "stage", "data": "Selecting top 7 best matches..."}
    
    final_results = select_top_seven(all_results)
    
    # Stage 6: Results ready
    yield {"type": "stage", "data": "Search complete"}
    yield {"type": "results", "data": final_results}