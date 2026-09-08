"""
TerraWatch AI - Evaluation Framework for Semantic Retrieval

This module provides a fair, reproducible evaluation framework for comparing
different CLIP models on satellite image semantic retrieval tasks.

Created for: OpenCLIP vs RemoteCLIP comparison experiment
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np
from collections import defaultdict

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# SATELLITE IMAGERY SEMANTIC QUERY CATEGORIES
# ============================================================

EVALUATION_QUERIES = {
    "water": [
        "water body reservoir lake",
        "deep water lake",
        "river and water",
        "ocean coastal water",
    ],
    "urban": [
        "urban city area with buildings",
        "dense residential urban area",
        "city buildings and roads",
        "urban infrastructure",
    ],
    "agriculture": [
        "agricultural farmland cropland",
        "fields and farms",
        "agricultural cultivation area",
        "crop fields",
    ],
    "forest": [
        "forest vegetation trees",
        "dense forest area",
        "woodland with trees",
        "forested region",
    ],
    "infrastructure": [
        "airport with runway",
        "highway and roads",
        "industrial facility",
        "port and harbor",
    ],
    "terrain": [
        "mountains and terrain",
        "desert arid land",
        "snow covered area",
        "bare soil land",
    ],
}


def get_all_queries() -> List[str]:
    """Return flattened list of all evaluation queries."""
    queries = []
    for category_queries in EVALUATION_QUERIES.values():
        queries.extend(category_queries)
    return queries


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_recall_at_k(
    query_results: List[Dict[str, Any]],
    relevant_image_ids: List[str],
    k: int,
) -> float:
    """
    Calculate Recall@K metric.
    
    Recall@K = Number of relevant items in top-K / Total relevant items
    
    Args:
        query_results: List of search results with image_id and score
        relevant_image_ids: Ground truth relevant image IDs for this query
        k: Number of top results to consider
    
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_image_ids:
        return 0.0
    
    # Get top-K retrieved image IDs
    top_k_ids = [r["image_id"] for r in query_results[:k]]
    
    # Count how many relevant images were retrieved
    retrieved_relevant = len(set(top_k_ids) & set(relevant_image_ids))
    
    # Recall = retrieved relevant / total relevant
    return retrieved_relevant / len(relevant_image_ids)


def calculate_precision_at_k(
    query_results: List[Dict[str, Any]],
    relevant_image_ids: List[str],
    k: int,
) -> float:
    """
    Calculate Precision@K metric.
    
    Precision@K = Number of relevant items in top-K / K
    
    Args:
        query_results: List of search results with image_id and score
        relevant_image_ids: Ground truth relevant image IDs for this query
        k: Number of top results to consider
    
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    # Get top-K retrieved image IDs
    top_k_ids = [r["image_id"] for r in query_results[:k]]
    
    # Count how many retrieved images are relevant
    retrieved_relevant = len(set(top_k_ids) & set(relevant_image_ids))
    
    # Precision = retrieved relevant / K
    return retrieved_relevant / k


def calculate_average_precision(
    query_results: List[Dict[str, Any]],
    relevant_image_ids: List[str],
) -> float:
    """
    Calculate Average Precision (AP) for a single query.
    
    AP is the average of precision values at each relevant document position.
    """
    if not relevant_image_ids:
        return 0.0
    
    relevant_set = set(relevant_image_ids)
    
    precisions = []
    num_relevant_seen = 0
    
    for i, result in enumerate(query_results):
        if result["image_id"] in relevant_set:
            num_relevant_seen += 1
            precision_at_i = num_relevant_seen / (i + 1)
            precisions.append(precision_at_i)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(relevant_image_ids)


# ============================================================
# EVALUATION RUNNER
# ============================================================

class RetrievalEvaluator:
    """
    Evaluates semantic retrieval performance using standard IR metrics.
    """
    
    def __init__(
        self,
        retriever_func,
        queries: List[str] = None,
        k_values: List[int] = None,
    ):
        """
        Args:
            retriever_func: Function that takes (query, top_k) and returns ranked results
            queries: List of evaluation queries (default: all categories)
            k_values: List of K values for Recall@K (default: [1, 5, 10])
        """
        self.retriever_func = retriever_func
        self.queries = queries or get_all_queries()
        self.k_values = k_values or [1, 5, 10]
    
    def evaluate(
        self,
        ground_truth: Dict[str, List[str]],
        max_top_k: int = 50,
    ) -> Dict[str, Any]:
        """
        Run evaluation on all queries.
        
        Args:
            ground_truth: Dict mapping query -> list of relevant image IDs
            max_top_k: Maximum number of results to retrieve per query
        
        Returns:
            Dictionary with evaluation metrics and per-query results
        """
        results = {
            "num_queries": len(self.queries),
            "k_values": self.k_values,
            "per_query_results": [],
            "aggregated_metrics": {},
        }
        
        # Metrics storage
        recall_at_k = defaultdict(list)
        precision_at_k = defaultdict(list)
        average_precisions = []
        
        # Evaluate each query
        for query in self.queries:
            # Skip queries without ground truth
            if query not in ground_truth:
                continue
            
            relevant_ids = ground_truth[query]
            
            # Retrieve results
            search_results = self.retriever_func(query, top_k=max_top_k)
            
            # Calculate metrics
            query_metrics = {"query": query, "num_relevant": len(relevant_ids)}
            
            for k in self.k_values:
                recall = calculate_recall_at_k(search_results, relevant_ids, k)
                precision = calculate_precision_at_k(search_results, relevant_ids, k)
                
                recall_at_k[k].append(recall)
                precision_at_k[k].append(precision)
                
                query_metrics[f"recall@{k}"] = recall
                query_metrics[f"precision@{k}"] = precision
            
            # Average Precision
            ap = calculate_average_precision(search_results, relevant_ids)
            average_precisions.append(ap)
            query_metrics["average_precision"] = ap
            
            results["per_query_results"].append(query_metrics)
        
        # Aggregate metrics across all queries
        aggregated = {}
        
        for k in self.k_values:
            if recall_at_k[k]:
                aggregated[f"recall@{k}"] = np.mean(recall_at_k[k])
                aggregated[f"precision@{k}"] = np.mean(precision_at_k[k])
        
        if average_precisions:
            aggregated["mean_average_precision"] = np.mean(average_precisions)
        
        results["aggregated_metrics"] = aggregated
        
        return results


# ============================================================
# AUTOMATIC PSEUDO-LABELING (for experiments without ground truth)
# ============================================================

def create_pseudo_labels_from_filenames(
    image_paths: List[Path],
    queries: List[str],
) -> Dict[str, List[str]]:
    """
    Create pseudo ground truth by matching query keywords to image filenames.
    
    This is a HEURISTIC approach for experiments where no ground truth exists.
    It assumes image filenames or metadata contain semantic hints.
    
    For TerraWatch Sentinel-2 dataset (generic S2_XXXXXX.jpg names),
    this will return empty labels - user must provide real ground truth.
    
    Returns:
        Dict mapping query -> list of potentially relevant image IDs
    """
    ground_truth = defaultdict(list)
    
    # Extract image IDs
    image_ids = [p.stem for p in image_paths]
    
    # Try to match queries to filenames (very basic heuristic)
    for query in queries:
        keywords = query.lower().split()
        
        for image_id, path in zip(image_ids, image_paths):
            filename_lower = path.name.lower()
            
            # If any keyword appears in filename, consider it relevant
            if any(keyword in filename_lower for keyword in keywords):
                ground_truth[query].append(image_id)
    
    return dict(ground_truth)


# ============================================================
# SAMPLE IMAGE SELECTION
# ============================================================

def select_evaluation_sample(
    image_paths: List[Path],
    sample_size: int = 100,
    random_seed: int = 42,
) -> List[Path]:
    """
    Select a deterministic random sample of images for evaluation.
    
    Args:
        image_paths: Full list of image paths
        sample_size: Number of images to sample
        random_seed: Fixed seed for reproducibility
    
    Returns:
        Sampled list of image paths
    """
    rng = np.random.RandomState(random_seed)
    
    if len(image_paths) <= sample_size:
        return image_paths
    
    indices = rng.choice(len(image_paths), size=sample_size, replace=False)
    indices = sorted(indices)  # Keep deterministic order
    
    return [image_paths[i] for i in indices]


# ============================================================
# UTILITY: SAVE RESULTS
# ============================================================

def save_evaluation_results(
    results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def print_evaluation_summary(results: Dict[str, Any], model_name: str) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "=" * 70)
    print(f"{model_name.upper()} - EVALUATION SUMMARY")
    print("=" * 70)
    
    metrics = results.get("aggregated_metrics", {})
    
    print(f"\nQueries evaluated: {results.get('num_queries', 0)}")
    
    print("\nRetrieval Metrics:")
    print("-" * 40)
    
    for k in results.get("k_values", []):
        recall = metrics.get(f"recall@{k}", 0.0)
        precision = metrics.get(f"precision@{k}", 0.0)
        print(f"  Recall@{k:<2}    : {recall:.4f} ({recall*100:.2f}%)")
        print(f"  Precision@{k:<2} : {precision:.4f} ({precision*100:.2f}%)")
    
    map_score = metrics.get("mean_average_precision", 0.0)
    print(f"\n  Mean Average Precision (mAP): {map_score:.4f} ({map_score*100:.2f}%)")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("Evaluation framework loaded successfully!")
    print(f"\nAvailable query categories: {list(EVALUATION_QUERIES.keys())}")
    print(f"Total evaluation queries: {len(get_all_queries())}")
