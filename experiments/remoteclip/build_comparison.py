# Script to build the comparison experiment
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# This will create the full comparison script
print("Building comparison script...")

script_content = '''"""
OpenCLIP vs RemoteCLIP Comparison Experiment
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# Import services
from backend.embeddings import embedding_service as openclip_service
from experiments.remoteclip.remoteclip_embeddings import remoteclip_service
from experiments.remoteclip.evaluation_framework import (
    get_all_queries,
    select_evaluation_sample,
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_average_precision,
)

SAMPLE_SIZE = 100
RANDOM_SEED = 42
K_VALUES = [1, 5, 10]
MAX_TOP_K = 50
RESULTS_DIR = BASE_DIR / "results" / "remoteclip"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_ground_truth():
    """Generate ground truth using model consensus."""
    print("\\nGenerating ground truth...")
    images_dir = BASE_DIR / "data" / "processed" / "images"
    all_images = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    sample_images = select_evaluation_sample(all_images, SAMPLE_SIZE, RANDOM_SEED)
    image_ids = [img.stem for img in sample_images]
    print(f"Sample: {len(sample_images)} images")
    
    print("Loading models...")
    openclip_service._lazy_load()
    remoteclip_service._lazy_load()
    
    print("Generating embeddings...")
    oc_embs, rc_embs = [], []
    for img_path in tqdm(sample_images, desc="Encoding"):
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            oc_embs.append(openclip_service.embed_image(img_rgb))
            rc_embs.append(remoteclip_service.embed_image(img_rgb))
    
    oc_embs = np.vstack(oc_embs)
    rc_embs = np.vstack(rc_embs)
    
    queries = get_all_queries()
    ground_truth = {}
    
    print(f"Processing {len(queries)} queries...")
    for query in tqdm(queries, desc="Queries"):
        oc_text = openclip_service.embed_text(query)
        rc_text = remoteclip_service.embed_text(query)
        oc_scores = np.dot(oc_embs, oc_text.T).flatten()
        rc_scores = np.dot(rc_embs, rc_text.T).flatten()
        oc_top10 = set(np.argsort(oc_scores)[::-1][:10])
        rc_top10 = set(np.argsort(rc_scores)[::-1][:10])
        oc_top5 = set(np.argsort(oc_scores)[::-1][:5])
        rc_top5 = set(np.argsort(rc_scores)[::-1][:5])
        relevant_idx = (oc_top10 & rc_top10) | oc_top5 | rc_top5
        ground_truth[query] = [image_ids[i] for i in relevant_idx]
    
    return ground_truth, sample_images, oc_embs, rc_embs

def evaluate_model(name, q_embs, img_embs, img_ids, gt, queries):
    """Evaluate a model."""
    print(f"\\n{'='*70}\\nEVALUATING {name.upper()}\\n{'='*70}")
    results = {"model": name, "timestamp": datetime.now().isoformat(), "num_queries": len(queries), "num_images": len(img_ids), "k_values": K_VALUES}
    recall_k, prec_k, aps = defaultdict(list), defaultdict(list), []
    start_time = time.time()
    
    for i, query in enumerate(queries):
        if query not in gt: continue
        relevant = gt[query]
        q_emb = q_embs[i:i+1]
        scores = np.dot(img_embs, q_emb.T).flatten()
        ranked = np.argsort(scores)[::-1][:MAX_TOP_K]
        search_res = [{"image_id": img_ids[idx], "score": float(scores[idx])} for idx in ranked]
        for k in K_VALUES:
            recall_k[k].append(calculate_recall_at_k(search_res, relevant, k))
            prec_k[k].append(calculate_precision_at_k(search_res, relevant, k))
        aps.append(calculate_average_precision(search_res, relevant))
    
    agg = {}
    for k in K_VALUES:
        if recall_k[k]:
            agg[f"recall@{k}"] = float(np.mean(recall_k[k]))
            agg[f"precision@{k}"] = float(np.mean(prec_k[k]))
    if aps: agg["mean_average_precision"] = float(np.mean(aps))
    results["aggregated_metrics"] = agg
    results["evaluation_time"] = time.time() - start_time
    
    print(f"\\nMetrics:")
    for k in K_VALUES:
        print(f"  Recall@{k}: {agg.get(f'recall@{k}', 0):.4f}")
        print(f"  Precision@{k}: {agg.get(f'precision@{k}', 0):.4f}")
    print(f"  mAP: {agg.get('mean_average_precision', 0):.4f}")
    print(f"Time: {results['evaluation_time']:.2f}s")
    return results

def compare(oc_res, rc_res):
    """Generate comparison."""
    print("\\n\\n" + "="*70 + "\\nOPENCLIP VS REMOTECLIP COMPARISON\\n" + "="*70)
    oc_m, rc_m = oc_res["aggregated_metrics"], rc_res["aggregated_metrics"]
    print(f"\\nImages: {oc_res['num_images']}, Queries: {oc_res['num_queries']}")
    print("\\n" + "-"*70)
    print(f"{'Metric':<30} {'OpenCLIP':<15} {'RemoteCLIP':<15} {'Diff':<15}")
    print("-"*70)
    wins = {"OpenCLIP": 0, "RemoteCLIP": 0}
    for k in K_VALUES:
        oc_r = oc_m.get(f"recall@{k}", 0)
        rc_r = rc_m.get(f"recall@{k}", 0)
        diff = rc_r - oc_r
        if diff > 0.001: wins["RemoteCLIP"] += 1
        elif diff < -0.001: wins["OpenCLIP"] += 1
        print(f"Recall@{k:<25} {oc_r:.4f}         {rc_r:.4f}          {diff:+.4f}")
    print("-"*70)
    oc_map = oc_m.get("mean_average_precision", 0)
    rc_map = rc_m.get("mean_average_precision", 0)
    print(f"mAP                           {oc_map:.4f}         {rc_map:.4f}          {rc_map-oc_map:+.4f}")
    print("-"*70)
    if wins["RemoteCLIP"] > wins["OpenCLIP"]: print(f"\\nWINNER: RemoteCLIP (won {wins['RemoteCLIP']} metrics)")
    elif wins["OpenCLIP"] > wins["RemoteCLIP"]: print(f"\\nWINNER: OpenCLIP (won {wins['OpenCLIP']} metrics)")
    else: print("\\nRESULT: TIE")
    print("="*70)

def main():
    print("\\nOPENCLIP VS REMOTECLIP EXPERIMENT\\n")
    gt, imgs, oc_e, rc_e = generate_ground_truth()
    img_ids = [i.stem for i in imgs]
    queries = get_all_queries()
    print("\\nEncoding queries...")
    oc_q = openclip_service.embed_text(queries)
    rc_q = remoteclip_service.embed_text(queries)
    oc_res = evaluate_model("OpenCLIP", oc_q, oc_e, img_ids, gt, queries)
    rc_res = evaluate_model("RemoteCLIP", rc_q, rc_e, img_ids, gt, queries)
    compare(oc_res, rc_res)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"openclip_{ts}.json").write_text(json.dumps(oc_res, indent=2))
    (RESULTS_DIR / f"remoteclip_{ts}.json").write_text(json.dumps(rc_res, indent=2))
    comp = {"timestamp": ts, "sample": SAMPLE_SIZE, "seed": RANDOM_SEED, "openclip": oc_res["aggregated_metrics"], "remoteclip": rc_res["aggregated_metrics"]}
    (RESULTS_DIR / f"comparison_{ts}.json").write_text(json.dumps(comp, indent=2))
    print(f"\\nResults saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
'''

output = Path('experiments/remoteclip/compare_models.py')
output.write_text(script_content, encoding='utf-8')
print(f'Created: {output}')
