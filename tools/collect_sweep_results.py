#!/usr/bin/env python
"""
Collect and summarize results from all basin structure sweeps.

Usage:
    python collect_sweep_results.py [--output results_summary.csv]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

# Base directories for each sweep
SWEEP_DIRS = {
    "structured_excl": "/network/scratch/l/lia/skae/structured_excl_sweep_lyapunov",
    "structured_entropy": "/network/scratch/l/lia/skae/structured_entropy_sweep_lyapunov",
    "structured_large": "/network/scratch/l/lia/skae/structured_large_dims_lyapunov",
    "unstructured_lista": "/network/scratch/l/lia/skae/unstructured_lista_lyapunov",
    "generic_baseline": "/network/scratch/l/lia/skae/generic_baseline_lyapunov",
}


def find_results_file(exp_dir: Path) -> Path | None:
    """Find the analysis results JSON file in an experiment directory."""
    # Check basin_eval for StructuredLISTAKM
    basin_eval = exp_dir / "basin_eval" / "analysis_results.json"
    if basin_eval.exists():
        return basin_eval

    # Check latent_eval for other models
    latent_eval = exp_dir / "latent_eval" / "analysis_results.json"
    if latent_eval.exists():
        return latent_eval

    # Check basin_structure_analysis (from train.py)
    basin_struct = list(exp_dir.glob("*/basin_structure_analysis/analysis_results.json"))
    if basin_struct:
        return basin_struct[0]

    # Check evaluation directories
    eval_dirs = list(exp_dir.glob("*/evaluation_*/"))
    for eval_dir in eval_dirs:
        results_file = eval_dir.parent / "basin_structure_analysis" / "analysis_results.json"
        if results_file.exists():
            return results_file

    return None


def load_config(exp_dir: Path) -> Dict[str, Any]:
    """Load config from experiment directory."""
    config_files = list(exp_dir.glob("*/config.json"))
    if config_files:
        with open(config_files[0]) as f:
            return json.load(f)
    return {}


def collect_results(sweep_name: str, sweep_dir: str) -> List[Dict[str, Any]]:
    """Collect results from all experiments in a sweep directory."""
    results = []
    sweep_path = Path(sweep_dir)

    if not sweep_path.exists():
        print(f"  Directory not found: {sweep_dir}")
        return results

    # Find all experiment subdirectories
    exp_dirs = [d for d in sweep_path.iterdir() if d.is_dir()]

    for exp_dir in sorted(exp_dirs):
        exp_name = exp_dir.name
        results_file = find_results_file(exp_dir)

        if results_file is None:
            print(f"  No results found for: {exp_name}")
            continue

        try:
            with open(results_file) as f:
                analysis = json.load(f)

            config = load_config(exp_dir)

            # Extract key metrics
            result = {
                "sweep": sweep_name,
                "experiment": exp_name,
                "model_type": config.get("MODEL", {}).get("MODEL_NAME", "unknown"),
                "target_size": config.get("MODEL", {}).get("TARGET_SIZE", 0),
            }

            # Structured config
            struct_cfg = config.get("MODEL", {}).get("STRUCTURED", {})
            if struct_cfg.get("ENABLED", False):
                result["d_global"] = struct_cfg.get("D_GLOBAL", 0)
                result["d_basin"] = struct_cfg.get("D_BASIN", 0)
                result["num_basins"] = struct_cfg.get("NUM_BASINS", 0)
                result["lambda_excl"] = struct_cfg.get("LAMBDA_EXCLUSIVITY", 0)
                result["lambda_entropy"] = struct_cfg.get("LAMBDA_ENTROPY", 0)
                result["lambda_dominance"] = struct_cfg.get("LAMBDA_DOMINANCE", 0)

            # LISTA config
            lista_cfg = config.get("MODEL", {}).get("ENCODER", {}).get("LISTA", {})
            result["lista_alpha"] = lista_cfg.get("ALPHA", 0)

            # Sparsity
            result["sparsity_coeff"] = config.get("MODEL", {}).get("SPARSITY_COEFF", 0)

            # Results metrics
            result["basin_accuracy"] = analysis.get("basin_assignment_accuracy",
                                                     analysis.get("linear_classifier_accuracy", 0))
            result["temporal_consistency"] = analysis.get("temporal_consistency", 0)
            result["activation_entropy"] = analysis.get("mean_activation_entropy", 0)
            result["silhouette_score"] = analysis.get("silhouette_score", 0)
            result["adjusted_rand_index"] = analysis.get("adjusted_rand_index", 0)
            result["sparsity_ratio"] = analysis.get("sparsity_ratio", 0)
            result["final_pred_error"] = analysis.get("final_pred_error", 0)

            results.append(result)
            print(f"  {exp_name}: accuracy={result['basin_accuracy']:.4f}")

        except Exception as e:
            print(f"  Error loading {exp_name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect sweep results")
    parser.add_argument("--output", type=str, default="results_summary.csv",
                        help="Output CSV file")
    args = parser.parse_args()

    all_results = []

    for sweep_name, sweep_dir in SWEEP_DIRS.items():
        print(f"\n=== {sweep_name} ===")
        results = collect_results(sweep_name, sweep_dir)
        all_results.extend(results)

    if not all_results:
        print("\nNo results found!")
        return

    # Sort by accuracy
    all_results.sort(key=lambda x: x.get("basin_accuracy", 0), reverse=True)

    # Print summary
    print("\n" + "=" * 80)
    print("TOP 10 RESULTS BY BASIN ACCURACY")
    print("=" * 80)

    for i, r in enumerate(all_results[:10]):
        print(f"{i+1}. {r['sweep']}/{r['experiment']}")
        print(f"   Model: {r['model_type']}, Dim: {r['target_size']}")
        print(f"   Basin Accuracy: {r['basin_accuracy']:.4f}")
        if r.get('d_basin'):
            print(f"   d_global={r.get('d_global')}, d_basin={r.get('d_basin')}, B={r.get('num_basins')}")
        print()

    # Save to CSV
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    except ImportError:
        # Fallback: save as JSON
        output_json = args.output.replace('.csv', '.json')
        with open(output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_json} (pandas not available)")


if __name__ == "__main__":
    main()
