"""
CLI commands for SAE research tools.
"""

import json
import fire
from sae_research.sae_bench_utils import find_saes


def find_sae_bench_models(
    repo_id: str,
    model_name: str | None = None,
    release: str | None = None,
    sae_id: str | None = None,
):
    """
    Find SAEs in a HuggingFace repository with optional filters.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'canrager/lm_sae')
        model_name: Filter by model name (e.g., 'pythia70m') - substring match
        release: Filter by release directory (e.g., 'standard', 'topk') - substring match
        sae_id: Filter by SAE ID/hook point (e.g., 'layer_4', 'trainer_8') - substring match

    Usage:
        find_sae_bench_models --repo-id canrager/lm_sae --model-name pythia70m
    """
    # Find SAEs
    results = find_saes(
        repo_id=repo_id,
        model_name=model_name,
        release=release,
        sae_id=sae_id,
    )

    # Output results
    print(json.dumps(results, indent=4))


def main():
    """Entrypoint for fire CLI."""
    fire.Fire(find_sae_bench_models)


if __name__ == "__main__":
    main()
