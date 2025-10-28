"""
SAE Bench Utilities

Helper functions for finding and working with SAEs from HuggingFace repositories.
"""

from huggingface_hub import list_repo_files


def find_saes(
    repo_id: str,
    model_name: str | None = None,
    release: str | None = None,
    sae_id: str | None = None,
) -> list[dict[str, str]]:
    """
    Find SAEs in a HuggingFace repository with optional filters.

    Args:
        repo_id: HuggingFace repository ID (e.g., "canrager/lm_sae")
        model_name: Optional filter by model name (e.g., "pythia70m") - substring match
        release: Optional filter by release directory (e.g., "standard", "topk") - substring match
        sae_id: Optional filter by SAE ID/hook point (e.g., "layer_4", "trainer_8") - substring match

    Returns:
        List of dicts with keys:
            - repo_id: The HuggingFace repo
            - filename: Full path to ae.pt file
            - release: Release directory name
            - sae_id: Hook point/trainer path

    Example:
        >>> # Get all pythia70m standard SAEs for layer 4
        >>> find_saes("canrager/lm_sae", model_name="pythia70m", release="standard", sae_id="layer_4")
        [
            {
                'repo_id': 'canrager/lm_sae',
                'filename': 'pythia70m_sweep_standard_ctx128_0712/resid_post_layer_4/trainer_8/ae.pt',
                'release': 'pythia70m_sweep_standard_ctx128_0712',
                'sae_id': 'resid_post_layer_4/trainer_8'
            },
            ...
        ]
    """
    # Get all files from repo
    all_files = list(list_repo_files(repo_id))

    # Filter for .pt files only
    pt_files = [f for f in all_files if f.endswith(".pt")]

    results = []

    for filepath in pt_files:
        # Parse file path structure: {release}/{hook_point}/{trainer}/ae.pt
        parts = filepath.split("/")

        # Skip if not in expected format (release/hook_point/trainer/ae.pt)
        if len(parts) != 4 or parts[3] != "ae.pt":
            continue

        release_dir = parts[0]
        hook_point = parts[1]
        trainer = parts[2]
        sae_id_str = f"{hook_point}/{trainer}"

        # Apply filters
        if model_name is not None and model_name not in release_dir:
            continue

        if release is not None and release not in release_dir:
            continue

        if sae_id is not None and sae_id not in sae_id_str:
            continue

        # Add to results
        results.append(
            {
                "repo_id": repo_id,
                "filename": filepath,
                "release": release_dir,
                "sae_id": sae_id_str,
            }
        )

    return results
