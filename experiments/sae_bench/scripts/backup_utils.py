"""
Utility functions for backing up evaluation results.
"""
import shutil
from pathlib import Path
from datetime import datetime


def backup_eval_results(base_path: str = "eval_results"):
    """
    Backup eval_results directory to eval_results_bkp before running evaluations.

    This prevents the sae_bench library from deleting previous evaluation results.
    The backup includes a timestamp to preserve multiple backups.

    Args:
        base_path: Path to the eval_results directory (relative to project root)
    """
    eval_results_path = Path(base_path)

    # Only backup if eval_results exists and has content
    if not eval_results_path.exists():
        print(f"No {base_path} directory to backup (this is normal for first run)")
        return

    # Check if there are any files/directories in eval_results
    has_content = any(eval_results_path.iterdir())
    if not has_content:
        print(f"{base_path} directory is empty, skipping backup")
        return

    # Create backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"{base_path}_bkp")
    timestamped_backup = backup_path / timestamp

    print(f"\n=== Backing up {base_path} ===")
    print(f"Source: {eval_results_path.absolute()}")
    print(f"Destination: {timestamped_backup.absolute()}")

    try:
        # Create backup directory if it doesn't exist
        backup_path.mkdir(exist_ok=True)

        # Copy the entire eval_results directory to timestamped backup
        shutil.copytree(eval_results_path, timestamped_backup, dirs_exist_ok=True)

        print(f"✓ Backup complete: {timestamped_backup}")

        # Also create/update a 'latest' symlink for easy access
        latest_link = backup_path / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(timestamp, target_is_directory=True)
        print(f"✓ Updated latest symlink: {latest_link} -> {timestamp}")

    except Exception as e:
        print(f"WARNING: Failed to backup {base_path}: {e}")
        print("Continuing with evaluation anyway...")


def backup_specific_eval_results(eval_type: str):
    """
    Backup a specific eval results subdirectory.

    Args:
        eval_type: Type of evaluation (e.g., 'core', 'scr', 'absorption')
    """
    eval_results_subdir = f"eval_results/{eval_type}"
    backup_eval_results(eval_results_subdir)
