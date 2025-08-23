from pathlib import Path

def get_project_root() -> Path:
    """
    Get the project root directory (ml_tokamak)
    Don't want to use absolute paths, but want to make it relative to the project root, regardless of current working directory.
    """

    current_dir = Path(__file__).resolve()
    
    # Go up the directory tree until there's ml_tokamak
    # (the one that contains src/, data/, runs/, etc.)
    while current_dir.name != 'ml_tokamak' and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    
    if current_dir.name != 'ml_tokamak':
        # Fallback: assume we're in src/training and go up two levels
        current_dir = Path(__file__).resolve().parent.parent.parent
    
    return current_dir