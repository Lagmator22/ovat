import os
import difflib

def resolve_path(hallucinated_path: str) -> str:
    """Fuzzy match a file path when small LLMs hallucinate typos."""
    if os.path.isfile(hallucinated_path):
        return hallucinated_path
        
    basename = os.path.basename(hallucinated_path)
    ext = os.path.splitext(basename)[1].lower()
    if not ext:
        return hallucinated_path
        
    best_match = hallucinated_path
    best_ratio = 0.5  # Need at least 50% similarity
    
    # Search current directory tree
    for root, _, files in os.walk("."):
        if ".git" in root or ".venv" in root:
            continue
            
        # Sort to ensure deterministic behavior (e.g. phase-1 before phase-2)
        for f in sorted(files):
            if f.lower().endswith(ext):
                ratio = difflib.SequenceMatcher(None, basename.lower(), f.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    # Return normalized path
                    best_match = os.path.normpath(os.path.join(root, f))
                    
    return best_match
