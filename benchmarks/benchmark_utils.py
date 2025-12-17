import io
from PIL import Image
import scanpy as sc


def prepare_img(fig):
    """
    Save matplotlib figure to PIL Image for wandb logging with high resolution.
    
    Args:
        fig: matplotlib figure object
        
    Returns:
        PIL Image object ready for wandb.Image()
    """
    # Save figure to buffer with high DPI and no cropping for wandb
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    return img


def find_leiden_resolution_for_n_clusters(adata, target_n_clusters, min_res=0.1, max_res=1.0, 
                                          max_iterations=30, tolerance=0, random_state=42):
    """
    Use binary search to find the Leiden resolution parameter that produces exactly the target number of clusters.
    
    Args:
        adata: AnnData object with neighbors already computed
        target_n_clusters: Target number of clusters (default: 12)
        min_res: Minimum resolution to search (default: 0.4)
        max_res: Maximum resolution to search (default: 0.7)
        max_iterations: Maximum number of binary search iterations (default: 20)
        tolerance: Acceptable difference from target (default: 0, meaning exact match)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (optimal_resolution, actual_n_clusters)
    """
    low, high = min_res, max_res
    best_resolution = None
    best_n_clusters = None
    best_diff = float('inf')
    
    print(f"Binary search for resolution to get {target_n_clusters} clusters...")
    
    for iteration in range(max_iterations):
        mid_res = (low + high) / 2.0
        
        # Perform Leiden clustering with current resolution
        sc.tl.leiden(adata, resolution=mid_res, key_added='leiden', random_state=random_state)
        n_clusters = len(adata.obs['leiden'].unique())
        
        print(f"  Iteration {iteration + 1}: resolution={mid_res:.4f}, n_clusters={n_clusters}")
        
        # Track the best result so far
        diff = abs(n_clusters - target_n_clusters)
        if diff < best_diff:
            best_diff = diff
            best_resolution = mid_res
            best_n_clusters = n_clusters
        
        # Check if we've found the target
        if abs(n_clusters - target_n_clusters) <= tolerance:
            print(f"  ✓ Found exact match: resolution={mid_res:.4f}, n_clusters={n_clusters}")
            return mid_res, n_clusters
        
        # Adjust search range
        # Higher resolution typically means more clusters
        if n_clusters < target_n_clusters:
            low = mid_res
        else:
            high = mid_res
        
        # If the search range becomes too small, break
        if high - low < 0.0001:
            print(f"  Search range too small, stopping. Best: resolution={best_resolution:.4f}, n_clusters={best_n_clusters}")
            break
    
    # If exact match not found, use the best result
    print(f"  Using best result: resolution={best_resolution:.4f}, n_clusters={best_n_clusters} (diff={best_diff})")
    
    # Re-run with the best resolution to ensure adata has the correct clustering
    sc.tl.leiden(adata, resolution=best_resolution, key_added='leiden', random_state=random_state)
    
    return best_resolution, best_n_clusters
