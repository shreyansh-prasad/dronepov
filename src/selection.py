import numpy as np
import logging

logger = logging.getLogger("Stage5_Selection")

def calculate_marginal_gain(candidate_grid, selected_grid):
    """
    Computes how much NEW solid angle coverage the candidate adds to the selected_grid.
    To enforce picking unique frames over duplicates, we strongly cap marginal gain.
    """
    MAX_COVERAGE = 1.0
    current_val = np.minimum(selected_grid, MAX_COVERAGE).sum()
    new_val = np.minimum(selected_grid + candidate_grid, MAX_COVERAGE).sum()
    return new_val - current_val

import heapq

def select_frames(candidates, k, processing_cost_per_frame=0.1):
    """
    Offline greedy submodular maximization using Minoux's Lazy Greedy algorithm.
    """
    selected = []
    if not candidates:
        return []
        
    selected_grid = np.zeros_like(candidates[0]['grid'])
    
    # Priority queue for max-heap (invert usefulness for min-heap)
    # Store elements as: [-usefulness, idx, cand_dict]
    pq = []
    
    # 1. Initialize queue with marginal gains against empty set
    for i, cand in enumerate(candidates):
        marginal_gain = calculate_marginal_gain(cand['grid'], selected_grid)
        usefulness = marginal_gain + cand['mission_value'] - processing_cost_per_frame
        cand['current_usefulness'] = usefulness
        cand['current_marginal_gain'] = marginal_gain
        heapq.heappush(pq, [-usefulness, i, cand])
        
    while len(selected) < k and pq:
        # 2. Pop the best candidate
        neg_usefulness, _, best_cand = heapq.heappop(pq)
        
        # 3. Re-evaluate marginal gain against CURRENT selected_grid
        new_gain = calculate_marginal_gain(best_cand['grid'], selected_grid)
        new_usefulness = new_gain + best_cand['mission_value'] - processing_cost_per_frame
        
        best_cand['current_usefulness'] = new_usefulness
        best_cand['current_marginal_gain'] = new_gain
        
        # 4. If new usefulness >= top of the heap's usefulness, it's still the best!
        # (Compare negative usefulness because it's a min-heap)
        if not pq or -new_usefulness <= pq[0][0]:
            selected.append(best_cand)
            selected_grid += best_cand['grid']
        else:
            # 5. Otherwise, push it back with its updated usefulness
            heapq.heappush(pq, [-new_usefulness, _, best_cand])
            
    return selected

def run_test():
    """ 
    Specific unit test from agents.md: 
    3 near-duplicate high-quality frames + 1 unique-but-mediocre frame.
    """
    dup_grid = np.zeros((10, 10))
    dup_grid[0:5, 0:5] = 1.0  # high quality, high coverage area
    
    unique_grid = np.zeros((10, 10))
    unique_grid[5:8, 5:8] = 1.0 # smaller coverage area (mediocre)
    
    candidates = [
        {'id': 'dup1', 'grid': dup_grid.copy(), 'mission_value': 1.0},
        {'id': 'dup2', 'grid': dup_grid.copy(), 'mission_value': 1.0},
        {'id': 'dup3', 'grid': dup_grid.copy(), 'mission_value': 1.0},
        {'id': 'unique', 'grid': unique_grid.copy(), 'mission_value': 0.5},
    ]
    
    selected = select_frames(candidates, k=2, processing_cost_per_frame=0.1)
    ids = [s['id'] for s in selected]
    print("Selected IDs:", ids)
    assert ids == ['dup1', 'unique'], f"Failed greedy selection test! Got {ids}"
    print("Greedy selection test passed!")

if __name__ == '__main__':
    run_test()
