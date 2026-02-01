# Phase 2 vs Phase 3: Object Association Methods Comparison

## Quick Reference

| Method | Accuracy | Speed | Best For |
|---|---|---|---|
| Geometric Heuristic | ~70% | Instant | Baseline only |
| **Phase 2: AffinityNet** | 92-95% | <10ms | ⭐ Most projects |
| **Phase 3: Scene Graph GNN** | 95-97% | ~20ms | Complex scenes, max accuracy |

## Files Overview

### Common
- `affinity_net.py` / `scene_graph_gnn.py` - Models
- `*_dataset.py` - Dataset classes  
- `*_integration.py` - Inference utilities
- `train_*.ipynb` - Kaggle notebooks

### Quick Start

**Phase 2:**
```bash
python train_affinity.py
# Output: best_affinity_net.pth (~10KB)
```

**Phase 3:**
```bash
python scene_graph_dataset.py  # Test
python scene_graph_gnn.py      # Test
# Then run train_scene_graph_gnn.ipynb on Kaggle
# Output: best_scene_graph_gnn.pth (~180KB)
```

See [phase2_vs_phase3_comparison.md](file:///c:/Users/NeverGonnaGiveYouUp/.gemini/antigravity/brain/773071cf-6911-4e4e-8a4b-db886b697d7f/phase2_vs_phase3_comparison.md) for details.
