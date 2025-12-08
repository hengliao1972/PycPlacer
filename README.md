# H-Anchor: Hierarchical Anchor-Based Placement Algorithm

A novel placement algorithm inspired by **HNSW (Hierarchical Navigable Small World)** graphs. Unlike traditional multilevel placement which clusters nodes into super-nodes, H-Anchor maintains individual cell identities but filters them by "importance" or "topological centrality" to create placement layers.

## 🎯 Core Concept

```
Layer L_top:  ●───────────────●───────────────●  (Global Anchors - Highest Centrality)
                   ╲         ╱ ╲         ╱
Layer L_mid:  ●─────●───────●───●───────●─────●  (Local Anchors - Bridge Gaps)  
                ╲ ╱   ╲   ╱       ╲   ╱   ╲ ╱
Layer L_0:    ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  (All Cells)
```

### HNSW Analogy

| HNSW | H-Anchor |
|------|----------|
| Top layers: few nodes, long links | Top layer: Global Anchors (high centrality) |
| Bottom layers: all nodes, local precision | Bottom layer: All cells in netlist |
| Navigate: "Which node is closest?" | Place: "Where should main blocks go?" |
| Descend for precision | Descend to place local logic between anchors |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python example.py clustered --viz
```

## 📁 Project Structure

```
hap/
├── h_anchor.py        # Core algorithm implementation
├── visualization.py   # Placement visualization tools
├── benchmarks.py      # Synthetic benchmark generators
├── example.py         # Usage examples and demos
└── requirements.txt   # Python dependencies
```

## 🔧 Algorithm Phases

### Phase 1: Hierarchy Construction (Bottom-Up)

Builds placement layers using **spatial inhibition** to ensure anchors are well-distributed:

```python
from h_anchor import HAnchorPlacer, PlacementConfig

config = PlacementConfig(
    num_layers=5,
    top_layer_size=100,
    scoring_method=ScoringMethod.HYBRID,  # PageRank + Degree
    decimation_factor=0.25,  # Each layer is ~25% of previous
)

placer = HAnchorPlacer(config)
placer.load_netlist(graph, cells)
placer.construct_hierarchy()
```

**Score Calculation:**
```
S(v) = α · Degree(v) + β · PageRank(v)
```

**Layer Assignment (Iterative Decimation):**
1. Sort cells by score
2. Select highest-scoring cell
3. Mark its neighbors as "covered" (spatial inhibition)
4. Select next highest unsuppressed cell
5. Repeat until target count reached

### Phase 2: Top-Down Placement (The "Descent")

#### Step A: Top-Level Placement
```python
placer.place_top_layer()  # Force-directed on global anchors
```

Uses **transitive closure edges** to handle disconnected anchor subgraphs:
- If Anchor A connects to Anchor B via 3 unplaced cells
- Add virtual edge with weight 1/3

#### Step B: Recursive Descent
```python
placer.descend_and_refine()
```

For each layer:
1. **Initial Projection:** Place new nodes at weighted center of placed neighbors
   ```
   Pos(u) = Σ Pos(v) · Weight(u,v) / Σ Weight(u,v)
   ```
2. **Add Jitter:** Prevent collapse when many cells project to same point
3. **Refinement:** Force-directed optimization with variable masses
   - Anchors have high mass (move less)
   - New cells have low mass (move freely)

### Phase 3: Legalization
```python
placer.legalize()  # Tetris-style legalization
```

## 📊 Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `random` | Erdős–Rényi random graph |
| `clustered` | Hierarchical blocks with sparse inter-connections |
| `mesh` | 2D grid topology (NoC, systolic arrays) |
| `datapath` | Pipelined datapath with feedback |
| `heterogeneous` | FPGA-like (RAMs, DSPs, IOs as natural anchors) |
| `smallworld` | Watts-Strogatz small-world network |

```bash
python example.py heterogeneous --viz
```

## 🎨 Visualization

```python
from visualization import PlacementVisualizer

viz = PlacementVisualizer(placer)
viz.plot_hierarchy_layers()      # Show layer structure
viz.plot_placement_progression() # Show descent through layers
viz.plot_placement()             # Final placement
viz.plot_wirelength_distribution()
```

## ⚙️ Configuration

```python
PlacementConfig(
    # Hierarchy
    num_layers=5,              # Number of hierarchy levels
    top_layer_size=100,        # Target size for top layer
    decimation_factor=0.25,    # Layer size reduction factor
    
    # Scoring
    scoring_method=ScoringMethod.HYBRID,
    alpha=0.4,                 # Degree weight
    beta=0.6,                  # PageRank weight
    
    # Force-directed
    top_layer_iterations=200,
    refinement_iterations=50,
    repulsion_strength=1.0,
    attraction_strength=0.1,
    anchor_mass_factor=10.0,   # Anchor inertia
    
    # Transitive edges
    use_transitive_edges=True,
    transitive_edge_hops=3,
    
    # Die area
    die_width=1000.0,
    die_height=1000.0,
)
```

## 🔑 Key Advantages

1. **No Clustering:** Maintains individual cell identities throughout
2. **Global-to-Local:** Places critical cells first, ensuring optimal global structure
3. **Natural Anchors:** RAMs, DSPs, IP cores automatically emerge as high-level anchors
4. **Scalability:** O(n log n) complexity with proper implementation

## 📚 Algorithm Comparison

| Feature | Traditional Multilevel | H-Anchor |
|---------|----------------------|----------|
| Cell Identity | Merged into super-nodes | Preserved |
| Global Nets | Cut during partitioning | Placed first (anchors) |
| Hierarchy Basis | Clustering | Centrality + Inhibition |
| Analogy | hMETIS/MLPart | HNSW |

## 🔬 Technical Details

### Transitive Closure Edges

Prevents anchor "folding" when anchors connect only via unplaced cells:

```
Before: A ──?── [unplaced] ──?── B
After:  A ────────────────────── B (weight = 1/path_length)
```

### Force-Directed with Variable Mass

```python
displacement = forces / mass[node]
# Anchors: high mass → small displacement
# New cells: low mass → large displacement
```

### Spatial Inhibition

Ensures anchors don't cluster together:
```python
for node in sorted_by_score:
    if node not in covered:
        select_as_anchor(node)
        for neighbor in graph.neighbors(node):
            covered.add(neighbor)  # Suppress neighbors
```

## 📝 License

MIT License

## 🙏 Acknowledgments

Inspired by:
- HNSW: Hierarchical Navigable Small World graphs
- Force-directed graph drawing (Fruchterman-Reingold)
- Multilevel placement (hMETIS, MLPart)

