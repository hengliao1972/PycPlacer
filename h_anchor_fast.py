"""
H-Anchor Fast: Python wrapper for C++ backend

Provides seamless integration between the C++ core algorithm
and Python utilities (parsing, visualization).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Try to import C++ backend
try:
    import h_anchor_cpp
    HAS_CPP_BACKEND = True
    print("✓ H-Anchor C++ backend loaded")
except ImportError:
    HAS_CPP_BACKEND = False
    raise ImportError("C++ backend is required. Please compile with: python setup.py build_ext --inplace")


class ScoringMethod(Enum):
    """Scoring methods for anchor selection."""
    PAGERANK = "pagerank"
    DEGREE = "degree"
    HYBRID = "hybrid"
    BETWEENNESS = "betweenness"


@dataclass
class PlacementConfig:
    """Configuration for H-Anchor placement."""
    # Hierarchy parameters
    num_layers: int = 5
    top_layer_size: int = 100
    decimation_factor: float = 0.25
    
    # Scoring parameters
    scoring_method: ScoringMethod = ScoringMethod.HYBRID
    alpha: float = 0.4
    beta: float = 0.6
    
    # Force-directed parameters
    top_layer_iterations: int = 300
    refinement_iterations: int = 100
    repulsion_strength: float = 2.0
    attraction_strength: float = 0.1
    overlap_repulsion: float = 5.0    # 防止重叠的强排斥力
    min_spacing: float = 8.0          # cells之间的最小间距
    center_gravity: float = 0.01
    spread_factor: float = 0.6        # 初始分布范围 (0-1, 1=整个die, 0.5=中心50%)
    global_attraction: float = 0.02   # 全局吸引力，让clusters互相靠近
    
    # Legalization parameters
    cell_width: float = 1.0
    cell_height: float = 1.0
    row_height: float = 1.0
    die_width: float = 1000.0
    die_height: float = 1000.0
    
    # Advanced options
    use_transitive_edges: bool = True
    transitive_edge_hops: int = 3
    jitter_scale: float = 20.0
    anchor_mass_factor: float = 5.0


@dataclass
class Cell:
    """Represents a cell/node in the netlist."""
    id: str
    width: float = 1.0
    height: float = 1.0
    fixed: bool = False
    layer: int = 0
    module: str = ""
    
    x: float = 0.0
    y: float = 0.0
    legal_x: float = 0.0
    legal_y: float = 0.0


class HAnchorPlacer:
    """
    H-Anchor placer using C++ backend for high performance.
    """
    
    def __init__(self, config: Optional[PlacementConfig] = None):
        self.py_config = config or PlacementConfig()
        self.graph = None
        self.cells: Dict[str, Cell] = {}
        self.positions: Dict[str, np.ndarray] = {}
        self.legal_positions: Dict[str, Tuple[float, float]] = {}
        self.layers: List[List[str]] = []
        self._node_names: List[str] = []
        self._node_to_idx: Dict[str, int] = {}
        
        self._cpp_config = h_anchor_cpp.PlacementConfig()
        self._sync_config()
        self._cpp_core = h_anchor_cpp.HAnchorCore(self._cpp_config)
    
    def _sync_config(self):
        """Sync Python config to C++ config."""
        c = self._cpp_config
        p = self.py_config
        
        c.num_layers = p.num_layers
        c.top_layer_size = p.top_layer_size
        c.decimation_factor = p.decimation_factor
        c.alpha = p.alpha
        c.beta = p.beta
        c.top_layer_iterations = p.top_layer_iterations
        c.refinement_iterations = p.refinement_iterations
        c.repulsion_strength = p.repulsion_strength
        c.attraction_strength = p.attraction_strength
        c.overlap_repulsion = p.overlap_repulsion
        c.min_spacing = p.min_spacing
        c.center_gravity = p.center_gravity
        c.spread_factor = p.spread_factor
        c.global_attraction = p.global_attraction
        c.die_width = p.die_width
        c.die_height = p.die_height
        c.use_transitive_edges = p.use_transitive_edges
        c.transitive_edge_hops = p.transitive_edge_hops
        c.jitter_scale = p.jitter_scale
        c.anchor_mass_factor = p.anchor_mass_factor
    
    def load_netlist(self, graph, cells: Optional[Dict[str, Cell]] = None):
        """
        Load a netlist graph.
        
        Args:
            graph: NetworkX graph representing the netlist
            cells: Optional dict of Cell objects with properties
        """
        self.graph = graph
        
        # Create default cells if not provided
        if cells:
            self.cells = cells
        else:
            self.cells = {
                node: Cell(id=node) for node in graph.nodes()
            }
        
        # Build node mapping
        self._node_names = list(graph.nodes())
        self._node_to_idx = {name: i for i, name in enumerate(self._node_names)}
        
        # Convert to C++ format
        node_widths = [self.cells[n].width for n in self._node_names]
        node_heights = [self.cells[n].height for n in self._node_names]
        
        edge_from = []
        edge_to = []
        edge_weights = []
        
        for u, v, data in graph.edges(data=True):
            edge_from.append(self._node_to_idx[u])
            edge_to.append(self._node_to_idx[v])
            edge_weights.append(data.get('weight', 1.0))
        
        self._cpp_core.load_graph(
            self._node_names,
            node_widths,
            node_heights,
            edge_from,
            edge_to,
            edge_weights
        )
    
    def run(self) -> Dict[str, Tuple[float, float]]:
        """
        Run the complete H-Anchor placement flow.
        
        Returns:
            Dictionary mapping cell IDs to (x, y) positions.
        """
        # Run C++ core
        self._cpp_core.run()
        
        # Get results
        pos_x = self._cpp_core.get_positions_x()
        pos_y = self._cpp_core.get_positions_y()
        
        # Convert to Python format
        self.positions = {}
        self.legal_positions = {}
        
        for i, name in enumerate(self._node_names):
            self.positions[name] = np.array([pos_x[i], pos_y[i]])
            self.legal_positions[name] = (pos_x[i], pos_y[i])
            self.cells[name].x = pos_x[i]
            self.cells[name].y = pos_y[i]
        
        # Get layers
        cpp_layers = self._cpp_core.get_layers()
        self.layers = []
        for layer_indices in cpp_layers:
            layer_names = [self._node_names[i] for i in layer_indices]
            self.layers.append(layer_names)
        
        return self.legal_positions
    
    def compute_wirelength(self, use_legal: bool = True) -> float:
        """Compute total Half-Perimeter Wirelength (HPWL)."""
        return self._cpp_core.get_hpwl()
    
    def get_placement_stats(self) -> str:
        """Return placement quality statistics."""
        wl = self.compute_wirelength()
        placed = len(self.legal_positions)
        total = len(self.cells)
        num_edges = self.graph.number_of_edges() if self.graph else 0
        
        lines = [
            "\nPlacement Statistics (C++ backend):",
            "=" * 40,
            f"  Cells placed: {placed:,} / {total:,}",
            f"  Total HPWL: {wl:,.2f}",
            f"  Avg HPWL per edge: {wl / max(num_edges, 1):,.2f}",
            "=" * 40,
        ]
        return "\n".join(lines)
    
    # Compatibility properties for visualization
    @property
    def config(self):
        return self.py_config


def run_benchmark(benchmark_name: str):
    """Run a benchmark using the C++ backend."""
    from blif_parser import load_blif_benchmark, get_available_benchmarks, print_netlist_stats
    from visualization import PlacementVisualizer
    import os
    import time
    
    benchmarks = get_available_benchmarks()
    
    if benchmark_name not in benchmarks:
        matches = [b for b in benchmarks if benchmark_name in b]
        if len(matches) == 1:
            benchmark_name = matches[0]
        else:
            print(f"Benchmark '{benchmark_name}' not found.")
            return
    
    filepath = benchmarks[benchmark_name]
    
    print(f"\n{'='*60}")
    print(f"  H-Anchor Placement (C++ Backend)")
    print(f"{'='*60}")
    print(f"  Benchmark: {benchmark_name}")
    print(f"{'='*60}\n")
    
    # Load netlist
    print("Loading BLIF netlist...")
    graph, cells, netlist = load_blif_benchmark(filepath)
    print_netlist_stats(netlist, graph)
    
    # Configure based on size
    num_cells = graph.number_of_nodes()
    
    config = PlacementConfig(
        num_layers=6 if num_cells > 5000 else 5,
        top_layer_size=200 if num_cells > 10000 else 100,
        decimation_factor=0.2,
        die_width=3000 if num_cells > 10000 else 1500,
        die_height=3000 if num_cells > 10000 else 1500,
        overlap_repulsion=5.0,
        min_spacing=8.0,
        center_gravity=0.02,
    )
    
    # Run placement
    placer = HAnchorPlacer(config)
    placer.load_netlist(graph, cells)
    
    start = time.time()
    placer.run()
    elapsed = time.time() - start
    
    print(placer.get_placement_stats())
    print(f"\n  Placement time: {elapsed:.2f} seconds")
    print(f"  Throughput: {num_cells / elapsed:.0f} cells/sec")
    
    # Save visualization
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = benchmark_name.replace("/", "_")
    
    print(f"\nGenerating visualizations...")
    viz = PlacementVisualizer(placer)
    viz.plot_hierarchy_layers(save_path=os.path.join(OUTPUT_DIR, f"{safe_name}_hierarchy.png"))
    viz.plot_placement(save_path=os.path.join(OUTPUT_DIR, f"{safe_name}_placement.png"))
    
    print(f"✓ Saved to {OUTPUT_DIR}/")
    
    return placer


# Aliases for compatibility
HAnchorPlacerFast = HAnchorPlacer


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_benchmark(sys.argv[1])
    else:
        print("Usage: python h_anchor_fast.py <benchmark_name>")
        print("Example: python h_anchor_fast.py iscas89/s38417")

