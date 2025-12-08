/**
 * H-Anchor Core Algorithm - C++ Implementation
 * 
 * High-performance implementation of the hierarchical anchor-based
 * placement algorithm inspired by HNSW.
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace hanchor {

// Forward declarations
struct Cell;
struct Edge;
class Graph;
class HierarchyBuilder;
class ForceDirectedEngine;
class HAnchorCore;

/**
 * Configuration for H-Anchor placement
 */
struct PlacementConfig {
    // Hierarchy parameters
    int num_layers = 5;
    int top_layer_size = 100;
    double decimation_factor = 0.25;
    
    // Scoring
    double alpha = 0.4;  // Degree weight
    double beta = 0.6;   // PageRank weight
    
    // Force-directed
    int top_layer_iterations = 300;
    int refinement_iterations = 100;
    double repulsion_strength = 2.0;
    double attraction_strength = 0.1;
    double overlap_repulsion = 5.0;   // 防止重叠的强排斥力
    double min_spacing = 8.0;         // cells之间的最小间距
    double center_gravity = 0.01;
    double spread_factor = 0.6;       // 初始分布范围 (0-1, 1=整个die, 0.5=中心50%区域)
    double global_attraction = 0.02;  // 全局吸引力，让clusters互相靠近
    
    // Die area
    double die_width = 1000.0;
    double die_height = 1000.0;
    
    // Advanced
    bool use_transitive_edges = true;
    int transitive_edge_hops = 3;
    double jitter_scale = 20.0;
    double anchor_mass_factor = 5.0;
};

/**
 * 2D Position
 */
struct Position {
    double x = 0.0;
    double y = 0.0;
    
    Position() = default;
    Position(double x_, double y_) : x(x_), y(y_) {}
    
    Position operator+(const Position& other) const {
        return Position(x + other.x, y + other.y);
    }
    
    Position operator-(const Position& other) const {
        return Position(x - other.x, y - other.y);
    }
    
    Position operator*(double s) const {
        return Position(x * s, y * s);
    }
    
    Position operator/(double s) const {
        return Position(x / s, y / s);
    }
    
    double norm() const {
        return std::sqrt(x * x + y * y);
    }
    
    Position normalized() const {
        double n = norm();
        if (n < 1e-10) return Position(0, 0);
        return *this / n;
    }
};

/**
 * Cell/Node in the netlist
 */
struct Cell {
    int id;
    std::string name;
    double width = 1.0;
    double height = 1.0;
    Position pos;
    Position legal_pos;
    int layer = -1;  // Which hierarchy layer (-1 = not assigned)
    double score = 0.0;  // Centrality score
};

/**
 * Edge in the graph
 */
struct Edge {
    int from;
    int to;
    double weight = 1.0;
};

/**
 * Graph representation optimized for placement
 */
class Graph {
public:
    std::vector<Cell> cells;
    std::vector<Edge> edges;
    std::vector<std::vector<int>> adjacency;  // adjacency[i] = list of neighbor indices
    std::vector<std::vector<double>> adj_weights;  // corresponding weights
    
    int num_nodes() const { return static_cast<int>(cells.size()); }
    int num_edges() const { return static_cast<int>(edges.size()); }
    
    void build_adjacency();
    void add_node(const std::string& name, double width = 1.0, double height = 1.0);
    void add_edge(int from, int to, double weight = 1.0);
    
    // Get neighbors of a node
    const std::vector<int>& neighbors(int node) const { return adjacency[node]; }
    double edge_weight(int from, int to) const;
};

/**
 * Hierarchy construction using spatial inhibition
 */
class HierarchyBuilder {
public:
    HierarchyBuilder(Graph& graph, const PlacementConfig& config);
    
    void compute_scores();
    void build_layers();
    
    const std::vector<std::vector<int>>& get_layers() const { return layers_; }
    
private:
    Graph& graph_;
    const PlacementConfig& config_;
    std::vector<std::vector<int>> layers_;  // layers_[0] = top (sparse), layers_[n] = bottom (all)
    
    void compute_pagerank(int iterations = 20);
    void compute_degree_centrality();
};

/**
 * Force-directed placement engine with density control
 */
class ForceDirectedEngine {
public:
    ForceDirectedEngine(const PlacementConfig& config);
    
    void run_layout(
        Graph& graph,
        const std::vector<int>& active_nodes,
        const std::unordered_set<int>& fixed_nodes,
        const std::unordered_map<int, double>& masses,
        int iterations
    );
    
private:
    const PlacementConfig& config_;
    std::mt19937 rng_;
    
    void compute_repulsion(
        const Graph& graph,
        const std::vector<int>& nodes,
        std::vector<Position>& forces,
        double k_repel
    );
    
    void compute_attraction(
        const Graph& graph,
        const std::vector<int>& nodes,
        std::vector<Position>& forces,
        double k_attract
    );
    
    void compute_overlap_repulsion(
        const Graph& graph,
        const std::vector<int>& nodes,
        std::vector<Position>& forces,
        double k_repel
    );
    
    void compute_center_gravity(
        const Graph& graph,
        const std::vector<int>& nodes,
        std::vector<Position>& forces
    );
    
    void compute_global_attraction(
        const Graph& graph,
        const std::vector<int>& nodes,
        std::vector<Position>& forces
    );
};

/**
 * Main H-Anchor placement algorithm
 */
class HAnchorCore {
public:
    HAnchorCore(const PlacementConfig& config);
    
    // Load graph from vectors (called from Python)
    void load_graph(
        const std::vector<std::string>& node_names,
        const std::vector<double>& node_widths,
        const std::vector<double>& node_heights,
        const std::vector<int>& edge_from,
        const std::vector<int>& edge_to,
        const std::vector<double>& edge_weights
    );
    
    // Run the complete placement flow
    void run();
    
    // Get results
    std::vector<double> get_positions_x() const;
    std::vector<double> get_positions_y() const;
    std::vector<int> get_layer_sizes() const;
    double get_hpwl() const;
    
    // Access to layers for visualization
    std::vector<std::vector<int>> get_layers() const;
    
private:
    PlacementConfig config_;
    Graph graph_;
    std::vector<std::vector<int>> layers_;
    
    void construct_hierarchy();
    void place_top_layer();
    void descend_and_refine();
    void project_new_nodes(
        const std::vector<int>& new_nodes,
        const std::unordered_set<int>& anchors
    );
    void refine_layer(
        const std::vector<int>& current_nodes,
        const std::unordered_set<int>& anchors
    );
};

}  // namespace hanchor

