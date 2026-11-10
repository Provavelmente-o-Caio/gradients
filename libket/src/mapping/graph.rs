// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Adjacency-matrix graph representation used by the qubit-mapping passes.
//!
//! [`GraphMatrix`] stores an undirected weighted graph in a compact lower-
//! triangular matrix (row `i`, column `j` where `i > j`). It supports edge
//! queries, BFS traversal, and all-pairs shortest-path distances via the
//! Floyd–Warshall algorithm.

use std::collections::VecDeque;

/// An undirected weighted graph stored as a lower-triangular adjacency matrix.
///
/// Nodes are identified by zero-based `usize` indices.  Edge weights are
/// `i64`; absence of an edge is represented as `None`.  A self-loop always
/// has implicit weight `0`.
///
/// Distances between nodes are computed separately via
/// [`GraphMatrix::calculate_distance`] (Floyd–Warshall) and cached in the
/// `distance` field.  Calling [`GraphMatrix::dist`] before
/// [`GraphMatrix::calculate_distance`] panics.
#[derive(Debug, Clone, Default)]
pub struct GraphMatrix {
    graph: Vec<Vec<Option<i64>>>,
    n: usize,
    distance: Option<Vec<Vec<i64>>>,
}

impl GraphMatrix {
    /// Creates a new graph with `n` isolated nodes.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut graph = vec![];
        for i in 1..n {
            graph.push(vec![None; i]);
        }
        Self {
            graph,
            n,
            distance: None,
        }
    }

    /// Returns the number of nodes in the graph.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Grows the graph by one node (appending a new row to the adjacency matrix).
    fn add_node(&mut self) {
        if self.n != 0 {
            self.graph.push(vec![None; self.n]);
        }
        self.n += 1;
    }

    /// Returns the weight of the edge between nodes `i` and `j`, or `None`
    /// if no edge exists.  Returns `Some(0)` for self-loops.
    #[must_use]
    pub fn edge(&self, i: usize, j: usize) -> Option<i64> {
        let (i, j) = (i, j);

        if self.n <= i || self.n <= j {
            return None;
        }

        if i == j {
            return Some(0);
        }

        let (i, j) = if i > j { (i, j) } else { (j, i) };
        let i = i - 1;
        self.graph[i][j]
    }

    /// Returns all neighbours of `node` as `(neighbour_index, edge_weight)` pairs.
    #[must_use]
    pub fn neighbors(&self, node: usize) -> Vec<(usize, i64)> {
        (0..self.n)
            .filter_map(|j| {
                if node == j {
                    None
                } else {
                    self.edge(node, j).map(|value| (j, value))
                }
            })
            .collect()
    }

    /// Sets (or overwrites) the edge weight between nodes `i` and `j`.
    ///
    /// Automatically grows the graph if `i` or `j` exceed the current node
    /// count.  Self-loops (`i == j`) are silently ignored.
    pub fn set_edge(&mut self, i: usize, j: usize, value: i64) {
        let (i, j) = (i, j);

        while self.n <= i || self.n <= j {
            self.add_node();
        }

        if i == j {
            return;
        }

        let (i, j) = if i > j { (i, j) } else { (j, i) };
        let i = i - 1;
        self.graph[i][j] = Some(value);
    }

    /// Returns the shortest-path distance between nodes `i` and `j`.
    ///
    /// # Panics
    /// Panics if [`GraphMatrix::calculate_distance`] has not been called first.
    #[must_use]
    pub fn dist(&self, i: usize, j: usize) -> i64 {
        let (i, j) = (i, j);

        if let Some(distance) = &self.distance {
            if i == j {
                return 0;
            }

            let (i, j) = if i > j { (i, j) } else { (j, i) };
            let i = i - 1;
            distance[i][j]
        } else {
            panic!("Calculate distance before")
        }
    }

    /// Updates the distance between nodes `i` and `j` to `min(current, value)`.
    ///
    /// # Panics
    /// Panics if [`GraphMatrix::calculate_distance`] has not been called first.
    fn set_dist_min(&mut self, i: usize, j: usize, value: i64) {
        let (i, j) = (i, j);

        assert!(
            self.distance.is_some(),
            "Cannot set distance without distance matrix"
        );
        if i == j {
            return;
        }

        let (i, j) = if i > j { (i, j) } else { (j, i) };
        let i = i - 1;
        let value = std::cmp::min(self.distance.as_ref().unwrap()[i][j], value);
        self.distance.as_mut().unwrap()[i][j] = value;
    }

    /// Computes all-pairs shortest-path distances using the Floyd–Warshall
    /// algorithm and caches the result.
    ///
    /// A no-op if distances have already been calculated.  After this call
    /// [`GraphMatrix::dist`] may be used freely.
    pub fn calculate_distance(&mut self) {
        if self.distance.is_some() {
            return;
        }

        self.distance = Some(
            self.graph
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|value| {
                            if value.is_some() {
                                1
                            } else {
                                i64::from(u32::MAX)
                            }
                        })
                        .collect::<Vec<i64>>()
                })
                .collect(),
        );

        for k in 0..self.n {
            for i in 0..self.n {
                for j in i..self.n {
                    let dist = self.dist(i, k) + self.dist(k, j);
                    self.set_dist_min(i, j, dist);
                }
            }
        }
    }

    /// Returns the index of the *graph centre*: the node that minimises the
    /// maximum shortest-path distance to any other node (i.e., the node with
    /// the smallest eccentricity).  Ties are broken by choosing the node with
    /// the smallest sum of all distances.
    ///
    /// # Panics
    /// Panics if the graph is empty or if [`GraphMatrix::calculate_distance`]
    /// has not been called first.
    #[must_use]
    pub fn get_center(&self) -> usize {
        let max_distance: Vec<i64> = (0..self.n)
            .map(|i| (0..self.n).map(|j| self.dist(i, j)).max().unwrap())
            .collect();

        let min = max_distance.iter().min().unwrap();

        let mut center_list: Vec<(usize, i64)> = max_distance
            .iter()
            .enumerate()
            .filter_map(|(node, value)| {
                if value <= min {
                    Some((node, (0..self.n).map(|other| self.dist(node, other)).sum()))
                } else {
                    None
                }
            })
            .collect();

        center_list.sort_by_key(|(_, dist)| *dist);

        center_list[0].0
    }

    /// Performs a breadth-first search starting from `start` and returns the
    /// visited nodes in BFS order.
    ///
    /// Among siblings at the same BFS depth, nodes are visited in ascending
    /// edge-weight order.
    #[must_use]
    pub fn breadth_first_search(&self, start: usize) -> VecDeque<usize> {
        let mut visited = Vec::new();
        let mut queue = VecDeque::from([start]);

        while let Some(front) = queue.pop_front() {
            visited.push(front);
            let neighbors = self.neighbors(front);
            let mut neighbors: Vec<_> = neighbors
                .iter()
                .filter_map(|(index, value)| {
                    if !visited.contains(index) && !queue.contains(index) {
                        Some((*index, *value))
                    } else {
                        None
                    }
                })
                .collect();

            neighbors.sort_by_key(|(_, value)| *value);

            for (next, _) in neighbors {
                queue.push_back(next);
            }
        }

        VecDeque::from(visited)
    }

    /// Returns the degree (number of neighbours) of `node`.
    #[must_use]
    pub fn degree(&self, node: usize) -> usize {
        self.neighbors(node).len()
    }
}
