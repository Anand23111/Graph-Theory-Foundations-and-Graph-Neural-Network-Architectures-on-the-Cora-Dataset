# Graph Theory & GCN/ChebNet on Cora: README

This repository demonstrates two main tasks:

1. **Foundational Graph Theory** operations on a custom NetworkX graph.
2. **Training Graph Convolutional Networks (GCN)** and **Chebyshev Networks (ChebNet)** on the Cora dataset using PyTorch Geometric.

Below, you will find an overview of each part, setup instructions, and how to run the code.

---

## Table of Contents
1. [Overview](#overview)
2. [Dependencies and Setup](#dependencies-and-setup)
3. [File Structure](#file-structure)
4. [Usage Instructions](#usage-instructions)
   - [Part 1: Foundational Graph Theory](#part-1-foundational-graph-theory)
   - [Part 2: GCN and ChebNet on Cora](#part-2-gcn-and-chebnet-on-cora)
5. [Results](#results)
6. [Experimental Insights](#experimental-insights)
7. [References](#references)

---

## Overview

### Part 1: Foundational Graph Theory
- **Objective**: Demonstrate how to construct and analyze basic graph properties using NetworkX.
- **Key Operations**:
  - Building a graph from an edge list.
  - Computing:
    - Adjacency Matrix \(`A`\)
    - Degree Matrix \(`D`\)
    - Unnormalized Laplacian \(`L = D - A`\)
    - Normalized Laplacian
    - Eigenvalues of the Laplacian
  - Plotting the **degree distribution**.

### Part 2: GCN and ChebNet on the Cora Dataset
- **Dataset**: Uses the **Cora** citation network, a node classification benchmark.
  - Nodes are scientific publications.
  - Edges indicate citations.
  - Each node has a feature vector and a class label.
- **Models**:
  - **GCN**: Two-layer Graph Convolutional Network.
  - **ChebNet**: Two-layer Chebyshev Convolution Network with polynomial order \(`K`\).
- **Training**:
  - Runs for a fixed number of epochs (e.g., 100 for GCN, 200 for ChebNet).
  - Monitors training, validation, and test accuracy at intervals.
- **Hyperparameter Search**:
  - Vary the Chebyshev polynomial order \(`K`\) and hidden-layer dimension for ChebNet.
  - Collect and print final accuracies.

---

## Dependencies and Setup

1. **Python 3.7+** recommended.
2. **Packages**:
   - **NumPy** for numeric operations.
   - **networkx** for graph manipulations.
   - **matplotlib** for plotting.
   - **torch**, **torchvision** for PyTorch basics.
   - **torch-geometric** for graph-based deep learning.

Install them via:

```bash
pip install numpy networkx matplotlib torch torchvision torch-geometric
