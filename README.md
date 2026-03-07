## Hi there 👋

# IB-FLOWSOL

![Python](https://img.shields.io/badge/language-Python-blue)
![GPU](https://img.shields.io/badge/accelerated-GPU-green)
![CFD](https://img.shields.io/badge/domain-CFD-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A foundational **Computational Fluid Dynamics (CFD) solver framework** developed as the base infrastructure for building an **Immersed Boundary Method (IBM) solver**.

This repository contains the **first core solver implementation**, which establishes the numerical framework, mesh generation tools, and visualization utilities that will later support a complete immersed boundary formulation.

---

# Project Goal

The long-term objective of this project is to develop a **flexible immersed boundary CFD framework** capable of handling complex geometries on structured grids.

The current version focuses on building a **robust solver foundation**, including:

* Pressure–velocity coupling infrastructure
* GPU accelerated linear algebra
* Mesh generation utilities
* Flow visualization and analysis tools

This solver acts as the **computational backbone** upon which the immersed boundary methodology will be integrated in future versions.

---

# Solver Versions

## FLOWSOL_v1

The primary solver implementation.

Characteristics:

* Hybrid **CPU + GPU architecture**
* Core CFD operations executed on **CPU**
* Linear algebra operations accelerated on **GPU**
* Pressure Poisson equation solved using **GPU Gauss-Seidel iteration**

This version establishes the initial solver pipeline while gradually introducing GPU acceleration.

---

## Hyper FLOWSOL_v1

A fully **GPU-accelerated implementation** of the solver.

Characteristics:

* Entire solver pipeline runs on **GPU**
* Uses **GPU-accelerated GMRES** for solving linear systems
* Designed for improved scalability and computational efficiency

This version represents the **next stage of solver performance development**.

---

# Meshing Tools

## Mesher_v1

A structured mesh generator capable of producing **uniform Cartesian grids** for a wide range of geometries.

Features:

* Supports discretization of arbitrary geometries
* Generates structured grids suitable for immersed boundary simulations
* Designed to integrate directly with the solver workflow

---

## Mesh Visualizer

A utility tool used to verify the generated mesh.

Capabilities:

* Visual inspection of discretized geometry
* Validation of mesh topology
* Quick debugging of geometry discretization

---

# Visualization Tools

The repository also includes tools for analyzing solver output.

## Matplotlib Visualizer

Used for:

* Plotting velocity fields
* Visualizing scalar fields
* Inspecting solver outputs

---

## Matplotlib Animator

Used for generating **time-dependent animations** of simulation results.

Typical uses include:

* Flow evolution visualization
* Transient flow analysis
* Presentation and validation of solver results

---

# Validation Cases

The solver has been tested on several classical CFD benchmark problems and has shown **positive outcomes**.

Current validation cases include:

* **Lid Driven Cavity Flow**
* **Channel Flow**
* **Backward Facing Step**

Detailed validation results and comparisons with benchmark solutions can be found in the documentation below.

**Validation Report**

https://github.com/IB-FLOWSOL/IB-FLOWSOL/blob/main/Simulation%20validation%20pdf.pdf

---

# Solver Demonstrations

Below are some animations and results generated using the solver.

![Video Project 1](https://github.com/user-attachments/assets/ef7cf365-c68d-4aa8-be4b-9bc738884bdd)
![animation_Re500](https://github.com/user-attachments/assets/a0e041ce-1c84-487a-b6ba-3d915dd6d42e)



---

# Tutorial Video

A tutorial video is provided explaining how to set up and run the solver.

The video demonstrates:

* Solver setup
* Running simulations
* Post-processing results
* Using the framework for research and learning

**YouTube Tutorial**

*(Insert YouTube link here)*

---

# Project Status

This project is currently **under active development**.

Work is ongoing to extend this solver into a **complete immersed boundary CFD framework**.

---


# License

This project is released under the **MIT License**.

---

