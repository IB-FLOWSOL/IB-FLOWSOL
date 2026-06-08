# IB-FLOWSOL

![Python](https://img.shields.io/badge/language-Python-blue)
![GPU](https://img.shields.io/badge/accelerated-GPU-green)
![CFD](https://img.shields.io/badge/domain-CFD-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A GPU-accelerated **Computational Fluid Dynamics (CFD) framework** for solving incompressible flows using the **SMAC (Simplified Marker and Cell) scheme**, **Immersed Boundary Method (IBM)**, and **Ghost Node approach** on a **collocated Cartesian grid**.

IB-FLOWSOL combines:

* SMAC pressure–velocity coupling for incompressible Navier–Stokes equations
* Collocated structured grid formulation
* Immersed Boundary Method using Ghost Nodes
* GPU-accelerated linear algebra and iterative solvers
* Integrated meshing, visualization, and post-processing utilities

The project was developed with the goal of providing a flexible framework capable of handling complex immersed geometries without requiring body-fitted meshes while maintaining computational efficiency through GPU acceleration.

---

# Performance Comparison

<img width="971" height="194" alt="image" src="https://github.com/user-attachments/assets/bd92135c-cc9f-4f7b-abe4-d5299c14d237" />

---

# Main Components

## Mesher

Structured Cartesian mesh generator used for immersed boundary simulations.

Features:

* Generation of uniform structured grids
* Geometry discretization support
* Direct integration with the IB-FLOWSOL solver workflow

---

## IB-FLOWSOL_GPU_v1

Main solver implementation.

Capabilities:

* Solves incompressible Navier–Stokes equations
* SMAC-based pressure–velocity coupling
* Immersed Boundary Method using Ghost Nodes
* GPU-accelerated matrix operations and linear solvers
* Support for complex immersed geometries on Cartesian grids

This is the primary code used for running CFD simulations within the framework.

---

## Mesh_Visualizer

Utility for inspecting generated meshes.

Features:

* Structured grid visualization
* Geometry discretization verification
* Mesh debugging and validation

---

## Matplotlib_Visualizer

Post-processing utility for simulation analysis.

Used for:

* Velocity field visualization
* Pressure contour visualization
* Scalar field inspection
* Solver result analysis

---

## Matplotlib_Animator

Animation utility for transient simulations.

Used for:

* Flow evolution visualization
* Time-dependent solution analysis
* Generation of simulation animations

---

# Validation Cases

The solver has been validated against several classical CFD benchmark problems.

Current validation cases include:

* Lid Driven Cavity Flow
* Channel Flow
* Backward Facing Step

Detailed validation results and benchmark comparisons are available in the validation report.

### Validation Report

https://github.com/IB-FLOWSOL/IB-FLOWSOL/blob/main/Simulation%20validation%20pdf.pdf

---

# Solver Demonstrations

### Lid Driven Cavity (Re = 400)

![Video Project 1](https://github.com/user-attachments/assets/ef7cf365-c68d-4aa8-be4b-9bc738884bdd)

### Backward Facing Step (Re = 500)

![Video Project 1 1](https://github.com/user-attachments/assets/6d7b171c-53ec-4dff-a0ea-37e097e00992)

Low-resolution GIFs are used to keep the repository lightweight.

High-quality simulation results and animations can be found here:

https://www.youtube.com/@nabeelhasan1541/playlists

---

# Current Status

The IB-FLOWSOL solver framework is now complete and fully functional.

Ongoing research and development efforts are focused on creating the next generation of solvers that are:

* Faster
* More memory efficient
* Better suited for large-scale simulations
* More scalable on modern GPU architectures

Future developments will build upon the numerical and software infrastructure established by IB-FLOWSOL.

---

# License

This project is released under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007

