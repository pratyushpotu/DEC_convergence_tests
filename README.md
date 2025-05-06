# DEC Hodge-Laplacian Simulations

This repository implements Discrete Exterior Calculus (DEC) simulations for solving the 2D Hodge-Laplacian problem on triangular meshes.

## Structure

```
dec_poisson_sim/
├── dec/                    # Core DEC operators and utilities
│   ├── hodge.py            # Hodge stars (*0, *1, *2)
│   ├── derivatives.py      # Discrete exterior derivatives (d0, d1)
│   └── integrators.py      # de-Rham maps
│   
│
├── mesh/
│   ├── generation.py       # Mesh generator for symmetric triangle mesh
│   ├── geometry.py         # Functions for triangle geometry
│   └── plotting.py         # Mesh plotting
│
├── experiments/
│   ├── k0_symmetric.py
│   ├── k0_perturbed.py
│   ├── k1_symmetric.py
│   ├── k1_perturbed.py
│   ├── k2_symmetric.py
│   ├── k2_perturbed.py
│   └── manufacturing.py    # Manufactured solution and rhs terms
│
├── tests/                  # Unit tests for each module
└── README.md
```

## Usage

To run an experiment, execute one of the scripts from the `experiments/` folder, e.g.,

```bash
python experiments/k0_symmetric.py
```

Each script computes convergence data over a sequence of mesh resolutions, then writes a table of errors and convergence rates to a corresponding `.txt` file (e.g., `k0_symmetric_convergence.txt`).

## Experiments

Each experiment solves the DEC Hodge Laplacian for:
- **0-forms**
- **1-forms**
- **2-forms**

Each case is tested on both:
- A **symmetric** barycentrically subdivided equilateral triangle mesh
- A **perturbed** mesh where interior points are shifted slightly

## Requirements

- Python 3.8+
- `numpy`, `scipy`, `sympy`, `matplotlib` (for visualization)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## License

MIT License
