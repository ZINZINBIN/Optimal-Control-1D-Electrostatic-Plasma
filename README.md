# Optimal control for the instability suppression in an electrostatic plasma system
## 🧭 Introduction

This repository provides **code and simulation tools** for the **optimal control of an external electric field** to suppress the **bump-on-tail instability** in a **one-dimensional electrostatic plasma system** governed by the **Vlasov–Poisson equations**.  

The project combines **Particle-In-Cell (PIC) simulations** with **numerical optimal control** techniques to design and apply an external control field that stabilizes the plasma distribution function.  

The repository includes:
- A **PIC simulation code** for solving the 1D Vlasov–Poisson system.
- An **optimization framework** for computing the optimal external control field.
- Visualization tools for analyzing distribution functions, electric fields, and system energy evolution.

---

## 📚 Background

### 1. Electrostatic Plasma and Vlasov–Poisson System

The **Vlasov–Poisson system** describes the self-consistent evolution of a collisionless plasma under an electrostatic potential.  
In one spatial dimension, it is written as:

```math
\frac{\partial f(x, v, t)}{\partial t} + v \frac{\partial f(x, v, t)}{\partial x} + \frac{q}{m} E(x, t) \frac{\partial f(x, v, t)}{\partial v} = 0,
```

where  
- \( f(x, v, t) \): distribution function of charged particles  
- \( q, m \): particle charge and mass  
- \( E(x, t) = -\frac{\partial \phi(x, t)}{\partial x} \): electric field  
- \( \phi(x, t) \): electrostatic potential determined by Poisson’s equation  

```math
\frac{\partial^2 \phi}{\partial x^2} = -\frac{q}{\epsilon_0} \left( n(x, t) - n_0 \right),
```
with \( n(x, t) = \int f(x, v, t) \, dv \).

These coupled equations govern how an initially perturbed distribution function evolves due to **collective plasma oscillations** and **instabilities**.

---

### 2. Bump-on-Tail Instability

The **bump-on-tail instability** arises when a **non-Maxwellian distribution** contains a **“bump” (high-energy tail)** in velocity space, typically caused by a **fast electron beam** injected into a background plasma.  

This bump introduces a **positive slope** in the velocity distribution function \( \partial f / \partial v > 0 \), which enables **wave-particle resonance** and the **growth of Langmuir waves**.  

Over time, this leads to:
- **Energy transfer** from particles to waves  
- **Phase-space vortex formation**  
- **Flattening** of the velocity distribution (quasilinear diffusion)  

Suppressing this instability is crucial in applications such as:
- Plasma heating and transport control  
- Space and astrophysical plasmas  
- Controlled fusion devices  

---

### 3. Optimal Control

This project formulates an **optimal control problem** to design an **external electric field \( E_c(t) \)** that minimizes the energy growth associated with the bump-on-tail instability.

#### Objective Functional

We define a cost functional \( J \) that balances **stabilization performance** and **control effort**:

\[
J = \frac{1}{2} \int_0^T \left[ \| E(x, t) \|^2 + \lambda \| E_c(t) \|^2 \right] dt,
\]

where  
- \( E(x, t) \): self-consistent field from the plasma  
- \( E_c(t) \): external control field to be optimized  
- \( \lambda \): regularization parameter controlling the trade-off between suppression and energy cost  

#### Optimization Method

The optimization loop involves:
1. **Forward Simulation (PIC):**  
   Solve the Vlasov–Poisson system with a candidate control field.  
2. **Adjoint Calculation:**  
   Compute gradients of the cost functional using adjoint equations or numerical sensitivities.  
3. **Control Update:**  
   Update \( E_c(t) \) via a gradient-based optimizer (e.g., conjugate gradient or L-BFGS).  

This iterative procedure continues until the instability is effectively suppressed or the cost functional converges below a threshold.

---

## 🧩 Code Structure

```
📦 optimal-control-vlasov
├── src/
│   ├── pic_solver.py          # Particle-In-Cell simulation for Vlasov-Poisson system
│   ├── control_optimization.py # Optimal control algorithm
│   ├── adjoint_solver.py      # Adjoint-based gradient computation
│   ├── visualization.py       # Tools for plotting f(x,v), E(x), etc.
│   └── utils.py               # Utility functions
├── data/
│   ├── initial_conditions/
│   ├── results/
│   └── figures/
├── examples/
│   └── bump_on_tail_control.py
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/optimal-control-vlasov.git
cd optimal-control-vlasov

# Install dependencies
pip install -r requirements.txt

# Run simulation example
python examples/bump_on_tail_control.py
```

---

## 📊 Results

The optimization algorithm generates:
- Suppressed electric field growth (stabilized plasma)
- Controlled energy exchange between field and particles
- Smoothed velocity distribution (bump suppression)

Example output plots:
- Distribution function \( f(x,v,t) \)
- Time evolution of electric field amplitude
- Control field \( E_c(t) \) profile
- Cost functional convergence

---

## 📘 Reference

1. J. P. Boyd, *The Vlasov–Poisson System and Plasma Waves*, Springer, 2003.  
2. F. Filbet and E. Sonnendrücker, “Comparison of Eulerian Vlasov Solvers,” *Comput. Phys. Commun.*, 150(3):247–266, 2003.  
3. C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer Simulation*, IOP Publishing, 2004.  
4. L. Chacón et al., “Optimal Control of Plasma Instabilities,” *Phys. Plasmas*, 27(12):122301, 2020.  

---

## 🧠 Acknowledgment

This work integrates methods from plasma kinetic theory, numerical optimization, and control theory. The authors acknowledge open-source scientific Python packages such as **NumPy**, **SciPy**, and **Matplotlib**, which form the computational backbone of this project.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

### ✨ Citation

If you use this code or results in your research, please cite this repository:

```bibtex
@software{yourname2025optimalcontrol,
  author = {Your Name},
  title = {Optimal Control of External Electric Field for Suppressing Bump-on-Tail Instability in a 1D Vlasov–Poisson Plasma},
  year = {2025},
  url = {https://github.com/your-username/optimal-control-vlasov}
}
```
