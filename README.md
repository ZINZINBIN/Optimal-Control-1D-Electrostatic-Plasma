# Optimal control for the instability suppression in an electrostatic plasma system via reinforcement learning
## 🧭 Introduction

This repository provides **code and simulation tools** for the **optimal control of an external electric field** to suppress the **bump-on-tail instability** in a **one-dimensional electrostatic plasma system** governed by the **Vlasov–Poisson equations** by using reinforcement learning. We integrate a Particle-In-Cell simulation with optimal control by solving a Hamilton-Jacobi-Bellman equation with reinforcement learning, and verify the suppression of instabilities using a linear damping rate and Fourier analysis. 

The repository includes:
- A **PIC simulation code** for solving the 1D Vlasov–Poisson system.
- An **optimization framework** for computing the optimal external control field.
- Visualization tools for analyzing distribution functions, electric fields, and system energy evolution.

---

## 📚 Background

### 1. Electrostatic Plasma and Vlasov–Poisson System

The **Vlasov-Poisson equation** is a differential equation that describes the time-evolving distribution function of a collisionless plasma \cite{swanson2008plasma}. For a one-dimensional electrostatic plasma on a short timescale, we can neglect the magnetic field term and assume the ions are stationary, allowing electron motion to dominate the dynamics. 

```math
\begin{aligned}
\frac{\partial f_e}{\partial t} + v \cdot \nabla_x f_e - E \cdot \nabla_v f_e = 0 \\
\nabla^2 \phi(x,t) = \int f_edv - 1 \\
E(x,t) = -\nabla \phi(x,t) + E_{in}(x,t) \\
\end{aligned}
```

Here, $f_e(x,v,t)$ is a distribution function of electrons on a phase space, $E$ and $\phi$ are the electric field and its potential, respectively. The electrons in this systems follow the equation described below.

```math
\begin{aligned}
\dot{x} &=v \\
\dot{v} &=-E \\
E_{in} &=\sum_{n}{a_n\sin\frac{2\pi n x}{L} + b_n\cos\frac{2\pi n x}{L}}
\end{aligned}
```

Here, $E_{in}$ is a parameterized external electric field that serves as a control input to suppress the instability. For numerical simulations, we use Particle-In-Cell method, discretizing the system by $N$ super-particles with a particle-grid mapping.

---

### 2. Bump-on-Tail Instability

The **Bump-on-tail instability** is a complex kinetic instability induced by the plasma-wave interaction. Suppose the high-energy electrons are injected into the thermalized plasma, which results in a **non-Maxwellian distribution** with a **“bump” (high-energy tail)** in velocity space. The distribution of electrons in a velocity space is then given as below, generating a bump on the tail of the distribution. 

```math
f_0(v)=\frac{1}{(1+a)\sqrt{2\pi}} e^{-\frac{v^2}{2 v_{th}^2}} + \frac{a}{(1+a)\sqrt{2\pi}} e^{-\frac{(v-v_b)^2}{2 v_{th}^2}}
```

This bump introduces a **positive slope** in the velocity distribution function $\partial f / \partial v > 0$, which enables **wave-particle resonance** and the **growth of Langmuir waves**. This transfers the energy from particles to the wave, results in a growth of the wave perturbation and is oscillated at the nonlinear stage.

---

### 3. Optimal Control
The **Bump-on-tail instability** itself leads to 

- **Energy transfer** from particles to waves  
- **Phase-space vortex formation**  
- **Flattening** of the velocity distribution (quasilinear diffusion)  

Thus, suppressing this instability is crucial in several applications such as:
- Plasma heating and transport control  
- Space and astrophysical plasmas  
- Controlled fusion devices  

#### Objective Functional

This project formulates an **optimal control problem** to design an **external electric field** that minimizes the energy growth associated with the bump-on-tail instability. We define a cost functional that balances **stabilization performance** and **control effort**:

```math
J:= \int_{t_0}^{t_f} \frac{1}{2}\int_0^L|\nabla \phi|^2 dxdt + \frac{\lambda}{2} \int_{t_0}^{t_f} \int_0^L E_{in}^2 dxdt
```

Since the particle-wave interaction transfers energy from high-energy particles to the wave via the electric field, we expect that the system's electric field increases as the instability grows. Thus, the minimization of the electric field, which is the first term, should be considered to be reduced. The second term denotes the electric energy of the external field, which should be minimized to enhance overall efficiency. 


#### Optimization Method

One might be issued to apply the optimal control for this PDE-constrained optimization problem since $f(x,v,t)$ follows high-dimensional nonlinear dynamics which makes it intractable to directly solve this PDE. Thus, we first make this optimal control problem in the sense of Hamilton-Jaocbi-Bellman (HJB) equation. Then, we approximate the value function, a solution for HJB eqaution, using approximate dynamic programming, known as reinforcement learning. We reduce computational cost and simplify solving the optimal control problem with an approximated value function

Among the various methods, we choose DDPG (Deep Deterministic Policy Gradient), which is a continuous-action actor–critic algorithm. The critic approximates the value function $V$. The actor learns the policy for the optimal control $u^{*}$. This is exactly what the HJB equation encodes: The critic network learns an approximation to the value function, effectively learning a numerical solution to the HJB equation. 

```math
V(x,t) = \inf_{E_{in}} \{ \frac{1}{2}\int_0^L|\nabla \phi|^2 + \frac{\lambda}{2} \int_0^L E_{in}^2 dx \}
```
Thus, DDPG = approximate dynamic programming = HJB value iteration in continuous control.

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/ZINZINBIN/Optimal-Control-1D-Electrostatic-Plasma.git
cd optimal-control-vlasov

# Run simulation with optimal control (DDPG)
python3 run_ddpg.py
```

---
## 📊 Results

---

## 📘 Reference
1. Donald Gary Swanson. Plasma kinetic theory. Crc Press, 2008.
2. Giovanni Lapenta. Particle in cell methods. In With Application to
Simulations in Space. Weather. Citeseer, 2016.
3. Luiz Fernando Ziebell, Rudi Gaelzer, and Peter H Yoon. Nonlinear
development of weak beam–plasma instability. Physics of Plasmas,
8(9):3982–3995, 2001.
4. I. Dodin. Plasma waves and instabilities, 2025.
5. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and
Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint
arXiv:1707.06347, 2017.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

### ✨ Citation

If you use this code or results in your research, please cite this repository:

```bibtex
@software{Kim_Optimal_Control_Instability_2025,
  author = {Kim, Jinsu},
  title = {Optimal control for instability suppression in an electrostatic plasma with reinforcement learning},
  year = {2025},
  url = {https://github.com/ZINZINBIN/Optimal-Control-1D-Electrostatic-Plasma}
}
```