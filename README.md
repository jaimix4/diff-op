# Edge Transport Modeling: Two-Channel Analytical Fits to Global Turbulence

This repository contains a dynamic Python tool designed to fit an analytical, velocity-dependent transport model to ground-truth edge/SOL turbulence data (from global simulations like GRILLIX or experimental data). 

The tool optimizes a complex, off-diagonal transport matrix to match particle and heat fluxes for both electrons and ions simultaneously, while respecting the ambipolarity constraint.

---

## 1. The Physical Model

In standard mean-field transport codes (like SOLPS), turbulent transport is often heavily simplified into pure diagonal diffusion (e.g., Fick's Law: $\Gamma = -D \nabla n$). However, first-principles turbulence simulations reveal that transport in the edge and Scrape-Off Layer (SOL) is highly complex, involving cross-gradient drives and ballistic blob transport.

To capture this, our model defines the radial particle flux ($\Gamma_\perp$) and the conductive heat flux ($q_\perp$) using a full transport matrix $\mathbf{A}$, coupled with a convective term:

$$
\begin{bmatrix} \Gamma_\perp \\ q_\perp \end{bmatrix} = -\mathbf{A} \begin{bmatrix} \nabla_\perp n \\ \nabla_\perp T \end{bmatrix} + \begin{bmatrix} C n \\ 0 \end{bmatrix}
$$

Expanding this matrix gives the governing equations for our fluxes:
1. **Particle Flux:** $\Gamma_\perp = -A_{11}\nabla_\perp n - A_{12}\nabla_\perp T + C n$
2. **Conductive Heat Flux:** $q_\perp = -A_{21}\nabla_\perp n - A_{22}\nabla_\perp T$

### Understanding the Terms:
* **$A_{11}$ (Pure Diffusion):** Particle flux driven by the density gradient.
* **$A_{12}$ (Thermodiffusion / Soret Effect):** Particle flux driven by the temperature gradient. This term is often responsible for driving inward particle pinches.
* **$A_{21}$ (Dufour Effect):** Heat flux driven by the density gradient.
* **$A_{22}$ (Thermal Conductivity):** Heat flux driven by the temperature gradient.
* **$C$ (Convection):** A ballistic velocity term. In the near and far SOL, massive filaments ("blobs") carry particles outwards independently of the local gradient.

*Note: The matrix elements $A_{jk}$ are evaluated analytically based on two base parameters ($D_0$ and $\alpha$). The script supports two analytical derivations: a `gaussian` model and a `sincx` model.*

---

## 2. The Two-Channel System & Ambipolarity

Electrons and ions have vastly different masses ($m_i \gg m_e$). If they diffused independently, they would separate and generate massive electric fields. To maintain plasma quasineutrality, a radial electric field naturally forms to equalize their macroscopic transport. This is the **ambipolarity constraint**:
$$\Gamma_e = \Gamma_i = \Gamma_{\text{total}}$$

To model this, we split the system into two coupled channels, each with their own set of transport parameters ($D_0, \alpha, C$):

**Electron Channel:**
$$\Gamma_e = -A_{11}^e \nabla n - A_{12}^e \nabla T_e + C_e n$$
$$q_e = -A_{21}^e \nabla n - A_{22}^e \nabla T_e$$

**Ion Channel:**
$$\Gamma_i = -A_{11}^i \nabla n - A_{12}^i \nabla T_i + C_i n$$
$$q_i = -A_{21}^i \nabla n - A_{22}^i \nabla T_i$$

The mass disparity between the species will naturally manifest in the optimized parameters (expect $D_{0e} \neq D_{0i}$). 

---

## 3. The Optimization Problem

To find the physical parameters ($D_{0e}, \alpha_e, C_e, D_{0i}, \alpha_i, C_i$), we must fit our analytical equations to the ground-truth data ($\Gamma_{\text{data}}, q_{e,\text{data}}, q_{i,\text{data}}$).

Because particle flux ($\sim 10^{20}$) and heat flux ($\sim 10^4$) exist on completely different scales, standard error calculations would cause the optimizer to ignore the heat flux entirely. We solve this using a **Normalized Mean Squared Error (NMSE)**.

The objective (loss) function $J$ that the code minimizes is:

$$ 
J = W_1 \left( \frac{\Gamma_e - \Gamma_{\text{data}}}{\max|\Gamma_{\text{data}}|} \right)^2 + 
W_2 \left( \frac{\Gamma_i - \Gamma_{\text{data}}}{\max|\Gamma_{\text{data}}|} \right)^2 + 
W_3 \left( \frac{q_e - q_{e,\text{data}}}{\max|q_{e,\text{data}}|} \right)^2 + 
W_4 \left( \frac{q_i - q_{i,\text{data}}}{\max|q_{i,\text{data}}|} \right)^2 
$$

* **Ambipolarity:** By forcing both $\Gamma_e$ and $\Gamma_i$ to match the shared $\Gamma_{\text{data}}$, the ambipolarity constraint is implicitly enforced.
* **Weights ($W_k$):** The equations are heavily constrained. The user can adjust $W_1 \dots W_4$ to force the optimizer to prioritize fitting the particle flux over the heat flux, or vice versa.

---

## 4. Numerical Techniques

* **The Optimizer (`scipy.optimize.minimize`):** The script utilizes the **L-BFGS-B** algorithm. This is a quasi-Newton method that approximates the Hessian matrix to efficiently navigate complex parameter spaces. The "-B" stands for "Bounded," allowing us to enforce physical realities (e.g., $\alpha$ cannot be negative).
* **Point-by-Point vs. Regional Fitting:**
    * *Regional Fitting (Regularization):* Splits the spatial domain into $N$ distinct zones. By forcing the parameters to remain constant across a region, we extract smooth, macroscopic effective transport coefficients that correspond to actual physical zones (e.g., Core, Pinch-Zone, Separatrix, Far-SOL).
    * *Point-by-Point:* Calculates an exact fit for every individual spatial coordinate. Normally, it produces smooth fits with interesting regions, check. 

---

## 5. Quick Start Guide

### Dependencies
Ensure you have a standard scientific Python environment installed:
```bash
pip install numpy scipy matplotlib
```
### Running the Script
Run the optimization script: 

```bash
python two_channel_fitter.py
``` 

### Using the interface
modify in scripts or just in interactive plots:
- CHOSEN_MODEL = "gaussian" or "sincx"
- N_regions slider (in interactive plot) to adjust the number of regions for regional fitting (or set to a high number for point-by-point fitting)
- Use weights W1-W4 in the objective function to prioritize fitting certain fluxes over others.