import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 0. PLOT FORMATTING FOR POWERPOINT
# ==========================================
# Increased font sizes for presentation visibility
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.bottom': True,
    'ytick.left': True,
    'axes.formatter.use_mathtext': True
})

# ==========================================
# 1. LOAD AND SMOOTH DATA
# ==========================================
data = np.loadtxt("aug36190_master_dataset_2.csv", delimiter=",", skiprows=1)

rho = data[:, 0]
Te_raw = data[:, 2]        
Ti_raw = data[:, 3]
Gamma_raw = data[:, 10]
Qe_total_raw = data[:, 11]
Qi_total_raw = data[:, 12]

# Smooth data as in your updated script
smooth_sigma = 1.0
Te = gaussian_filter1d(Te_raw, sigma=smooth_sigma)
Ti = gaussian_filter1d(Ti_raw, sigma=smooth_sigma)
Gamma = gaussian_filter1d(Gamma_raw, sigma=smooth_sigma)
Qe = gaussian_filter1d(Qe_total_raw, sigma=smooth_sigma)
Qi = gaussian_filter1d(Qi_total_raw, sigma=smooth_sigma)

e_charge = 1.602e-19 # J/eV

# ==========================================
# 2. NORMALIZATIONS & HEAT FLUX CORRECTION
# ==========================================
# Normalization factors
norm_factor_particle = 1e19
norm_factor_heat = (1e19 * 200 * e_charge)

Gamma_norm = Gamma / norm_factor_particle

# Correcting to conductive heat flux (q) and normalizing
Qe_norm = (Qe - (1.5 * Gamma * Te * e_charge)) / norm_factor_heat
Qi_norm = (Qi - (1.5 * Gamma * Ti * e_charge)) / norm_factor_heat

RHO_SEPARATRIX = 1.0

# ==========================================
# 3. PLOTTING THE FIGURE (1x2 Square Layout)
# ==========================================
# figsize=(12, 6) creates two side-by-side 6x6 (square) plots, perfect for 16:9 slides
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.15, wspace=0.3)

LW = 2.5
AL = 0.9 

# Color Palette
c_n = '#1f77b4'  # Blue
c_e = '#d62728'  # Red
c_i = '#daa520'  # Goldenrod/Yellow

# --- PANEL 1: Normalized Radial Particle Flux ---
axs[0].plot(rho, Gamma_norm, color=c_n, lw=LW, alpha=AL, label=r'$\frac{\Gamma_r}{10^{19}} [\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$')
#axs[0].set_ylabel("Normalized Radial Particle Flux")
axs[0].set_title("Normalized Radial Particle Flux")
axs[0].set_xlabel(r"$\rho_{pol}[-]$")
axs[0].legend(loc='upper left')

# --- PANEL 2: Normalized Radial Heat Fluxes ---
axs[1].plot(rho, Qe_norm, color=c_e, lw=LW, alpha=AL, label=r'$\frac{q_{e,r}}{200} [\mathrm{eV}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$')
axs[1].plot(rho, Qi_norm, color=c_i, lw=LW, alpha=AL, label=r'$\frac{q_{i,r}}{200} [\mathrm{eV}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$')
#axs[1].set_ylabel("Normalized Radial Heat Fluxes")
axs[1].set_title("Normalized Radial Conductive Heat Fluxes")
axs[1].set_xlabel(r"$\rho_{pol}[-]$")
axs[1].legend(loc='upper left')

# Common formatting for both panels
for ax in axs:
    ax.axvline(RHO_SEPARATRIX, color='k', linestyle='--', alpha=0.8, zorder=0, lw=1.5)
    ax.grid(True, alpha=0.4)
    # Ensure ticks point inward on all sides
    ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, which='both', length=6)

plt.savefig("presentation_fluxes_square.png", dpi=300, bbox_inches='tight')
plt.show()