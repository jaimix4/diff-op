from fit_fluxes_convection import Gamma_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 0. PLOT FORMATTING & A4 PAGE SETUP
# ==========================================
# A4 width = 8.27 in. 2.5 cm margins (~1 in) leaves ~6.27 in wide.
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
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
n_raw = data[:, 1]
Te_raw = data[:, 2]        
Ti_raw = data[:, 3]
dn_dR_raw = data[:, 4]
dTe_dR_raw = data[:, 5]
dTi_dR_raw = data[:, 6]
D_raw = data[:, 7]
chi_e_raw = data[:, 8]
chi_i_raw = data[:, 9]
Gamma_raw = data[:, 10]
Qe_total_raw = data[:, 11]
Qi_total_raw = data[:, 12]

# Smooth data as requested
smooth_sigma = 1.0
n = gaussian_filter1d(n_raw, sigma=smooth_sigma)
Te = gaussian_filter1d(Te_raw, sigma=smooth_sigma)
Ti = gaussian_filter1d(Ti_raw, sigma=smooth_sigma)
dn_dR = gaussian_filter1d(dn_dR_raw, sigma=smooth_sigma)
dTe_dR = gaussian_filter1d(dTe_dR_raw, sigma=smooth_sigma)
dTi_dR = gaussian_filter1d(dTi_dR_raw, sigma=smooth_sigma)
D = gaussian_filter1d(D_raw, sigma=smooth_sigma)
chi_e = gaussian_filter1d(chi_e_raw, sigma=smooth_sigma)
chi_i = gaussian_filter1d(chi_i_raw, sigma=smooth_sigma)
Gamma = gaussian_filter1d(Gamma_raw, sigma=smooth_sigma)
Qe = gaussian_filter1d(Qe_total_raw, sigma=smooth_sigma)
Qi = gaussian_filter1d(Qi_total_raw, sigma=smooth_sigma)

e_charge = 1.602e-19 # J/eV

# Apply normalizations based on the reference images
n_norm = n / 1e19
Te_norm = Te / 200.0
Ti_norm = Ti / 200.0

dn_norm = -dn_dR / 1e19
dTe_norm = -dTe_dR / 200.0
dTi_norm = -dTi_dR / 200.0

Gamma_norm = Gamma / 1e19

# To get the heat flux into the specific eV-normalized scale shown in the plot, 
# we divide by n0 (1e19), T0 (200), and e_charge.
norm_factor_heat = (1e19 * 200 * e_charge)

# qe_data = Qe_total_data - (1.5 * Gamma_data * Te * e_charge)
# qi_data = Qi_total_data - (1.5 * Gamma_data * Ti * e_charge)


Qe_norm = (Qe - (1.5 * Gamma * Te * e_charge)) / norm_factor_heat
Qi_norm = (Qi - (1.5 * Gamma * Ti * e_charge))/ norm_factor_heat
# Qi_norm = Qi / norm_factor_heat

RHO_SEPARATRIX = 1.0

# ==========================================
# 2. PLOTTING THE FIGURE
# ==========================================
# Set height to 9.5 to accommodate 5 panels gracefully without squashing
fig, axs = plt.subplots(5, 1, figsize=(6.27, 9.5), sharex=True)
plt.subplots_adjust(left=0.15, right=0.92, top=0.95, bottom=0.08, hspace=0.30)

LW = 2.0
AL = 0.9 

# Color Palette mapping to the reference images
c_n = '#1f77b4'  # Blue
c_e = '#d62728'  # Red
c_i = '#daa520'  # Goldenrod/Yellow

# --- PANEL 1: Normalized Profiles ---
axs[0].plot(rho, n_norm, color=c_n, lw=LW, alpha=AL, label=r'$\frac{n_e}{10^{19}} [\mathrm{m}^{-3}]$')
axs[0].plot(rho, Te_norm, color=c_e, lw=LW, alpha=AL, label=r'$\frac{T_e}{200} [\mathrm{eV}]$')
axs[0].plot(rho, Ti_norm, color=c_i, lw=LW, alpha=AL, label=r'$\frac{T_i}{200} [\mathrm{eV}]$')
axs[0].set_ylabel("a)", rotation=0, labelpad=15)
axs[0].set_title("Normalized Profiles")
axs[0].legend(loc='upper right')
axs[0].set_ylim(bottom=0)  # Adjusted to ensure all normalized profiles fit well

# --- PANEL 2: Normalized Gradients ---
axs[1].plot(rho, dn_norm, color=c_n, lw=LW, alpha=AL, label=r'$-\frac{\partial_r n_e}{10^{19}} [\mathrm{m}^{-4}]$')
axs[1].plot(rho, dTe_norm, color=c_e, lw=LW, alpha=AL, label=r'$-\frac{\partial_r T_e}{200} [\mathrm{eV}\cdot\mathrm{m}^{-1}]$')
axs[1].plot(rho, dTi_norm, color=c_i, lw=LW, alpha=AL, label=r'$-\frac{\partial_r T_i}{200} [\mathrm{eV}\cdot\mathrm{m}^{-1}]$')
axs[1].set_ylabel("b)", rotation=0, labelpad=15)
axs[1].set_title("Normalized Radial Gradient Profiles")
axs[1].legend(loc='upper right')
axs[1].set_ylim(bottom=0)  # Adjusted to ensure all normalized gradients fit well
# --- PANEL 3: Radial Transport Coefficients ---
axs[2].plot(rho, D, color=c_n, lw=LW, alpha=AL, label=r'$D_{r} [\mathrm{m}^2\cdot\mathrm{s}^{-1}]$')
axs[2].plot(rho, chi_e, color=c_e, lw=LW, alpha=AL, label=r'$\chi_{e,r} [\mathrm{m}^2\cdot\mathrm{s}^{-1}]$')
axs[2].plot(rho, chi_i, color=c_i, lw=LW, alpha=AL, label=r'$\chi_{i,r} [\mathrm{m}^2\cdot\mathrm{s}^{-1}]$')
axs[2].set_ylabel("c)", rotation=0, labelpad=15)
axs[2].set_title("Radial Transport Coefficients")
axs[2].legend(loc='upper left')

# --- PANEL 4: Normalized Radial Particle Fluxes ---
axs[3].plot(rho, Gamma_norm, color=c_n, lw=LW, alpha=AL, label=r'$\frac{\Gamma_{e,r}}{10^{19}} [\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$')
axs[3].set_ylabel("d)", rotation=0, labelpad=15)
axs[3].set_title("Normalized Radial Particle Fluxes")
axs[3].legend(loc='upper left')

# --- PANEL 5: Normalized Radial Heat Fluxes ---
axs[4].plot(rho, Qe_norm, color=c_e, lw=LW, alpha=AL, label=r'$\frac{q_{e,r}}{200} [\mathrm{eV}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$')
axs[4].plot(rho, Qi_norm, color=c_i, lw=LW, alpha=AL, label=r'$\frac{q_{i,r}}{200} [\mathrm{eV}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$')
axs[4].set_ylabel("e)", rotation=0, labelpad=15)
axs[4].set_title("Normalized Radial Conductive Heat Fluxes")
axs[4].set_xlabel(r"$\rho_{pol}[-]$")
axs[4].legend(loc='upper left')

# Align Y-axis labels and format grids/ticks
fig.align_ylabels(axs)

for ax in axs:
    ax.axvline(RHO_SEPARATRIX, color='k', linestyle='--', alpha=0.8, zorder=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, which='both')

plt.savefig("master_data_profiles_normalized.png", dpi=300, bbox_inches='tight')
plt.show()