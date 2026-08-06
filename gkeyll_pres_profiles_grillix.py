import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import FormatStrFormatter

# ==========================================
# 0. PLOT FORMATTING (POWERPOINT & LATEX)
# ==========================================
# Use serif fonts and Computer Modern math text for a native LaTeX look
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 14,
    'axes.titlesize': 18,    # Large titles for slides
    'axes.labelsize': 16,    # Large axis labels
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
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
Gamma_raw = data[:, 10]
Qe_total_raw = data[:, 11]
Qi_total_raw = data[:, 12]

# Smooth data
smooth_sigma = 1.0
n = gaussian_filter1d(n_raw, sigma=smooth_sigma)
Te = gaussian_filter1d(Te_raw, sigma=smooth_sigma)
Ti = gaussian_filter1d(Ti_raw, sigma=smooth_sigma)
Gamma = gaussian_filter1d(Gamma_raw, sigma=smooth_sigma)
Qe = gaussian_filter1d(Qe_total_raw, sigma=smooth_sigma)
Qi = gaussian_filter1d(Qi_total_raw, sigma=smooth_sigma)

e_charge = 1.602e-19 # J/eV
RHO_SEPARATRIX = 1.0

# ==========================================
# 2. NORMALIZATIONS & CONDUCTIVE CORRECTION
# ==========================================
norm_factor_particle = 1e19
norm_factor_heat = (1e19 * 200 * e_charge)

# Profile normalizations
n_norm = n / norm_factor_particle
Te_norm = Te / 200.0
Ti_norm = Ti / 200.0

# Flux normalizations
Gamma_norm = Gamma / norm_factor_particle

# Correcting total heat flux (Q) to conductive heat flux (q) and normalizing
Qe_norm = (Qe - (1.5 * Gamma * Te * e_charge)) / norm_factor_heat
Qi_norm = (Qi - (1.5 * Gamma * Ti * e_charge)) / norm_factor_heat

# ==========================================
# COMMON VISUAL SETTINGS
# ==========================================
LW = 4.0 # Thicker lines for projector visibility
AL = 0.9 

# Distinct Color Palette
c_n = '#1f77b4'  # Blue
c_e = '#d62728'  # Red
c_i = '#daa520'  # Goldenrod/Yellow

def format_axis(ax):
    """Applies common presentation formatting to a given axis."""
    ax.axvline(RHO_SEPARATRIX, color='k', linestyle='--', alpha=0.8, zorder=0, lw=2.0)
    ax.grid(True, alpha=0.3)
    # Ensure ticks point inward on all sides with good visibility
    ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, which='both', length=6, width=1.5)
    
    ax.locator_params(axis='y', nbins=6)
    #ax.locator_params(axis='both', nbins=5)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Enhance spine (border) thickness for presentation clarity
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# ==========================================
# 3. FIGURE 1: NORMALIZED PROFILES
# ==========================================
fig1, ax1 = plt.subplots(figsize=(7*0.7, 6.5*0.7)) # Large, slightly wide format for slides

ax1.plot(rho, n_norm, color=c_n, lw=LW, alpha=AL, label=r'$n/10^{19} \, [\mathrm{m}^{-3}]$')
ax1.plot(rho, Te_norm, color=c_e, lw=LW, alpha=AL, label=r'$T_e/200 \, [\mathrm{eV}]$')
ax1.plot(rho, Ti_norm, color=c_i, lw=LW, alpha=AL, label=r'$T_i/200 \, [\mathrm{eV}]$')

# ax1.set_ylabel("Normalized Profiles")
# ax1.set_title("Normalized Profiles")
ax1.set_xlabel(r"$\rho_{pol} \, [-]$")
ax1.legend(loc='upper right', framealpha=0.9)
ax1.set_ylim(bottom=0) # Grounds the y-axis firmly at 0

format_axis(ax1)
fig1.tight_layout()
fig1.savefig("presentation_profiles_latex.png", dpi=300, bbox_inches='tight')

# ==========================================
# 4. FIGURE 2: FLUXES (1x2 Layout)
# ==========================================
# figsize=(14, 6.5) creates two square plots side-by-side on a 16:9 aspect ratio
fig2, axs2 = plt.subplots(1, 2, figsize=(14*0.7, 7*0.7))
plt.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.15, wspace=0.25)

# --- PANEL 1: Normalized Radial Particle Flux ---
axs2[0].plot(rho, Gamma_norm, color=c_n, lw=LW, alpha=AL, label=r'$\Gamma_r/10^{19} \, [\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$')
# axs2[0].set_ylabel("Normalized Radial Particle Flux")
# axs2[0].set_title("Normalized Radial Particle Flux")
axs2[0].set_xlabel(r"$\rho_{pol} \, [-]$")
axs2[0].legend(loc='upper left', framealpha=0.9)

# --- PANEL 2: Normalized Radial Heat Fluxes ---
axs2[1].plot(rho, Qe_norm, color=c_e, lw=LW, alpha=AL, label=r'$q_{e,r}/200 \, [\mathrm{eV}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$')
axs2[1].plot(rho, Qi_norm, color=c_i, lw=LW, alpha=AL, label=r'$q_{i,r}/200 \, [\mathrm{eV}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$')
# axs2[1].set_ylabel("Normalized Radial Conductive Heat Fluxes")
# axs2[1].set_title("Normalized Radial Conductive Heat Fluxes")
axs2[1].set_xlabel(r"$\rho_{pol} \, [-]$")
axs2[1].legend(loc='upper left', framealpha=0.9)
axs2[1].set_ylim(bottom=0.0)  # Ensures the negative pinch is visible

for ax in axs2:
    # Anchor the y=0 line clearly, since setting bottom=0 would cut off the negative pinch
    ax.axhline(0, color='black', linewidth=1.0, alpha=0.3, zorder=0)
    format_axis(ax)

fig2.savefig("presentation_fluxes_latex_square.png", dpi=300, bbox_inches='tight')

# Display both figures at once
plt.show()