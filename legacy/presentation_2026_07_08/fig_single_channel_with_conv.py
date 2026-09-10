import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    'legend.fontsize': 8,
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
n = data[:, 1]
Te = data[:, 2]        

# Load raw arrays
dn_dR_raw = data[:, 4]
dTe_dR_raw = data[:, 5]
Gamma_raw = data[:, 10]
Qe_total_raw = data[:, 11]

e_charge = 1.602e-19 # J/eV

# Apply Gaussian filter to smooth out manual digitization noise
smooth_sigma = 2.0
dn_dR = gaussian_filter1d(dn_dR_raw, sigma=smooth_sigma)
dTe_dR = gaussian_filter1d(dTe_dR_raw, sigma=smooth_sigma)
Gamma_data = gaussian_filter1d(Gamma_raw, sigma=smooth_sigma)
Qe_total_data = gaussian_filter1d(Qe_total_raw, sigma=smooth_sigma)

# Compute pure conductive electron heat flux
qe_data = Qe_total_data - (1.5 * Gamma_data * Te * e_charge)

NUM_POINTS = len(rho)
RHO_SEPARATRIX = 1.0

# ==========================================
# 2. DEFINING THE SINCX MODEL (WITH CONVECTION)
# ==========================================
def compute_model(B_e, beta_e, C_conv, n_val, T_val, dn_val, dT_val):
    # Using the sincx equations with D0 -> B_e and alpha -> beta_e
    A11 = B_e * np.exp((-beta_e**2) / 4)
    A12 = - B_e * np.exp((-beta_e**2) / 4) * ((n_val*beta_e**2)/(4*T_val))
    A21 = - B_e * np.exp((-beta_e**2) / 4) * ((T_val*beta_e**2)/(4))
    A22 = B_e * np.exp((-beta_e**2) / 4) * (n_val*(24 - 8*beta_e**2 + beta_e**4)/16)
        
    G_n = -1 * A11 * dn_val
    G_T = -1 * A12 * dT_val
    G_C = C_conv * n_val
    G_tot = G_n + G_T + G_C
    
    q_n = -1 * (A21 * dn_val) * e_charge
    q_T = -1 * (A22 * dT_val) * e_charge
    q_tot = q_n + q_T
    
    return G_tot, q_tot, G_n, G_T, G_C, q_n, q_T

def objective_function(params, n_val, T_val, dn_val, dT_val, G_true, q_true):
    B_e, beta_e, C_conv = params
    G_mod, q_mod, *_ = compute_model(B_e, beta_e, C_conv, n_val, T_val, dn_val, dT_val)
    
    norm_G = np.max(np.abs(G_true)) + 1e-30
    norm_q = np.max(np.abs(q_true)) + 1e-30
    
    err_G = np.mean(((G_true - G_mod) / norm_G)**2)
    err_q = np.mean(((q_true - q_mod) / norm_q)**2)
    
    return err_G + err_q

# ==========================================
# 3. OPTIMIZATION (Exact Point-by-Point)
# ==========================================
out = {k: np.zeros_like(rho) for k in ['Be', 'beta_e', 'Ce', 'G_tot', 'q_tot', 'G_n', 'G_T', 'G_C', 'q_n', 'q_T']}
# Unbounded for B_e and C_conv, bounded for beta_e
bnds = ((None, None), (-1.0, 20.0), (None, None)) 
initial_guess = [1.0, 1.0, 0.0]

for i in range(NUM_POINTS):
    args = (np.array([n[i]]), np.array([Te[i]]), np.array([dn_dR[i]]), np.array([dTe_dR[i]]), 
            np.array([Gamma_data[i]]), np.array([qe_data[i]]))
    
    res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B', args=args)
    
    out['Be'][i] = res.x[0]
    out['beta_e'][i] = res.x[1]
    out['Ce'][i] = res.x[2]
    
    # Compute the physical flux components at this specific point
    comps = compute_model(res.x[0], res.x[1], res.x[2], n[i], Te[i], dn_dR[i], dTe_dR[i])
    out['G_tot'][i], out['q_tot'][i] = comps[0], comps[1]
    out['G_n'][i], out['G_T'][i] = comps[2], comps[3]
    out['G_C'][i] = comps[4]
    out['q_n'][i], out['q_T'][i] = comps[5], comps[6]


# ==========================================
# 4. PLOTTING THE FIGURE
# ==========================================
fig, axs = plt.subplots(3, 1, figsize=(7.0, 5.8), sharex=True)

# Spacing configured for vertical breathing room
plt.subplots_adjust(left=0.15, right=0.90, top=0.93, bottom=0.10, hspace=0.20)

LW = 1.5
AL = 0.85 

# --- PANEL 1: Parameters ---
axs[0].plot(rho, out['Be'], color='blue', lw=LW, label=r'$A_e [-]$')
axs[0].plot(rho, out['beta_e'], color='green', lw=LW, label=r'$\alpha_e [-]$')
axs[0].tick_params(axis='y')
axs[0].set_ylabel("Parameters operator", color='black')
axs[0].set_ylim(-0.5, 4.0)

# Add Convection to right twin axis
ax0_t = axs[0].twinx()
ax0_t.plot(rho, out['Ce'], color='purple', lw=LW, label=r'$C_e [m/s]$')
ax0_t.set_ylabel(r'Convection $C_e$', color='purple')
ax0_t.set_ylim(-60, 60)

# Merge legends
lines_1, labels_1 = axs[0].get_legend_handles_labels()
lines_2, labels_2 = ax0_t.get_legend_handles_labels()
axs[0].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', ncol=3)

axs[0].set_title(r"Sinc Operator Electron Channel with Convection, Exact Point-by-Point")

# --- PANEL 2: Particle Flux ---
axs[1].plot(rho, Gamma_data, 'k--', lw=LW, label='GRILLIX')
axs[1].plot(rho, out['G_tot'], color='#4a00e0', lw=LW+0.5, alpha=AL, label=r'Model Total')
axs[1].plot(rho, out['G_n'], color="#00b0e0", lw=LW, ls='-.', alpha=AL, label=r'$D_{11}\nabla n_e$')
axs[1].plot(rho, out['G_T'], color='#ff8c00', lw=LW, ls='-.', alpha=AL, label=r'$D_{12}\nabla T_e$')
axs[1].plot(rho, out['G_C'], color='#38b000', lw=LW, ls='-.', alpha=AL, label=r'$C_e n_e$ (Convection)')
axs[1].set_ylabel(r"$\Gamma_e [m^{-2}\cdot s^{-1}]$")
axs[1].legend(loc='upper left', ncol=2)
axs[1].set_ylim(-3e20, 4e20)  # Dynamic y-limits based on data

# --- PANEL 3: Heat Flux ---
axs[2].plot(rho, qe_data, 'k--', lw=LW, label='GRILLIX')
axs[2].plot(rho, out['q_tot'], color="#dd0606", lw=LW+0.5, alpha=AL, label=r'Model Total')
axs[2].plot(rho, out['q_T'], color="#00b0e0", lw=LW, ls='-.', alpha=AL, label=r'$D_{22}\nabla T_e$')
axs[2].plot(rho, out['q_n'], color='#ff8c00', lw=LW, ls='-.', alpha=AL, label=r'$D_{21}\nabla n_e$')
axs[2].set_ylabel(r"$q_e [W\cdot m^{-2}]$")
axs[2].set_xlabel(r"$\rho_{pol}[-]$")
axs[2].set_ylim(-7e3, 9e3)  

# Scientific notation for Heat Flux
axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[2].yaxis.get_offset_text().set_fontsize(8)

axs[2].legend(loc='upper left', ncol=2)

# Force perfectly aligned Y-axis labels across all subplots
fig.align_ylabels(axs)

# Add separatrix line and ensure all ticks are inward
for i, ax in enumerate(axs):
    ax.axvline(RHO_SEPARATRIX, color='grey', linestyle='--', alpha=0.6, zorder=0)
    ax.grid(True, alpha=0.25)
    
    # Check if we are formatting the first subplot panel
    if i == 0:
        # Prevent the primary (blue/green) y-axis from placing tick marks on the right side
        ax.tick_params(direction='in', top=True, right=False, bottom=True, left=True, which='both')
    else:
        # Standard inward ticks on all four sides for panel 1 and 2
        ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, which='both')

# Re-apply explicit colored ticks for twinx right side
ax0_t.tick_params(axis='y', colors='purple', direction='in', right=True, left=False, which='both')

plt.savefig("single_channel_with_conv_point_by_point.png", dpi=300, bbox_inches='tight')
plt.show()