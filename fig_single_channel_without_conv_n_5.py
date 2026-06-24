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
# 2. DEFINING THE SINCX MODEL (Without Convection)
# ==========================================
def compute_model(B_e, beta_e, n_val, T_val, dn_val, dT_val):
    # Using the sincx equations with D0 -> B_e and alpha -> beta_e
    A11 = B_e * np.exp((-beta_e**2) / 4)
    A12 = - B_e * np.exp((-beta_e**2) / 4) * ((n_val*beta_e**2)/(4*T_val))
    A21 = - B_e * np.exp((-beta_e**2) / 4) * ((T_val*beta_e**2)/(4))
    A22 = B_e * np.exp((-beta_e**2) / 4) * (n_val*(24 - 8*beta_e**2 + beta_e**4)/16)
        
    G_n = -1 * A11 * dn_val
    G_T = -1 * A12 * dT_val
    G_tot = G_n + G_T
    
    q_n = -1 * (A21 * dn_val) * e_charge
    q_T = -1 * (A22 * dT_val) * e_charge
    q_tot = q_n + q_T
    
    return G_tot, q_tot, G_n, G_T, q_n, q_T

def objective_function(params, n_val, T_val, dn_val, dT_val, G_true, q_true):
    B_e, beta_e = params
    G_mod, q_mod, *_ = compute_model(B_e, beta_e, n_val, T_val, dn_val, dT_val)
    
    norm_G = np.max(np.abs(G_true)) + 1e-30
    norm_q = np.max(np.abs(q_true)) + 1e-30
    
    err_G = np.mean(((G_true - G_mod) / norm_G)**2)
    err_q = np.mean(((q_true - q_mod) / norm_q)**2)
    
    return err_G + err_q

# ==========================================
# 3. REGIONAL OPTIMIZATION (N=5: 3 inside, 2 outside)
# ==========================================
# Create custom edges to force the 3/2 split at the separatrix
edges_in = np.linspace(rho.min(), RHO_SEPARATRIX, 4)  # 3 regions
edges_out = np.linspace(RHO_SEPARATRIX, rho.max(), 3) # 2 regions
edges = np.concatenate((edges_in[:-1], edges_out))    # Total 5 regions

out = {k: np.zeros_like(rho) for k in ['Be', 'beta_e', 'G_tot', 'q_tot', 'G_n', 'G_T', 'q_n', 'q_T']}
bnds = ((None, None), (-1.0, 20.0)) 
initial_guess = [1.0, 1.0]

for i in range(len(edges) - 1):
    r_min = edges[i]
    if i == len(edges) - 2:
        mask = (rho >= r_min) & (rho <= edges[i+1])
    else:
        mask = (rho >= r_min) & (rho < edges[i+1])
        
    if not np.any(mask): continue
        
    args = (n[mask], Te[mask], dn_dR[mask], dTe_dR[mask], Gamma_data[mask], qe_data[mask])
    res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B', args=args)
    
    out['Be'][mask] = res.x[0]
    out['beta_e'][mask] = res.x[1]
    
    comps = compute_model(res.x[0], res.x[1], n[mask], Te[mask], dn_dR[mask], dTe_dR[mask])
    out['G_tot'][mask], out['q_tot'][mask] = comps[0], comps[1]
    out['G_n'][mask], out['G_T'][mask] = comps[2], comps[3]
    out['q_n'][mask], out['q_T'][mask] = comps[4], comps[5]


# ==========================================
# 4. PLOTTING THE FIGURE
# ==========================================
# Exact width for A4 with margins, slightly taller to accommodate increased hspace
fig, axs = plt.subplots(3, 1, figsize=(7.0, 5.8), sharex=True)

# Increased hspace for vertical breathing room
plt.subplots_adjust(left=0.15, right=0.90, top=0.93, bottom=0.10, hspace=0.20)

LW = 1.5
AL = 0.85 

# --- PANEL 1: Parameters ---
# Blue Be axis
axs[0].step(rho, out['Be'], color='blue', lw=LW, where='post', label=r'$A_e [-]$')
axs[0].step(rho, out['beta_e'], color='green', lw=LW, where='post', label=r'$\alpha_e [-]$')
axs[0].tick_params(axis='y')
axs[0].set_ylabel("Parameters operator", color='black')

# Red beta_e axis (Moved to the right side for much cleaner aesthetics)
# ax0_t = axs[0].twinx()
# ax0_t.step(rho, out['beta_e'], color='red', lw=LW, where='post', label=r'$\alpha_e [-]$')
# ax0_t.tick_params(axis='y', labelcolor='red', color='red', direction='in')
# We can optionally label the right axis, but the legend explains it perfectly

# Combine legends in top graph
# lines_1, labels_1 = axs[0].get_legend_handles_labels()
# lines_2, labels_2 = ax0_t.get_legend_handles_labels()
# axs[0].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
axs[0].legend(loc='upper right')

axs[0].set_title(r"Sinc Operator Electron Channel without Convection, $N_{regions}=5$")

# --- PANEL 2: Particle Flux ---
axs[1].plot(rho, Gamma_data, 'k--', lw=LW, label='GRILLIX')
axs[1].plot(rho, out['G_tot'], color='#4a00e0', lw=LW+0.5, alpha=AL, label=r'Model Total')
axs[1].plot(rho, out['G_n'], color="#00b0e0", lw=LW, ls='-.', alpha=AL, label=r'$D_{11}\nabla n_e$')
axs[1].plot(rho, out['G_T'], color='#ff8c00', lw=LW, ls='-.', alpha=AL, label=r'$D_{12}\nabla T_e$')
axs[1].set_ylabel(r"$\Gamma_e [m^{-2}\cdot s^{-1}]$")
axs[1].legend(loc='lower right', ncol=2)

# --- PANEL 3: Heat Flux ---
axs[2].plot(rho, qe_data, 'k--', lw=LW, label='GRILLIX')
axs[2].plot(rho, out['q_tot'], color="#dd0606", lw=LW+0.5, alpha=AL, label=r'Model Total')
axs[2].plot(rho, out['q_T'], color="#00b0e0", lw=LW, ls='-.', alpha=AL, label=r'$D_{22}\nabla T_e$')
axs[2].plot(rho, out['q_n'], color='#ff8c00', lw=LW, ls='-.', alpha=AL, label=r'$D_{21}\nabla n_e$')
axs[2].set_ylabel(r"$q_e [W\cdot m^{-2}]$")
axs[2].set_xlabel(r"$\rho_{pol}[-]$")

# Scientific notation for Heat Flux
axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[2].yaxis.get_offset_text().set_fontsize(8)

axs[2].legend(loc='lower right', ncol=2)

# Force perfectly aligned Y-axis labels across all subplots
fig.align_ylabels(axs)

# Add separatrix line and ensure all ticks are inward
# for ax in axs:
#     ax.axvline(RHO_SEPARATRIX, color='grey', linestyle='--', alpha=0.6, zorder=0)
#     ax.grid(True, alpha=0.25)
#     # Re-enforce inward ticks for main axes
#     ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, which='both')

for i, ax in enumerate(axs):
    ax.axvline(RHO_SEPARATRIX, color='grey', linestyle='--', alpha=0.6, zorder=0)
    ax.grid(True, alpha=0.25)
    
    # Check if we are formatting the first subplot panel
    if i == 0:
        # Prevent the primary (blue) y-axis from placing tick marks on the right side
        ax.tick_params(direction='in', top=True, right=False, bottom=True, left=True, which='both')
    else:
        # Standard inward ticks on all four sides for panel 1 and 2
        ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, which='both')

# axs[0].tick_params(axis='y', colors='blue', which='both')
# ax0_t.tick_params(axis='y', colors='red', right=True, left=False, which='both')

plt.savefig("single_channel_without_conv.png", dpi=300, bbox_inches='tight')
plt.show()