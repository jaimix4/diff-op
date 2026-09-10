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
Ti = data[:, 3]

# Load raw arrays
dn_dR_raw = data[:, 4]
dTe_dR_raw = data[:, 5]
dTi_dR_raw = data[:, 6]
Gamma_raw = data[:, 10]
Qe_total_raw = data[:, 11]
Qi_total_raw = data[:, 12]

e_charge = 1.602e-19 # J/eV

# Apply Gaussian filter to smooth out manual digitization noise
smooth_sigma = 2.0
dn_dR = gaussian_filter1d(dn_dR_raw, sigma=smooth_sigma)
dTe_dR = gaussian_filter1d(dTe_dR_raw, sigma=smooth_sigma)
dTi_dR = gaussian_filter1d(dTi_dR_raw, sigma=smooth_sigma)
Gamma_data = gaussian_filter1d(Gamma_raw, sigma=smooth_sigma)
Qe_total_data = gaussian_filter1d(Qe_total_raw, sigma=smooth_sigma)
Qi_total_data = gaussian_filter1d(Qi_total_raw, sigma=smooth_sigma)

# Compute pure conductive heat fluxes
qe_data = Qe_total_data - (1.5 * Gamma_data * Te * e_charge)
qi_data = Qi_total_data - (1.5 * Gamma_data * Ti * e_charge)

NUM_POINTS = len(rho)
RHO_SEPARATRIX = 1.0

# ==========================================
# 2. DEFINING THE SINCX MODEL (WITH CONVECTION)
# ==========================================
def compute_single_channel(A_param, alpha_param, C_conv, n_val, T_val, dn_val, dT_val):
    # Using the sincx equations with D0 -> A_param and alpha -> alpha_param
    A11 = A_param * np.exp((-alpha_param**2) / 4)
    A12 = - A_param * np.exp((-alpha_param**2) / 4) * ((n_val*alpha_param**2)/(4*T_val))
    A21 = - A_param * np.exp((-alpha_param**2) / 4) * ((T_val*alpha_param**2)/(4))
    A22 = A_param * np.exp((-alpha_param**2) / 4) * (n_val*(24 - 8*alpha_param**2 + alpha_param**4)/16)
        
    G_n = -1 * A11 * dn_val
    G_T = -1 * A12 * dT_val
    G_C = C_conv * n_val
    G_tot = G_n + G_T + G_C
    
    q_n = -1 * (A21 * dn_val) * e_charge
    q_T = -1 * (A22 * dT_val) * e_charge
    q_tot = q_n + q_T
    
    return G_tot, q_tot, G_n, G_T, G_C, q_n, q_T

def objective_function(params, n_val, Te_val, Ti_val, dn_val, dTe_val, dTi_val, G_true, qe_true, qi_true):
    Ae, alpha_e, Ce, Ai, alpha_i, Ci = params
    
    Ge_mod, qe_mod, *_ = compute_single_channel(Ae, alpha_e, Ce, n_val, Te_val, dn_val, dTe_val)
    Gi_mod, qi_mod, *_ = compute_single_channel(Ai, alpha_i, Ci, n_val, Ti_val, dn_val, dTi_val)
    
    norm_G = np.max(np.abs(G_true)) + 1e-30
    norm_qe = np.max(np.abs(qe_true)) + 1e-30
    norm_qi = np.max(np.abs(qi_true)) + 1e-30
    
    # Ambipolarity forces Ge and Gi to match G_true independently
    err_Ge = np.mean(((G_true - Ge_mod) / norm_G)**2)
    err_Gi = np.mean(((G_true - Gi_mod) / norm_G)**2)
    err_qe = np.mean(((qe_true - qe_mod) / norm_qe)**2)
    err_qi = np.mean(((qi_true - qi_mod) / norm_qi)**2)
    
    # Weights assumed to be 1.0 for the plotting script
    return err_Ge + err_Gi + err_qe + err_qi

# ==========================================
# 3. OPTIMIZATION (Exact Point-by-Point)
# ==========================================
keys = ['Ae', 'alpha_e', 'Ce', 'Ai', 'alpha_i', 'Ci', 
        'Ge_tot', 'Ge_n', 'Ge_T', 'Ge_C', 'Gi_tot', 'Gi_n', 'Gi_T', 'Gi_C',
        'qe_tot', 'qe_n', 'qe_T', 'qi_tot', 'qi_n', 'qi_T']
out = {k: np.zeros_like(rho) for k in keys}

# Unbounded for A and C_conv, bounded for alpha
bnds = ((None, None), (-1.0, 20.0), (None, None),
        (None, None), (-1.0, 20.0), (None, None)) 
initial_guess = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]

for i in range(NUM_POINTS):
    args = (np.array([n[i]]), np.array([Te[i]]), np.array([Ti[i]]), 
            np.array([dn_dR[i]]), np.array([dTe_dR[i]]), np.array([dTi_dR[i]]), 
            np.array([Gamma_data[i]]), np.array([qe_data[i]]), np.array([qi_data[i]]))
    
    res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B', args=args)
    
    out['Ae'][i], out['alpha_e'][i], out['Ce'][i] = res.x[0:3]
    out['Ai'][i], out['alpha_i'][i], out['Ci'][i] = res.x[3:6]
    
    # Compute the physical flux components at this specific point
    e_comps = compute_single_channel(res.x[0], res.x[1], res.x[2], n[i], Te[i], dn_dR[i], dTe_dR[i])
    out['Ge_tot'][i], out['qe_tot'][i], out['Ge_n'][i], out['Ge_T'][i], out['Ge_C'][i], out['qe_n'][i], out['qe_T'][i] = e_comps
    
    i_comps = compute_single_channel(res.x[3], res.x[4], res.x[5], n[i], Ti[i], dn_dR[i], dTi_dR[i])
    out['Gi_tot'][i], out['qi_tot'][i], out['Gi_n'][i], out['Gi_T'][i], out['Gi_C'][i], out['qi_n'][i], out['qi_T'][i] = i_comps


# ==========================================
# 4. PLOTTING THE FIGURE
# ==========================================
# Height increased to 10.5 to fit 6 panels perfectly without squashing them
fig, axs = plt.subplots(6, 1, figsize=(7.0, 10.5), sharex=True)

# Spacing configured for vertical breathing room
plt.subplots_adjust(left=0.15, right=0.88, top=0.95, bottom=0.06, hspace=0.30)

LW = 1.5
AL = 0.85 

# --- PANEL 1: Electron Parameters ---
axs[0].plot(rho, out['Ae'], color='blue', lw=LW, label=r'$A_e [-]$')
axs[0].plot(rho, out['alpha_e'], color='green', lw=LW, label=r'$\alpha_e [-]$')
axs[0].tick_params(axis='y')
axs[0].set_ylabel("Electron Params", color='black')

ax0_t = axs[0].twinx()
ax0_t.plot(rho, out['Ce'], color='purple', lw=LW, label=r'$C_e [m/s]$')
ax0_t.set_ylabel(r'Convection $C_e$', color='purple')

lines_1, labels_1 = axs[0].get_legend_handles_labels()
lines_2, labels_2 = ax0_t.get_legend_handles_labels()
axs[0].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', ncol=2)
axs[0].set_title(r"Two-Channel Sinc Operator with Convection, Exact Point-by-Point")

# --- PANEL 2: Ion Parameters ---
axs[1].plot(rho, out['Ai'], color='#c00000', lw=LW, label=r'$A_i [-]$') # Dark red
axs[1].plot(rho, out['alpha_i'], color='#d9a400', lw=LW, label=r'$\alpha_i [-]$') # Golden
axs[1].tick_params(axis='y')
axs[1].set_ylabel("Ion Params", color='black')

ax1_t = axs[1].twinx()
ax1_t.plot(rho, out['Ci'], color='#8b4513', lw=LW, label=r'$C_i [m/s]$') # Saddle brown
ax1_t.set_ylabel(r'Convection $C_i$', color='#8b4513')

lines_3, labels_3 = axs[1].get_legend_handles_labels()
lines_4, labels_4 = ax1_t.get_legend_handles_labels()
axs[1].legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right', ncol=2)

# --- PANEL 3: Electron Particle Flux ---
axs[2].plot(rho, Gamma_data, 'k--', lw=LW, label='GRILLIX Total')
axs[2].plot(rho, out['Ge_tot'], color='#4a00e0', lw=LW+0.5, alpha=AL, label=r'Model Total ($\Gamma_e$)')
axs[2].plot(rho, out['Ge_n'], color="#00b0e0", lw=LW, ls='-.', alpha=AL, label=r'$D_{11}^e\nabla n$')
axs[2].plot(rho, out['Ge_T'], color='#ff8c00', lw=LW, ls='-.', alpha=AL, label=r'$D_{12}^e\nabla T_e$')
axs[2].plot(rho, out['Ge_C'], color='#38b000', lw=LW, ls=':', alpha=AL, label=r'$C_e n$ (Conv)')
axs[2].set_ylabel(r"$\Gamma_e [m^{-2}\cdot s^{-1}]$")
axs[2].legend(loc='upper left', ncol=3)
axs[2].set_ylim(-3e20, 4e20)

# --- PANEL 4: Ion Particle Flux ---
axs[3].plot(rho, Gamma_data, 'k--', lw=LW, label='GRILLIX Total')
axs[3].plot(rho, out['Gi_tot'], color='#e60000', lw=LW+0.5, alpha=AL, label=r'Model Total ($\Gamma_i$)')
axs[3].plot(rho, out['Gi_n'], color="#ff8c00", lw=LW, ls='-.', alpha=AL, label=r'$D_{11}^i\nabla n$')
axs[3].plot(rho, out['Gi_T'], color='#ff0055', lw=LW, ls='-.', alpha=AL, label=r'$D_{12}^i\nabla T_i$')
axs[3].plot(rho, out['Gi_C'], color='#8b4513', lw=LW, ls=':', alpha=AL, label=r'$C_i n$ (Conv)')
axs[3].set_ylabel(r"$\Gamma_i [m^{-2}\cdot s^{-1}]$")
axs[3].legend(loc='upper left', ncol=3)
axs[3].set_ylim(-3e20, 4e20)

# --- PANEL 5: Electron Heat Flux ---
axs[4].plot(rho, qe_data, 'k--', lw=LW, label='GRILLIX ($q_e$)')
axs[4].plot(rho, out['qe_tot'], color="#4a00e0", lw=LW+0.5, alpha=AL, label=r'Model Total ($q_e$)')
axs[4].plot(rho, out['qe_T'], color="#00b0e0", lw=LW, ls='-.', alpha=AL, label=r'$D_{22}^e\nabla T_e$')
axs[4].plot(rho, out['qe_n'], color='#ff8c00', lw=LW, ls='-.', alpha=AL, label=r'$D_{21}^e\nabla n$')
axs[4].set_ylabel(r"$q_e [W\cdot m^{-2}]$")
axs[4].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[4].yaxis.get_offset_text().set_fontsize(8)
axs[4].legend(loc='upper left', ncol=2)
axs[4].set_ylim(-7e3, 9e3)  

# --- PANEL 6: Ion Heat Flux ---
axs[5].plot(rho, qi_data, 'k--', lw=LW, label='GRILLIX ($q_i$)')
axs[5].plot(rho, out['qi_tot'], color="#e60000", lw=LW+0.5, alpha=AL, label=r'Model Total ($q_i$)')
axs[5].plot(rho, out['qi_T'], color="#ff0055", lw=LW, ls='-.', alpha=AL, label=r'$D_{22}^i\nabla T_i$')
axs[5].plot(rho, out['qi_n'], color='#ff8c00', lw=LW, ls='-.', alpha=AL, label=r'$D_{21}^i\nabla n$')
axs[5].set_ylabel(r"$q_i [W\cdot m^{-2}]$")
axs[5].set_xlabel(r"$\rho_{pol}[-]$")
axs[5].set_ylim(-7e3, 9e3)
axs[5].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[5].yaxis.get_offset_text().set_fontsize(8)
axs[5].legend(loc='upper left', ncol=1)

# Force perfectly aligned Y-axis labels across all subplots
fig.align_ylabels(axs)

# Add separatrix line and ensure all ticks are inward
for i, ax in enumerate(axs):
    ax.axvline(RHO_SEPARATRIX, color='grey', linestyle='--', alpha=0.6, zorder=0)
    ax.grid(True, alpha=0.25)
    
    # Check if we are formatting the parameter panels with twin axes
    if i in [0, 1]:
        # Prevent the primary y-axis from placing tick marks on the right side
        ax.tick_params(direction='in', top=True, right=False, bottom=True, left=True, which='both')
    else:
        # Standard inward ticks on all four sides for flux panels
        ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, which='both')

# Re-apply explicit colored ticks for twinx right sides
ax0_t.tick_params(axis='y', colors='purple', direction='in', right=True, left=False, which='both')
ax1_t.tick_params(axis='y', colors='#8b4513', direction='in', right=True, left=False, which='both')

plt.savefig("two_channel_with_conv_point_by_point.png", dpi=300, bbox_inches='tight')
plt.show()