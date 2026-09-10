import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. LOAD THE GROUND TRUTH DATA
# ==========================================
data = np.loadtxt("aug36190_master_dataset_2.csv", delimiter=",", skiprows=1)

rho = data[:, 0]
n = data[:, 1]
T = data[:, 2]        
dn_dR_raw = data[:, 4]
dT_dR_raw = data[:, 5]
Gamma_raw = data[:, 10]
Qe_total_raw = data[:, 11]

e_charge = 1.602e-19 # J/eV

smooth_sigma = 3.0

# Apply the convolution filter
dn_dR = gaussian_filter1d(dn_dR_raw, sigma=smooth_sigma)
dT_dR = gaussian_filter1d(dT_dR_raw, sigma=smooth_sigma)
Gamma_data = gaussian_filter1d(Gamma_raw, sigma=smooth_sigma)
Qe_total_data = gaussian_filter1d(Qe_total_raw, sigma=smooth_sigma)

# Compute pure conductive heat flux from the data
q_data = Qe_total_data - (1.5 * Gamma_data * T * e_charge)

NUM_POINTS = len(rho)

# Toggle analytical models: "gaussian" or "sincx"
CHOSEN_MODEL = "sincx" # or "gaussian"

# ==========================================
# 2. DEFINING THE MODEL & COST FUNCTION
# ==========================================
def compute_model(D0, alpha, C_conv, n_val, T_val, dn_val, dT_val, model=CHOSEN_MODEL):
    """Calculates Gamma, qe, and all sub-components including convection."""
    if model == "gaussian":
        A11 = D0 / (1 + alpha)**1.5
        A12 = - (3 * alpha * D0 * n_val) / (2 * (1 + alpha)**2.5 * T_val)
        A21 = - (3 * alpha * D0 * T_val) / (2 * (1 + alpha)**2.5) 
        A22 = (3 * (2 + 3 * alpha**2) * D0 * n_val) / (4 * (1 + alpha)**3.5)

    elif model == "sincx":
        A11 = D0 * np.exp((-alpha**2) / 4)
        A12 = - D0 * np.exp((-alpha**2) / 4) * ((n_val*alpha**2)/(4*T_val))
        A21 = - D0 * np.exp((-alpha**2) / 4) * ((T_val*alpha**2)/(4))
        A22 = D0 * np.exp((-alpha**2) / 4) * (n_val*(24 - 8*alpha**2 + alpha**4)/16)
        
    # Sub-components of Particle Flux
    Gamma_n = -1 * A11 * dn_val
    Gamma_T = -1 * A12 * dT_val
    Gamma_C = C_conv * n_val
    Gamma_mod = Gamma_n + Gamma_T + Gamma_C
    
    # Sub-components of Conductive Heat Flux
    q_n = -1 * (A21 * dn_val) * e_charge
    q_T = -1 * (A22 * dT_val) * e_charge
    q_mod = q_n + q_T
    
    A12 = A12 * (T_val / n_val)  # Convert to thermodiffusion coefficient for better interpretability
    A21 = A21 * (1 / T_val)  # Convert to Dufour coefficient for better interpretability
    A22 = A22 * (1 / n_val)  # Convert to thermal conductivity for better interpretability

    return Gamma_mod, q_mod, Gamma_n, Gamma_T, Gamma_C, q_n, q_T, A11, A12, A21, A22

def objective_function(params, n_val, T_val, dn_val, dT_val, Gamma_true, q_true):
    # Now unpacking 3 parameters
    D0, alpha, C_conv = params
    
    res = compute_model(D0, alpha, C_conv, n_val, T_val, dn_val, dT_val)
    Gamma_mod, q_mod = res[0], res[1]
    
    norm_Gamma = np.max(np.abs(Gamma_true)) + 1e-30
    norm_q = np.max(np.abs(q_true)) + 1e-30
    
    err_Gamma = np.mean(((Gamma_true - Gamma_mod) / norm_Gamma)**2)
    err_q = np.mean(((q_true - q_mod) / norm_q)**2)
    
    return err_Gamma + err_q

# ==========================================
# 3. DYNAMIC OPTIMIZATION ROUTINE
# ==========================================
def optimize_dynamic(N_regions):
    """Handles regional splitting and point-by-point fitting."""
    out = {k: np.zeros_like(rho) for k in 
           ['D0', 'alpha', 'C_conv', 'G_tot', 'q_tot', 'G_n', 'G_T', 'G_C', 'q_n', 'q_T', 'A11', 'A12', 'A21', 'A22']}
    
    # Bounds: D0 (unbounded), alpha (0 to 20), C_conv (unbounded)
    bnds = ((None, None), (0.0, 20.0), (None, None)) 
    initial_guess = [1.0, 1.0, 0.0] 

    def store_results(mask, res_x, n_m, T_m, dn_m, dT_m):
        out['D0'][mask] = res_x[0]
        out['alpha'][mask] = res_x[1]
        out['C_conv'][mask] = res_x[2]
        
        comps = compute_model(res_x[0], res_x[1], res_x[2], n_m, T_m, dn_m, dT_m)
        out['G_tot'][mask], out['q_tot'][mask] = comps[0], comps[1]
        out['G_n'][mask], out['G_T'][mask] = comps[2], comps[3]
        out['G_C'][mask] = comps[4]
        out['q_n'][mask], out['q_T'][mask] = comps[5], comps[6]
        out['A11'][mask], out['A12'][mask] = comps[7], comps[8]
        out['A21'][mask], out['A22'][mask] = comps[9], comps[10]

    if N_regions >= NUM_POINTS:
        for i in range(NUM_POINTS):
            res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B',
                           args=(np.array([n[i]]), np.array([T[i]]), np.array([dn_dR[i]]), np.array([dT_dR[i]]), 
                                 np.array([Gamma_data[i]]), np.array([q_data[i]])))
            store_results(i, res.x, n[i], T[i], dn_dR[i], dT_dR[i])
            
    else:
        edges = np.linspace(rho.min(), rho.max(), int(N_regions) + 1)
        for i in range(len(edges) - 1):
            r_min = edges[i]
            if i == len(edges) - 2:
                mask = (rho >= r_min) & (rho <= edges[i+1])
            else:
                mask = (rho >= r_min) & (rho < edges[i+1])
                
            if not np.any(mask): continue
                
            res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B',
                           args=(n[mask], T[mask], dn_dR[mask], dT_dR[mask], Gamma_data[mask], q_data[mask]))
            store_results(mask, res.x, n[mask], T[mask], dn_dR[mask], dT_dR[mask])

    return out

# ==========================================
# 4. INTERACTIVE PLOT SETUP
# ==========================================
fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
plt.subplots_adjust(bottom=0.08, right=0.85) # Leave room for 3rd y-axis and slider

init_N = 5
results = optimize_dynamic(init_N)

# --- Panel 1: D0, alpha, and C_conv ---
l_D0, = axs[0].step(rho, results['D0'], 'b-', lw=2, where='post', label=r'$D_0$')
axs[0].set_ylabel(r'$D_0$', color='b')
axs[0].tick_params(axis='y', labelcolor='b')
axs[0].set_ylim(-2.0, 10.0)

# Second Y-axis for alpha
ax0_twin1 = axs[0].twinx()
l_alpha, = ax0_twin1.step(rho, results['alpha'], 'g-', lw=2, where='post', label=r'$\alpha$')
ax0_twin1.set_ylabel(r'$\alpha$', color='g')
ax0_twin1.tick_params(axis='y', labelcolor='g')
ax0_twin1.set_ylim(-0.2, 7.5)

# Third Y-axis for C_conv
ax0_twin2 = axs[0].twinx()
ax0_twin2.spines["right"].set_position(("axes", 1.1)) # Offset the 3rd axis
l_C, = ax0_twin2.step(rho, results['C_conv'], 'purple', lw=2, where='post', label=r'$C$ (Convection)')
ax0_twin2.set_ylabel(r'Convective Vel $C$ [m/s]', color='purple')
ax0_twin2.tick_params(axis='y', labelcolor='purple')
ax0_twin2.set_ylim(-20.0, 50.0)

title_text = axs[0].set_title(f'Dynamic Regional Fitting [{CHOSEN_MODEL} model] (N={init_N})')

# --- Panel 2: Particle Flux & Components ---
axs[1].plot(rho, Gamma_data, 'k--', lw=2, label='Data (Total)')
l_G_tot, = axs[1].plot(rho, results['G_tot'], 'purple', lw=2.5, alpha=0.9, label='Model Total')
l_G_n, = axs[1].plot(rho, results['G_n'], 'tab:blue', lw=1.5, linestyle='-.', label=r'$-A_{11}\nabla n$')
l_G_T, = axs[1].plot(rho, results['G_T'], 'tab:orange', lw=1.5, linestyle='-.', label=r'$-A_{12}\nabla T$')
l_G_C, = axs[1].plot(rho, results['G_C'], 'magenta', lw=1.5, linestyle=':', label=r'$C n$ (Convective)')
axs[1].set_ylabel(r'Particle Flux $\Gamma_e$')
axs[1].legend(loc='upper right', fontsize=8)

# --- Panel 3: Conductive Heat Flux & Components ---
axs[2].plot(rho, q_data, 'k--', lw=2, label='Data (Total)')
l_q_tot, = axs[2].plot(rho, results['q_tot'], 'red', lw=2.5, alpha=0.9, label='Model Total')
l_q_T, = axs[2].plot(rho, results['q_T'], 'tab:orange', lw=1.5, linestyle='-.', label=r'$-A_{22}\nabla T$')
l_q_n, = axs[2].plot(rho, results['q_n'], 'tab:blue', lw=1.5, linestyle='-.', label=r'$-A_{21}\nabla n$')
axs[2].set_ylabel(r'Conductive Heat Flux $q_e$')
axs[2].legend(loc='upper right', fontsize=8)

# --- Panel 4: Matrix Elements ---
l_A11, = axs[3].step(rho, results['A11'], color='tab:blue', lw=2, where='post', label=r'$A_{11}$')
l_A12, = axs[3].step(rho, results['A12'], color='tab:cyan', lw=2, where='post', label=r'$A_{12}$')
axs[3].set_ylabel(r'$A_{11}, A_{12}$ Elements', color='tab:blue')
axs[3].tick_params(axis='y', labelcolor='tab:blue')

ax3_twin = axs[3].twinx()
l_A21, = ax3_twin.step(rho, results['A21'], color='tab:red', lw=2, where='post', label=r'$A_{21}$')
l_A22, = ax3_twin.step(rho, results['A22'], color='tab:orange', lw=2, where='post', label=r'$A_{22}$')
ax3_twin.set_ylabel(r'$A_{21}, A_{22}$ Elements', color='tab:red')
ax3_twin.tick_params(axis='y', labelcolor='tab:red')
axs[3].set_xlabel(r'$\rho_{pol}$')

# Combine legends for Panel 4
lines4, labels4 = axs[3].get_legend_handles_labels()
lines4_t, labels4_t = ax3_twin.get_legend_handles_labels()
ax3_twin.legend(lines4 + lines4_t, labels4 + labels4_t, loc='upper left', fontsize=8)

for ax in axs:
    ax.axvline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

# ==========================================
# 5. THE SLIDER
# ==========================================
axcolor = 'lightgoldenrodyellow'
ax_N = plt.axes([0.15, 0.02, 0.65, 0.02], facecolor=axcolor)
s_N = Slider(ax_N, 'Regions (N)', 1, NUM_POINTS, valinit=init_N, valstep=1)

def update(val):
    N_regions = int(s_N.val)
    res = optimize_dynamic(N_regions)
    
    # Update Data
    l_D0.set_ydata(res['D0'])
    l_alpha.set_ydata(res['alpha'])
    l_C.set_ydata(res['C_conv'])
    
    l_G_tot.set_ydata(res['G_tot'])
    l_G_n.set_ydata(res['G_n'])
    l_G_T.set_ydata(res['G_T'])
    l_G_C.set_ydata(res['G_C'])
    
    l_q_tot.set_ydata(res['q_tot'])
    l_q_n.set_ydata(res['q_n'])
    l_q_T.set_ydata(res['q_T'])
    
    l_A11.set_ydata(res['A11'])
    l_A12.set_ydata(res['A12'])
    l_A21.set_ydata(res['A21'])
    l_A22.set_ydata(res['A22'])
    
    # Plot Style Switcher
    lines_to_switch = [l_D0, l_alpha, l_C, l_A11, l_A12, l_A21, l_A22]
    
    if N_regions >= 50: #NUM_POINTS:
        title_text.set_text(f'Exact Point-by-Point Fitting Active [{CHOSEN_MODEL} model]')
        for line in lines_to_switch:
            line.set_drawstyle('default')
    else:
        title_text.set_text(f'Dynamic Regional Fitting [{CHOSEN_MODEL} model] (N={N_regions})')
        for line in lines_to_switch:
            line.set_drawstyle('steps-post')
    
    # Auto-scale axes
    for ax in [axs[0], ax0_twin1, ax0_twin2, axs[1], axs[2], axs[3], ax3_twin]:
        ax.relim()
        ax.autoscale_view()
    
    fig.canvas.draw_idle()

s_N.on_changed(update)

plt.show()