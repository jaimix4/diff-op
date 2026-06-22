import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize

# ==========================================
# 1. LOAD THE GROUND TRUTH DATA & SETTINGS
# ==========================================
data = np.loadtxt("aug36190_master_dataset.csv", delimiter=",", skiprows=1)

rho = data[:, 0]
n = data[:, 1]
Te = data[:, 2]        
Ti = data[:, 3]
dn_dR = data[:, 4]
dTe_dR = data[:, 5]
dTi_dR = data[:, 6]
Gamma_data = data[:, 10]
Qe_total = data[:, 11]
Qi_total = data[:, 12]

e_charge = 1.602e-19 # J/eV

# Compute pure conductive heat fluxes: q = Q - 1.5 * Gamma * T * e
qe_data = Qe_total - (1.5 * Gamma_data * Te * e_charge)
qi_data = Qi_total - (1.5 * Gamma_data * Ti * e_charge)

NUM_POINTS = len(rho)
CHOSEN_MODEL = "sincx" #"gaussian"  # Options: "gaussian" or "sincx"

# ==========================================
# 2. DEFINING THE MODEL & COST FUNCTION
# ==========================================
def compute_single_channel(D0, alpha, C_conv, n_val, T_val, dn_val, dT_val, model=CHOSEN_MODEL):
    """Calculates Gamma, q, and components for a single species channel."""
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
        
    G_n = -1 * A11 * dn_val
    G_T = -1 * A12 * dT_val
    G_C = C_conv * n_val
    G_tot = G_n + G_T + G_C
    
    q_n = -1 * (A21 * dn_val) * e_charge
    q_T = -1 * (A22 * dT_val) * e_charge
    q_tot = q_n + q_T
    
    return G_tot, q_tot, G_n, G_T, G_C, q_n, q_T

def objective_function(params, n_val, Te_val, Ti_val, dn_val, dTe_val, dTi_val, 
                       G_true, qe_true, qi_true, w1, w2, w3, w4):
    # Unpack 6 parameters (3 for electrons, 3 for ions)
    D0e, ae, Ce, D0i, ai, Ci = params
    
    # Compute both channels
    Ge_tot, qe_tot, *_ = compute_single_channel(D0e, ae, Ce, n_val, Te_val, dn_val, dTe_val)
    Gi_tot, qi_tot, *_ = compute_single_channel(D0i, ai, Ci, n_val, Ti_val, dn_val, dTi_val)
    
    # Normalize
    norm_G = np.max(np.abs(G_true)) + 1e-30
    norm_qe = np.max(np.abs(qe_true)) + 1e-30
    norm_qi = np.max(np.abs(qi_true)) + 1e-30
    
    # Ambipolarity & Heat Penalties
    err_Ge = np.mean(((G_true - Ge_tot) / norm_G)**2)
    err_Gi = np.mean(((G_true - Gi_tot) / norm_G)**2)
    err_qe = np.mean(((qe_true - qe_tot) / norm_qe)**2)
    err_qi = np.mean(((qi_true - qi_tot) / norm_qi)**2)
    
    return (w1 * err_Ge) + (w2 * err_Gi) + (w3 * err_qe) + (w4 * err_qi)

# ==========================================
# 3. DYNAMIC OPTIMIZATION ROUTINE
# ==========================================
def optimize_dynamic(N_regions, w1, w2, w3, w4):
    keys = ['D0e', 'ae', 'Ce', 'D0i', 'ai', 'Ci', 
            'Ge_tot', 'Ge_n', 'Ge_T', 'Ge_C', 'Gi_tot', 'Gi_n', 'Gi_T', 'Gi_C',
            'qe_tot', 'qe_n', 'qe_T', 'qi_tot', 'qi_n', 'qi_T']
    out = {k: np.zeros_like(rho) for k in keys}
    
    # Bounds: D0 (None), alpha (0 to 20), C_conv (None)
    bnds = ((None, None), (0.0, 20.0), (None, None), 
            (None, None), (0.0, 20.0), (None, None)) 
    initial_guess = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0] 

    def store(mask, res_x, n_m, Te_m, Ti_m, dn_m, dTe_m, dTi_m):
        out['D0e'][mask], out['ae'][mask], out['Ce'][mask] = res_x[0:3]
        out['D0i'][mask], out['ai'][mask], out['Ci'][mask] = res_x[3:6]
        
        e_comps = compute_single_channel(res_x[0], res_x[1], res_x[2], n_m, Te_m, dn_m, dTe_m)
        out['Ge_tot'][mask], out['qe_tot'][mask], out['Ge_n'][mask], out['Ge_T'][mask], out['Ge_C'][mask], out['qe_n'][mask], out['qe_T'][mask] = e_comps
        
        i_comps = compute_single_channel(res_x[3], res_x[4], res_x[5], n_m, Ti_m, dn_m, dTi_m)
        out['Gi_tot'][mask], out['qi_tot'][mask], out['Gi_n'][mask], out['Gi_T'][mask], out['Gi_C'][mask], out['qi_n'][mask], out['qi_T'][mask] = i_comps

    if N_regions >= NUM_POINTS:
        for i in range(NUM_POINTS):
            args = (np.array([n[i]]), np.array([Te[i]]), np.array([Ti[i]]), np.array([dn_dR[i]]), np.array([dTe_dR[i]]), np.array([dTi_dR[i]]), 
                    np.array([Gamma_data[i]]), np.array([qe_data[i]]), np.array([qi_data[i]]), w1, w2, w3, w4)
            res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B', args=args)
            store(i, res.x, n[i], Te[i], Ti[i], dn_dR[i], dTe_dR[i], dTi_dR[i])
    else:
        edges = np.linspace(rho.min(), rho.max(), int(N_regions) + 1)
        for i in range(len(edges) - 1):
            r_min = edges[i]
            mask = (rho >= r_min) & (rho <= edges[i+1]) if i == len(edges) - 2 else (rho >= r_min) & (rho < edges[i+1])
            if not np.any(mask): continue
            
            args = (n[mask], Te[mask], Ti[mask], dn_dR[mask], dTe_dR[mask], dTi_dR[mask], 
                    Gamma_data[mask], qe_data[mask], qi_data[mask], w1, w2, w3, w4)
            res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B', args=args)
            store(mask, res.x, n[mask], Te[mask], Ti[mask], dn_dR[mask], dTe_dR[mask], dTi_dR[mask])

    return out

# ==========================================
# 4. PLOT SETUP (Aesthetics & Color Theory)
# ==========================================
# Thin lines and transparency for elegance
LW = 1.2
AL = 0.85 

# Color Palettes
C_E = '#4a00e0' # Deep Blue-Purple (Electrons Total)
C_En = '#00c9ff' # Cyan (Electrons dn)
C_ET = '#bd00ff' # Pink/Magenta (Electrons dT)

C_I = '#e60000' # Deep Red (Ions Total)
C_In = '#ff8c00' # Orange (Ions dn)
C_IT = '#ff0055' # Bright Red-Pink (Ions dT)

C_Conv = '#38b000' # Earthy Green (Convection)
C_Data = '#2b2b2b' # Deep Grey (Ground Truth)

fig, axs = plt.subplots(4, 1, figsize=(11, 15), sharex=True)
plt.subplots_adjust(bottom=0.22, right=0.85) 

init_N = 5
init_w = [1.0, 1.0, 1.0, 1.0]
res = optimize_dynamic(init_N, *init_w)

# --- PANEL 1: Parameters ---
l_D0e, = axs[0].step(rho, res['D0e'], color=C_E, lw=LW+0.5, where='post', label=r'$D_{0,e}$')
l_D0i, = axs[0].step(rho, res['D0i'], color=C_I, lw=LW+0.5, where='post', linestyle='--', label=r'$D_{0,i}$')
axs[0].set_ylabel(r'$D_0$', color='k')
axs[0].legend(loc='upper left', fontsize=8)

ax0_t1 = axs[0].twinx()
l_ae, = ax0_t1.step(rho, res['ae'], color=C_ET, lw=LW, where='post', label=r'$\alpha_e$')
l_ai, = ax0_t1.step(rho, res['ai'], color=C_IT, lw=LW, where='post', linestyle='--', label=r'$\alpha_i$')
ax0_t1.set_ylabel(r'$\alpha$', color='k')
ax0_t1.legend(loc='center left', fontsize=8)

ax0_t2 = axs[0].twinx()
ax0_t2.spines["right"].set_position(("axes", 1.1)) 
l_Ce, = ax0_t2.step(rho, res['Ce'], color=C_Conv, lw=LW, where='post', label=r'$C_e$')
l_Ci, = ax0_t2.step(rho, res['Ci'], color=C_Conv, lw=LW, where='post', linestyle='--', label=r'$C_i$')
ax0_t2.set_ylabel(r'Convection $C$', color=C_Conv)
ax0_t2.legend(loc='lower left', fontsize=8)

title_text = axs[0].set_title(f'Two-Channel Fitting [{CHOSEN_MODEL}] (N={init_N})')

# --- PANEL 2: Particle Fluxes ---
axs[1].plot(rho, Gamma_data, color=C_Data, ls='--', lw=LW+0.5, label='Data (Ambipolar Total)')
l_Getot, = axs[1].plot(rho, res['Ge_tot'], color=C_E, lw=LW+0.5, alpha=AL, label=r'Model $\Gamma_e$')
l_Gitot, = axs[1].plot(rho, res['Gi_tot'], color=C_I, lw=LW+0.5, alpha=AL, label=r'Model $\Gamma_i$')

l_Gen, = axs[1].plot(rho, res['Ge_n'], color=C_En, lw=LW, ls=':', alpha=AL, label=r'$\Gamma_e (\nabla n)$')
l_GeT, = axs[1].plot(rho, res['Ge_T'], color=C_ET, lw=LW, ls=':', alpha=AL, label=r'$\Gamma_e (\nabla T_e)$')
l_GeC, = axs[1].plot(rho, res['Ge_C'], color=C_Conv, lw=LW, ls=':', alpha=AL, label=r'$\Gamma_e (C_e)$')
axs[1].set_ylabel(r'Particle Flux $\Gamma$')
axs[1].legend(loc='upper right', fontsize=8, ncol=2)

# --- PANEL 3: Electron Heat Flux ---
axs[2].plot(rho, qe_data, color=C_Data, ls='--', lw=LW+0.5, label='Data $q_e$')
l_qetot, = axs[2].plot(rho, res['qe_tot'], color=C_E, lw=LW+0.5, alpha=AL, label=r'Model $q_e$ Total')
l_qen, = axs[2].plot(rho, res['qe_n'], color=C_En, lw=LW, ls='-.', alpha=AL, label=r'$q_e (\nabla n)$ Dufour')
l_qeT, = axs[2].plot(rho, res['qe_T'], color=C_ET, lw=LW, ls='-.', alpha=AL, label=r'$q_e (\nabla T_e)$')
axs[2].set_ylabel(r'Electron Heat $q_e$')
axs[2].legend(loc='upper right', fontsize=8)

# --- PANEL 4: Ion Heat Flux ---
axs[3].plot(rho, qi_data, color=C_Data, ls='--', lw=LW+0.5, label='Data $q_i$')
l_qitot, = axs[3].plot(rho, res['qi_tot'], color=C_I, lw=LW+0.5, alpha=AL, label=r'Model $q_i$ Total')
l_qin, = axs[3].plot(rho, res['qi_n'], color=C_In, lw=LW, ls='-.', alpha=AL, label=r'$q_i (\nabla n)$ Dufour')
l_qiT, = axs[3].plot(rho, res['qi_T'], color=C_IT, lw=LW, ls='-.', alpha=AL, label=r'$q_i (\nabla T_i)$')
axs[3].set_ylabel(r'Ion Heat $q_i$')
axs[3].set_xlabel(r'$\rho_{pol}$')
axs[3].legend(loc='upper right', fontsize=8)

for ax in axs:
    ax.axvline(1.0, color='k', linestyle=':', alpha=0.4)
    ax.grid(True, alpha=0.25)

# ==========================================
# 5. UI SLIDERS (Regions & Weights)
# ==========================================
bg_col = 'lightgoldenrodyellow'
ax_N  = plt.axes([0.15, 0.14, 0.65, 0.02], facecolor=bg_col)
ax_w1 = plt.axes([0.15, 0.11, 0.65, 0.02], facecolor=bg_col)
ax_w2 = plt.axes([0.15, 0.08, 0.65, 0.02], facecolor=bg_col)
ax_w3 = plt.axes([0.15, 0.05, 0.65, 0.02], facecolor=bg_col)
ax_w4 = plt.axes([0.15, 0.02, 0.65, 0.02], facecolor=bg_col)

s_N  = Slider(ax_N, 'Regions (N)', 1, NUM_POINTS, valinit=init_N, valstep=1)
s_w1 = Slider(ax_w1, r'$W_1 (\Gamma_e)$', 0, 10, valinit=1.0)
s_w2 = Slider(ax_w2, r'$W_2 (\Gamma_i)$', 0, 10, valinit=1.0)
s_w3 = Slider(ax_w3, r'$W_3 (q_e)$', 0, 10, valinit=1.0)
s_w4 = Slider(ax_w4, r'$W_4 (q_i)$', 0, 10, valinit=1.0)

def update(val):
    N_reg = int(s_N.val)
    w = [s_w1.val, s_w2.val, s_w3.val, s_w4.val]
    
    r = optimize_dynamic(N_reg, *w)
    
    # Update Parameter Plots
    l_D0e.set_ydata(r['D0e']); l_D0i.set_ydata(r['D0i'])
    l_ae.set_ydata(r['ae']);   l_ai.set_ydata(r['ai'])
    l_Ce.set_ydata(r['Ce']);   l_Ci.set_ydata(r['Ci'])
    
    # Update Particle Fluxes
    l_Getot.set_ydata(r['Ge_tot']); l_Gitot.set_ydata(r['Gi_tot'])
    l_Gen.set_ydata(r['Ge_n']);     l_GeT.set_ydata(r['Ge_T']); l_GeC.set_ydata(r['Ge_C'])
    # Update Heat Fluxes
    l_qetot.set_ydata(r['qe_tot']); l_qen.set_ydata(r['qe_n']); l_qeT.set_ydata(r['qe_T'])
    l_qitot.set_ydata(r['qi_tot']); l_qin.set_ydata(r['qi_n']); l_qiT.set_ydata(r['qi_T'])
    
    # Toggle Drawstyle
    lines = [l_D0e, l_D0i, l_ae, l_ai, l_Ce, l_Ci]
    if N_reg >= NUM_POINTS:
        title_text.set_text(f'Point-by-Point Fitting [{CHOSEN_MODEL}]')
        for line in lines: line.set_drawstyle('default')
    else:
        title_text.set_text(f'Dynamic Regional Fitting [{CHOSEN_MODEL}] (N={N_reg})')
        for line in lines: line.set_drawstyle('steps-post')
    
    # Autoscale
    for ax in [axs[0], ax0_t1, ax0_t2, axs[1], axs[2], axs[3]]:
        ax.relim()
        ax.autoscale_view()
        
    fig.canvas.draw_idle()

for s in [s_N, s_w1, s_w2, s_w3, s_w4]:
    s.on_changed(update)

plt.show()