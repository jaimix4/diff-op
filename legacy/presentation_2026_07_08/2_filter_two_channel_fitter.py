import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. LOAD THE GROUND TRUTH DATA & SETTINGS
# ==========================================
data = np.loadtxt("aug36190_master_dataset_2.csv", delimiter=",", skiprows=1)

rho = data[:, 0]
n = data[:, 1]
Te = data[:, 2]        
Ti = data[:, 3]

# Store the raw data
dn_dR_raw = data[:, 4]
dTe_dR_raw = data[:, 5]
dTi_dR_raw = data[:, 6]
Gamma_raw = data[:, 10]
Qe_total_raw = data[:, 11]
Qi_total_raw = data[:, 12]

e_charge = 1.602e-19 # J/eV

NUM_POINTS = len(rho)
CHOSEN_MODEL = "sincx"  # Options: "gaussian" or "sincx"
RHO_SEPARATRIX = 1.0

# ==========================================
# 2. DEFINING THE MODEL & COST FUNCTION
# ==========================================
def compute_single_channel(D0, alpha, C_conv, n_val, T_val, dn_val, dT_val, model=CHOSEN_MODEL):
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
    D0e, ae, Ce, D0i, ai, Ci = params
    
    Ge_tot, qe_tot, *_ = compute_single_channel(D0e, ae, Ce, n_val, Te_val, dn_val, dTe_val)
    Gi_tot, qi_tot, *_ = compute_single_channel(D0i, ai, Ci, n_val, Ti_val, dn_val, dTi_val)
    
    norm_G = np.max(np.abs(G_true)) + 1e-30
    norm_qe = np.max(np.abs(qe_true)) + 1e-30
    norm_qi = np.max(np.abs(qi_true)) + 1e-30
    
    err_Ge = np.mean(((G_true - Ge_tot) / norm_G)**2)
    err_Gi = np.mean(((G_true - Gi_tot) / norm_G)**2)
    err_qe = np.mean(((qe_true - qe_tot) / norm_qe)**2)
    err_qi = np.mean(((qi_true - qi_tot) / norm_qi)**2)
    
    return (w1 * err_Ge) + (w2 * err_Gi) + (w3 * err_qe) + (w4 * err_qi)

# ==========================================
# 3. DYNAMIC OPTIMIZATION ROUTINE
# ==========================================
def optimize_dynamic(N_regions, w1, w2, w3, w4, dn_curr, dTe_curr, dTi_curr, G_curr, qe_curr, qi_curr):
    keys = ['D0e', 'ae', 'Ce', 'D0i', 'ai', 'Ci', 
            'Ge_tot', 'Ge_n', 'Ge_T', 'Ge_C', 'Gi_tot', 'Gi_n', 'Gi_T', 'Gi_C',
            'qe_tot', 'qe_n', 'qe_T', 'qi_tot', 'qi_n', 'qi_T']
    out = {k: np.zeros_like(rho) for k in keys}
    
    bnds = ((None, None), (0.0, 20.0), (0.0, 0.0), 
            (None, None), (0.0, 20.0), (0.0, 0.0)) 
    initial_guess = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0] 

    def store(mask, res_x, n_m, Te_m, Ti_m, dn_m, dTe_m, dTi_m):
        out['D0e'][mask], out['ae'][mask], out['Ce'][mask] = res_x[0:3]
        out['D0i'][mask], out['ai'][mask], out['Ci'][mask] = res_x[3:6]
        
        e_comps = compute_single_channel(res_x[0], res_x[1], res_x[2], n_m, Te_m, dn_m, dTe_m)
        out['Ge_tot'][mask], out['qe_tot'][mask], out['Ge_n'][mask], out['Ge_T'][mask], out['Ge_C'][mask], out['qe_n'][mask], out['qe_T'][mask] = e_comps
        
        i_comps = compute_single_channel(res_x[3], res_x[4], res_x[5], n_m, Ti_m, dn_m, dTi_m)
        out['Gi_tot'][mask], out['qi_tot'][mask], out['Gi_n'][mask], out['Gi_T'][mask], out['Gi_C'][mask], out['qi_n'][mask], out['qi_T'][mask] = i_comps

    if N_regions >= NUM_POINTS:
        for i in range(NUM_POINTS):
            args = (np.array([n[i]]), np.array([Te[i]]), np.array([Ti[i]]), 
                    np.array([dn_curr[i]]), np.array([dTe_curr[i]]), np.array([dTi_curr[i]]), 
                    np.array([G_curr[i]]), np.array([qe_curr[i]]), np.array([qi_curr[i]]), w1, w2, w3, w4)
            res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B', args=args)
            store(i, res.x, n[i], Te[i], Ti[i], dn_curr[i], dTe_curr[i], dTi_curr[i])
    else:
        # Determine the regional edges based on Odd/Even logic
        if N_regions % 2 != 0:
            # Odd: equally divide the whole space
            edges = np.linspace(rho.min(), rho.max(), N_regions + 1)
        else:
            # Even: divide inside and outside separatrix equally
            N_half = N_regions // 2
            edges_in = np.linspace(rho.min(), RHO_SEPARATRIX, N_half + 1)
            edges_out = np.linspace(RHO_SEPARATRIX, rho.max(), N_half + 1)
            # Concatenate, dropping the duplicate separatrix point
            edges = np.concatenate((edges_in[:-1], edges_out))

        for i in range(len(edges) - 1):
            r_min = edges[i]
            if i == len(edges) - 2:
                mask = (rho >= r_min) & (rho <= edges[i+1])
            else:
                mask = (rho >= r_min) & (rho < edges[i+1])
                
            if not np.any(mask): continue
            
            args = (n[mask], Te[mask], Ti[mask], dn_curr[mask], dTe_curr[mask], dTi_curr[mask], 
                    G_curr[mask], qe_curr[mask], qi_curr[mask], w1, w2, w3, w4)
            res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B', args=args)
            store(mask, res.x, n[mask], Te[mask], Ti[mask], dn_curr[mask], dTe_curr[mask], dTi_curr[mask])

    return out

# ==========================================
# 4. PLOT SETUP (Aesthetics & Color Theory)
# ==========================================
LW = 1.2
AL = 0.85 

C_E = '#4a00e0' 
C_En = '#00c9ff' 
C_ET = '#bd00ff' 

C_I = '#e60000' 
C_In = '#ff8c00' 
C_IT = '#ff0055' 

C_Conv = '#38b000' 
C_Data = '#2b2b2b' 

fig, axs = plt.subplots(4, 1, figsize=(11, 15), sharex=True)
plt.subplots_adjust(bottom=0.28, right=0.85)

# Start with an even number to showcase the separatrix split
init_N = 4
init_w = [1.0, 1.0, 1.0, 1.0]

init_qe_data = Qe_total_raw - (1.5 * Gamma_raw * Te * e_charge)
init_qi_data = Qi_total_raw - (1.5 * Gamma_raw * Ti * e_charge)

res = optimize_dynamic(init_N, *init_w, dn_dR_raw, dTe_dR_raw, dTi_dR_raw, Gamma_raw, init_qe_data, init_qi_data)

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

title_text = axs[0].set_title(f'Dynamic Regional: {init_N//2} Core / {init_N//2} SOL [{CHOSEN_MODEL}]')

# --- PANEL 2: Particle Fluxes ---
l_Gamma_data, = axs[1].plot(rho, Gamma_raw, color=C_Data, ls='--', lw=LW+0.5, label='Data (Ambipolar Total)')
l_Getot, = axs[1].plot(rho, res['Ge_tot'], color=C_E, lw=LW+0.5, alpha=AL, label=r'Model $\Gamma_e$')
l_Gitot, = axs[1].plot(rho, res['Gi_tot'], color=C_I, lw=LW+0.5, alpha=AL, label=r'Model $\Gamma_i$')

l_Gen, = axs[1].plot(rho, res['Ge_n'], color=C_En, lw=LW, ls=':', alpha=AL, label=r'$\Gamma_e (\nabla n)$')
l_GeT, = axs[1].plot(rho, res['Ge_T'], color=C_ET, lw=LW, ls=':', alpha=AL, label=r'$\Gamma_e (\nabla T_e)$')
l_GeC, = axs[1].plot(rho, res['Ge_C'], color=C_Conv, lw=LW, ls=':', alpha=AL, label=r'$\Gamma_e (C_e)$')
axs[1].set_ylabel(r'Particle Flux $\Gamma$')
axs[1].legend(loc='upper right', fontsize=8, ncol=2)

# --- PANEL 3: Electron Heat Flux ---
l_qe_data, = axs[2].plot(rho, init_qe_data, color=C_Data, ls='--', lw=LW+0.5, label='Data $q_e$')
l_qetot, = axs[2].plot(rho, res['qe_tot'], color=C_E, lw=LW+0.5, alpha=AL, label=r'Model $q_e$ Total')
l_qen, = axs[2].plot(rho, res['qe_n'], color=C_En, lw=LW, ls='-.', alpha=AL, label=r'$q_e (\nabla n)$ Dufour')
l_qeT, = axs[2].plot(rho, res['qe_T'], color=C_ET, lw=LW, ls='-.', alpha=AL, label=r'$q_e (\nabla T_e)$')
axs[2].set_ylabel(r'Electron Heat $q_e$')
axs[2].legend(loc='upper right', fontsize=8)

# --- PANEL 4: Ion Heat Flux ---
l_qi_data, = axs[3].plot(rho, init_qi_data, color=C_Data, ls='--', lw=LW+0.5, label='Data $q_i$')
l_qitot, = axs[3].plot(rho, res['qi_tot'], color=C_I, lw=LW+0.5, alpha=AL, label=r'Model $q_i$ Total')
l_qin, = axs[3].plot(rho, res['qi_n'], color=C_In, lw=LW, ls='-.', alpha=AL, label=r'$q_i (\nabla n)$ Dufour')
l_qiT, = axs[3].plot(rho, res['qi_T'], color=C_IT, lw=LW, ls='-.', alpha=AL, label=r'$q_i (\nabla T_i)$')
axs[3].set_ylabel(r'Ion Heat $q_i$')
axs[3].set_xlabel(r'$\rho_{pol}$')
axs[3].legend(loc='upper right', fontsize=8)

for ax in axs:
    ax.axvline(RHO_SEPARATRIX, color='k', linestyle=':', alpha=0.4)
    ax.grid(True, alpha=0.25)

# ==========================================
# 5. UI SLIDERS
# ==========================================
bg_col = 'lightgoldenrodyellow'
ax_smooth = plt.axes([0.15, 0.20, 0.65, 0.02], facecolor=bg_col)
ax_N  = plt.axes([0.15, 0.17, 0.65, 0.02], facecolor=bg_col)
ax_w1 = plt.axes([0.15, 0.14, 0.65, 0.02], facecolor=bg_col)
ax_w2 = plt.axes([0.15, 0.11, 0.65, 0.02], facecolor=bg_col)
ax_w3 = plt.axes([0.15, 0.08, 0.65, 0.02], facecolor=bg_col)
ax_w4 = plt.axes([0.15, 0.05, 0.65, 0.02], facecolor=bg_col)

s_smooth = Slider(ax_smooth, 'Smoothing ($\sigma$)', 0.0, 5.0, valinit=0.0)
s_N  = Slider(ax_N, 'Regions (N)', 1, NUM_POINTS, valinit=init_N, valstep=1)
s_w1 = Slider(ax_w1, r'$W_1 (\Gamma_e)$', 0, 10, valinit=1.0)
s_w2 = Slider(ax_w2, r'$W_2 (\Gamma_i)$', 0, 10, valinit=1.0)
s_w3 = Slider(ax_w3, r'$W_3 (q_e)$', 0, 10, valinit=1.0)
s_w4 = Slider(ax_w4, r'$W_4 (q_i)$', 0, 10, valinit=1.0)

def update(val):
    N_reg = int(s_N.val)
    sigma = s_smooth.val
    w = [s_w1.val, s_w2.val, s_w3.val, s_w4.val]
    
    if sigma > 0.1:
        dn_curr = gaussian_filter1d(dn_dR_raw, sigma)
        dTe_curr = gaussian_filter1d(dTe_dR_raw, sigma)
        dTi_curr = gaussian_filter1d(dTi_dR_raw, sigma)
        Gamma_curr = gaussian_filter1d(Gamma_raw, sigma)
        Qe_curr = gaussian_filter1d(Qe_total_raw, sigma)
        Qi_curr = gaussian_filter1d(Qi_total_raw, sigma)
    else:
        dn_curr, dTe_curr, dTi_curr = dn_dR_raw, dTe_dR_raw, dTi_dR_raw
        Gamma_curr, Qe_curr, Qi_curr = Gamma_raw, Qe_total_raw, Qi_total_raw
        
    qe_curr = Qe_curr - (1.5 * Gamma_curr * Te * e_charge)
    qi_curr = Qi_curr - (1.5 * Gamma_curr * Ti * e_charge)
    
    l_Gamma_data.set_ydata(Gamma_curr)
    l_qe_data.set_ydata(qe_curr)
    l_qi_data.set_ydata(qi_curr)
    
    r = optimize_dynamic(N_reg, *w, dn_curr, dTe_curr, dTi_curr, Gamma_curr, qe_curr, qi_curr)
    
    l_D0e.set_ydata(r['D0e']); l_D0i.set_ydata(r['D0i'])
    l_ae.set_ydata(r['ae']);   l_ai.set_ydata(r['ai'])
    l_Ce.set_ydata(r['Ce']);   l_Ci.set_ydata(r['Ci'])
    
    l_Getot.set_ydata(r['Ge_tot']); l_Gitot.set_ydata(r['Gi_tot'])
    l_Gen.set_ydata(r['Ge_n']);     l_GeT.set_ydata(r['Ge_T']); l_GeC.set_ydata(r['Ge_C'])
    
    l_qetot.set_ydata(r['qe_tot']); l_qen.set_ydata(r['qe_n']); l_qeT.set_ydata(r['qe_T'])
    l_qitot.set_ydata(r['qi_tot']); l_qin.set_ydata(r['qi_n']); l_qiT.set_ydata(r['qi_T'])
    
    lines = [l_D0e, l_D0i, l_ae, l_ai, l_Ce, l_Ci]
    if N_reg >= NUM_POINTS:
        title_text.set_text(f'Point-by-Point Fitting [{CHOSEN_MODEL}]')
        for line in lines: line.set_drawstyle('default')
    else:
        if N_reg % 2 == 0:
            title_text.set_text(f'Dynamic Regional: {N_reg//2} Core / {N_reg//2} SOL [{CHOSEN_MODEL}]')
        else:
            title_text.set_text(f'Dynamic Regional: {N_reg} Equal Regions [{CHOSEN_MODEL}]')
            
        for line in lines: line.set_drawstyle('steps-post')
    
    for ax in [axs[0], ax0_t1, ax0_t2, axs[1], axs[2], axs[3]]:
        ax.relim()
        ax.autoscale_view()
        
    fig.canvas.draw_idle()

for s in [s_smooth, s_N, s_w1, s_w2, s_w3, s_w4]:
    s.on_changed(update)

plt.show()