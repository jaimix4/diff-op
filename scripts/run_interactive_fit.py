"""Interactive two-channel diffusion-convection fit against AUG #36190 (GRILLIX) data.

Same slider UI as legacy/presentation_2026_07_08/5_two_channel_fit.py, but the model
itself now comes from src/diffop (operators.py + transport.py + regions.py +
fitting.py) instead of being embedded in this script. Switch CHOSEN_MODEL to any
name registered in diffop.operators (currently "gaussian" or "sincx").
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, CheckButtons
from scipy.ndimage import gaussian_filter1d

from diffop.fitting import fit_two_channel

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "processed" / "aug36190_master_dataset_2.csv"

CHOSEN_MODEL = "gaussian"  # any name in diffop.operators.available()
RHO_SEPARATRIX = 1.0
E_CHARGE = 1.602e-19  # J/eV

# ==========================================
# 1. LOAD THE GROUND TRUTH DATA
# ==========================================
data = np.loadtxt(DATA_CSV, delimiter=",", skiprows=1)

rho = data[:, 0]
n = data[:, 1]
Te = data[:, 2]
Ti = data[:, 3]

dn_dR_raw = data[:, 4]
dTe_dR_raw = data[:, 5]
dTi_dR_raw = data[:, 6]
Gamma_raw = data[:, 10]
Qe_total_raw = data[:, 11]
Qi_total_raw = data[:, 12]

NUM_POINTS = len(rho)


def run_fit(N_regions, w1, w2, w3, w4, dn, dTe, dTi, Gamma, qe, qi, rho_c_min, rho_c_max, use_conv):
    result = fit_two_channel(
        rho, n, Te, Ti, dn, dTe, dTi, Gamma, qe, qi,
        operator=CHOSEN_MODEL,
        n_regions=N_regions,
        separatrix=RHO_SEPARATRIX,
        weights=(w1, w2, w3, w4),
        use_convection=use_conv,
        conv_range=(rho_c_min, rho_c_max),
    )
    return {
        "D0e": result.electron.A, "ae": result.electron.alpha, "Ce": result.electron.V,
        "D0i": result.ion.A, "ai": result.ion.alpha, "Ci": result.ion.V,
        "Ge_tot": result.electron.Gamma, "Ge_n": result.electron.Gamma_n,
        "Ge_T": result.electron.Gamma_T, "Ge_C": result.electron.Gamma_C,
        "Gi_tot": result.ion.Gamma, "Gi_n": result.ion.Gamma_n,
        "Gi_T": result.ion.Gamma_T, "Gi_C": result.ion.Gamma_C,
        "qe_tot": result.electron.q, "qe_n": result.electron.q_n, "qe_T": result.electron.q_T,
        "qi_tot": result.ion.q, "qi_n": result.ion.q_n, "qi_T": result.ion.q_T,
    }


# ==========================================
# 2. PLOT SETUP (Aesthetics & Color Theory)
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
plt.subplots_adjust(bottom=0.32, right=0.85)

init_N = 4
init_w = [1.0, 1.0, 1.0, 1.0]
init_rho_c = (rho.min(), rho.max())
init_use_conv = True

init_qe_data = Qe_total_raw - (1.5 * Gamma_raw * Te * E_CHARGE)
init_qi_data = Qi_total_raw - (1.5 * Gamma_raw * Ti * E_CHARGE)

res = run_fit(init_N, *init_w, dn_dR_raw, dTe_dR_raw, dTi_dR_raw, Gamma_raw, init_qe_data, init_qi_data, *init_rho_c, init_use_conv)

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
# 3. UI SLIDERS
# ==========================================
bg_col = 'lightgoldenrodyellow'
ax_smooth = plt.axes([0.15, 0.24, 0.65, 0.02], facecolor=bg_col)
ax_N      = plt.axes([0.15, 0.21, 0.65, 0.02], facecolor=bg_col)
ax_rho_c  = plt.axes([0.15, 0.18, 0.50, 0.02], facecolor=bg_col)
ax_conv_t = plt.axes([0.68, 0.175, 0.12, 0.03], frameon=False)
ax_w1     = plt.axes([0.15, 0.14, 0.65, 0.02], facecolor=bg_col)
ax_w2     = plt.axes([0.15, 0.11, 0.65, 0.02], facecolor=bg_col)
ax_w3     = plt.axes([0.15, 0.08, 0.65, 0.02], facecolor=bg_col)
ax_w4     = plt.axes([0.15, 0.05, 0.65, 0.02], facecolor=bg_col)

s_smooth = Slider(ax_smooth, 'Smoothing ($\\sigma$)', 0.0, 5.0, valinit=0.0)
s_N      = Slider(ax_N, 'Regions (N)', 1, NUM_POINTS, valinit=init_N, valstep=1)
s_rho_c  = RangeSlider(ax_rho_c, 'Conv. Range ($\\rho$)', rho.min(), rho.max(), valinit=init_rho_c)
cb_conv  = CheckButtons(ax_conv_t, ['Convection On'], [init_use_conv])
s_w1     = Slider(ax_w1, r'$W_1 (\Gamma_e)$', 0, 10, valinit=1.0)
s_w2     = Slider(ax_w2, r'$W_2 (\Gamma_i)$', 0, 10, valinit=1.0)
s_w3     = Slider(ax_w3, r'$W_3 (q_e)$', 0, 10, valinit=1.0)
s_w4     = Slider(ax_w4, r'$W_4 (q_i)$', 0, 10, valinit=1.0)


def update(val):
    N_reg = int(s_N.val)
    sigma = s_smooth.val
    rho_c_min, rho_c_max = s_rho_c.val
    w = [s_w1.val, s_w2.val, s_w3.val, s_w4.val]
    use_conv = cb_conv.get_status()[0]

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

    qe_curr = Qe_curr - (1.5 * Gamma_curr * Te * E_CHARGE)
    qi_curr = Qi_curr - (1.5 * Gamma_curr * Ti * E_CHARGE)

    l_Gamma_data.set_ydata(Gamma_curr)
    l_qe_data.set_ydata(qe_curr)
    l_qi_data.set_ydata(qi_curr)

    r = run_fit(N_reg, *w, dn_curr, dTe_curr, dTi_curr, Gamma_curr, qe_curr, qi_curr, rho_c_min, rho_c_max, use_conv)

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


for s in [s_smooth, s_N, s_rho_c, s_w1, s_w2, s_w3, s_w4]:
    s.on_changed(update)
cb_conv.on_clicked(update)

if __name__ == "__main__":
    plt.show()
