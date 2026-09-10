import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. PLOT FORMATTING (POWERPOINT & LATEX)
# ==========================================
# Use serif fonts and Computer Modern math text for a native LaTeX look
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 14,
    'axes.titlesize': 18,    
    'axes.labelsize': 16,    
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.bottom': True,
    'ytick.left': True,
    'axes.formatter.use_mathtext': True
})

# ==========================================
# 1. GENERATE DATA
# ==========================================
# Create a normalized velocity axis matching the reference [0, 10]
v = np.linspace(0, 10, 500)

# 1a. Gaussian Operator f2(v, 1, 1, 1, 1)
# B_s * exp(-beta_s * m_s * v^2 / (2 * T_s))
f_gauss = np.exp(-v**2 / 2.0)

# 1b. Sinc Operator f1(v, 1, 1, 1, 1)
# A_s * Sinc( alpha_s * (m_s * v^2 / (2 * T_s))^(1/2) )
# Note: The reference plot uses the unnormalized mathematical Sinc(x) = sin(x)/x.
# Numpy's np.sinc(x) computes the normalized sin(pi*x)/(pi*x). 
# To get the unnormalized version, we divide the argument by pi.
arg = v / np.sqrt(2.0)
f_sinc = np.sinc(arg / np.pi) 

# ==========================================
# 2. PLOTTING THE FIGURE
# ==========================================
# figsize=(7, 10) creates two stacked, roughly square plots
fig, axs = plt.subplots(2, 1, figsize=(4*0.7, 8*0.7), sharex=True)
plt.subplots_adjust(hspace=0.25) # Add breathing room for the large LaTeX titles

LW = 4 # Line thickness
c_line = 'tab:blue' # Matches the Mathematica default blue from the reference
c_line =  '#daa520'

# --- TOP PANEL: GAUSSIAN ---
axs[0].plot(v, f_gauss, color=c_line, lw=LW, label=r'$f_2(v, \mathbf{1}, \mathbf{1}, \mathbf{1}, \mathbf{1})$')
# axs[0].set_ylabel(r"$f_2(v, 1, 1, 1, 1) \, [\mathrm{s}^3\cdot\mathrm{m}^{-6}]$")
# axs[0].set_title(r"$f_2(v, B_s, \beta_s, m_s, T_s) = B_s \exp\left( -\frac{\beta_s m_s v^2}{2 T_s} \right)$", pad=15)

# --- BOTTOM PANEL: SINC ---
axs[1].plot(v, f_sinc, color=c_line, lw=LW, label=r'$f_1(v, \mathbf{1}, \mathbf{1}, \mathbf{1}, \mathbf{1})$')
#axs[1].set_ylabel(r"$f_1(v, 1, 1, 1, 1) \, [\mathrm{s}^3\cdot\mathrm{m}^{-6}]$")
# axs[1].set_title(r"$f_1(v, A_s, \alpha_s, m_s, T_s) = A_s \mathrm{Sinc}\left( \alpha_s \left( \frac{m_s v^2}{2 T_s} \right)^{\frac{1}{2}} \right)$", pad=15)
# axs[1].set_xlabel(r"$v \, [\mathrm{m}\cdot\mathrm{s}^{-1}]$")
axs[1].set_xlabel(r"$v$")

# --- COMMON FORMATTING ---
for ax in axs:
    # Explicit zero line for visual grounding
    ax.axhline(0, color='black', lw=1.5, alpha=0.4, zorder=0)
    
    # Restrict axes exactly as seen in the reference images
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 1.05) 
    
    # Legends matching the reference styling
    #ax.legend(loc='upper right', framealpha=0.9, handlelength=1.0)
    
    # Presentation-ready styling
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6, width=1.5)
    
    # Keep the number of ticks minimal for clean presentation
    ax.locator_params(axis='both', nbins=4)
    
    # Enhance spine (border) thickness for presentation clarity
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_alpha(0.8)

plt.savefig("presentation_operators_stacked.png", dpi=300, bbox_inches='tight')
plt.show()