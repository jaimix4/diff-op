import numpy as np
import matplotlib.pyplot as plt
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
# 1. GENERATE GAUSSIAN DATA
# ==========================================
# Create a normalized velocity axis
v = np.linspace(-4, 10, 500)

# Simple Gaussian: D_s(v) = D_0 * exp(-v^2 / (2*sigma^2))
sigma = 1.0
D_0 = 1.0
D_s = D_0 * np.exp(-(v - 1.5)**2 / (2 * sigma**2))

# ==========================================
# 2. PLOTTING THE FIGURE
# ==========================================
fig, ax = plt.subplots(figsize=(7*0.7, 6*0.7))

LW = 5.0 # Very thick line to match the icon style in your slide
c_gaussian = '#daa520' # Goldenrod, matching your presentation's warm palette



# Plot the thick dashed centerline (the mean)
ax.axvline(0, color='black', linestyle='--', lw=LW-3, alpha=0.5)
ax.axvline(1.5, color=c_gaussian, linestyle='--', lw=LW-2, alpha=0.5)

# Plot the main Gaussian curve
ax.plot(v, D_s, color=c_gaussian, lw=LW)

# Labels and Title
ax.set_xlabel(r"Velocity $v$")
ax.set_ylabel(r"$D_s(v) $")
# ax.set_title("Gaussian Velocity-Dependent Operator")

# Format axes to match the presentation style exactly
ax.grid(True, alpha=0.1)
ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6, width=1.5)

# Keep the number of ticks minimal for clean presentation
ax.locator_params(axis='both', nbins=5)

# Set limits: ground the y-axis exactly at 0
ax.set_xlim(-3, 6)
ax.set_ylim(bottom=0, top=1.1)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# Enhance spine (border) thickness for presentation clarity
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_alpha(0.8)

plt.tight_layout()
plt.savefig("presentation_gaussian_operator.png", dpi=300, bbox_inches='tight')
plt.show()