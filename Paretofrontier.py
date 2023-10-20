import numpy as np
import matplotlib.pyplot as plt

# Define the utility functions
def utility_main_street(y, R_m=1.02):
    return R_m*y - 0.2*y**2

def utility_speculator(k, beta=1.2, A=1.7):
    return beta*A*k - 0.05*k**2

# Generate y and k values
y_range = np.linspace(2.5, 3.5, 100)
k_range = np.linspace(21, 25, 100)

# Calculate utility levels for Main Street and Speculators
utility_matrix = np.array([[utility_main_street(y) + utility_speculator(k) for k in k_range] for y in y_range])

# Optimal values
y_values = [3.2, 3, 3.1]
k_values = [21.4, 24.6, 22.5]
labels = ['Pre-Pandemic', 'Pandemic', 'Post-Pandemic']

# Plotting
plt.figure(figsize=(10, 6))
contour = plt.contour(k_range, y_range, utility_matrix, levels=50, colors='grey', linestyles='dashed')
plt.clabel(contour, inline=1, fontsize=10, fmt='%1.1f')

# Plot optimal values
for i, label in enumerate(labels):
    plt.scatter(k_values[i], y_values[i], label=label, s=100, edgecolors='black', zorder=5, cmap='viridis')

# Fit a polynomial of degree 2 (quadratic) through the points and plot
z = np.polyfit(k_values, y_values, 2)
p = np.poly1d(z)
k_plot = np.linspace(min(k_values), max(k_values), 100)
plt.plot(k_plot, p(k_plot), 'r-', label='Pareto Frontier')

plt.title("Investment Strategies Across Different Periods", fontsize=14)
plt.xlabel("Investment fraction by Speculators (k)", fontsize=12)
plt.ylabel("Investment fraction by Main Street (y)", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("research_figure.png", dpi=300)  # Save the figure in high resolution
plt.show()





##Nash Equilibrium


import numpy as np
import matplotlib.pyplot as plt

# Define the Nash equilibrium for the regulated scenario
def nash_equilibrium(tau, delta=1, R_m=1.05, beta=1.2, A=1.7):
    # Ensure the denominators are not too close to zero or negative
    denom_y = max(0.01, 2 * (0.2 + tau))
    denom_k = max(0.01, 2 * (0.05 + tau))
    
    y_re = (delta * R_m) / denom_y
    k_re = (delta * beta * A) / denom_k
    return y_re, k_re

# Generate a range of tau values (including negative values for subsidies)
tau_values = np.linspace(-0.2, 0.2, 100)
y_nash_values = []
k_nash_values = []

for tau in tau_values:
    y_re, k_re = nash_equilibrium(tau)
    y_nash_values.append(y_re)
    k_nash_values.append(k_re)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plot the Nash equilibrium for Main Street
axs[0].plot(tau_values, y_nash_values, 'b-', label='Main Street (y)')
axs[0].set_title()
axs[0].set_xlabel("Tau (\u03C4)", fontsize=10)
axs[0].set_ylabel("Investment fraction by Main Street (y)", fontsize=10)
axs[0].legend(fontsize=12)

# Plot the Nash equilibrium for Speculators
axs[1].plot(tau_values, k_nash_values, 'r-', label='Speculators (k)')
axs[1].set_title()
axs[1].set_xlabel("Tau (\u03C4)", fontsize=10)
axs[1].set_ylabel("Investment fraction by Speculators (k)", fontsize=10)
axs[1].legend(fontsize=12)

plt.tight_layout()
plt.savefig("nash_equilibrium_subplots.png", dpi=300) 
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the utility functions and Nash equilibrium
def utility_main_street(y, R_m=1.02):
    return R_m*y - 0.2*y**2

def utility_speculator(k, beta=1.2, A=1.7):
    return beta*A*k - 0.05*k**2

def nash_equilibrium(tau, delta=1, R_m=1.05, beta=1.2, A=1.7):
    denom_y = max(0.01, 2 * (0.2 + tau))
    denom_k = max(0.01, 2 * (0.05 + tau))
    y_re = (delta * R_m) / denom_y
    k_re = (delta * beta * A) / denom_k
    return y_re, k_re

# Generate tau values
tau_values = np.linspace(-0.01, 3, 100)
y_nash_values = [nash_equilibrium(tau)[0] for tau in tau_values]
k_nash_values = [nash_equilibrium(tau)[1] for tau in tau_values]

# Find the intersection point
k_pre_pandemic = 21.4
idx = np.where(np.array(k_nash_values) >= k_pre_pandemic)[0][0]
y_intersect = y_nash_values[idx]
tau_intersect = tau_values[idx]

# Plotting
plt.figure(figsize=(10, 6))

# Plot Nash equilibrium curves
plt.plot(k_nash_values, y_nash_values, 'g-', label='Nash Equilibrium (τ variation)')

# Highlight the pre-pandemic point and draw the vertical line
plt.scatter(k_pre_pandemic, 3.2, color='blue', s=100, edgecolors='black', zorder=5, label='Pre-Pandemic')
plt.axvline(k_pre_pandemic, color='black', linestyle='--', ymin=0, ymax=(y_intersect/max(y_nash_values)))

# Annotate the intersection point with the value of tau
plt.scatter(k_pre_pandemic, y_intersect, color='red', s=100, edgecolors='black', zorder=6)
plt.annotate(f'τ = {tau_intersect:.2f}', (k_pre_pandemic+0.5, y_intersect-0.2), fontsize=10, color='black')

# Fit a polynomial of degree 2 (quadratic) through the optimal points and plot
z = np.polyfit([21.4, 24.6, 22.5], [3.2, 3, 3.1], 2)
p = np.poly1d(z)
k_plot = np.linspace(min([21.4, 24.6, 22.5]), max([21.4, 24.6, 22.5]), 100)
plt.plot(k_plot, p(k_plot), 'r-', label='Pareto Frontier')

plt.title("Investment Strategies and Nash Equilibrium", fontsize=14)
plt.xlabel("Investment fraction by Speculators (k)", fontsize=12)
plt.ylabel("Investment fraction by Main Street (y)", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("combined_figure.png", dpi=300)
plt.show()


