from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# Define the Nash equilibrium functions
def y_re(tau, delta=1, R_m=1.05):
    return delta * R_m / (2 * (0.2 + tau))

def k_re(tau, delta=1, beta=1.2, A=1.7):
    return delta * beta * A / (2 * (0.05 + tau))

# Define the objective function
def objective(tau):
    return (y_re(tau[0]) - 3.2)**2 + (k_re(tau[0]) - 21.4)**2

# Callback function to print intermediate results
def callback(xk):
    print(f"Current tau: {xk[0]}, Objective value: {objective(xk)}")

# Initial guess for tau
tau_initial = [0]

# Solve the optimization problem
result = minimize(objective, tau_initial, bounds=[(-0.005, 3)], method='L-BFGS-B', callback=callback)

optimal_tau = result.x[0]
print(f"Optimal tau: {optimal_tau}")


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
plt.savefig("nash_equilibrium_subplots.png", dpi=300)  # Save the figure in high resolution
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

# Pre-pandemic values
y_pre_pandemic = 3.2
k_pre_pandemic = 21.4

# Optimal tau value and corresponding investment levels
optimal_tau = -0.0023726119876094653
y_optimal, k_optimal = nash_equilibrium(optimal_tau)

# Generate tau values and Nash equilibrium values
tau_values = np.linspace(-0.005, 3, 100)
y_nash_values = [nash_equilibrium(tau)[0] for tau in tau_values]
k_nash_values = [nash_equilibrium(tau)[1] for tau in tau_values]

# Find the intersection point
idx = np.where(np.array(k_nash_values) >= k_pre_pandemic)[0][0]
y_intersect = y_nash_values[idx]
tau_intersect = tau_values[idx]

# Plotting
plt.figure(figsize=(10, 6))

# Plot Nash equilibrium curves
plt.plot(k_nash_values, y_nash_values, 'g-', label='Nash Equilibrium (τ variation)')

# Highlight the pre-pandemic point
plt.scatter(k_pre_pandemic, y_pre_pandemic, color='blue', s=100, edgecolors='black', zorder=5, label='Pre-Pandemic')
plt.axvline(k_pre_pandemic, color='black', linestyle='--', ymin=0, ymax=(y_intersect/max(y_nash_values)))

# Highlight the optimal tau point
plt.scatter(k_optimal, y_optimal, color='purple', s=100, edgecolors='black', zorder=6, label=f'Optimal τ = {optimal_tau:.4f}')

plt.title("Investment Strategies and Nash Equilibrium", fontsize=14)
plt.xlabel("Investment fraction by Speculators (k)", fontsize=12)
plt.ylabel("Investment fraction by Main Street (y)", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()



from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

# Define the utility functions and Nash equilibrium
def utility_main_street(y, R_m=1.02,w=1,L=1):
    return w*L*R_m*y - 0.2*y**2

def utility_speculator(k, beta=1.2, A=1.7):
    return beta*A*k - 0.05*k**2

def nash_equilibrium(tau, delta=1, R_m=1.05, beta=1.2, A=1.7):
    # Define the system of equations for the Nash equilibrium
    def equations(p):
        y, k = p
        # Derivative of utility function for Main Street
        dy = R_m - 0.4*y - 2*tau*y
        # Derivative of utility function for Speculators
        dk = beta*A - 0.1*k - 2*tau*k
        return (dy, dk)

    # Solve the system of equations
    y_re, k_re = fsolve(equations, (1, 1))
    return y_re, k_re

# Pre-pandemic values
y_pre_pandemic = 3.2
k_pre_pandemic = 21.4

# Optimal tau value and corresponding investment levels
optimal_tau = -0.0023726119876094653
y_optimal, k_optimal = nash_equilibrium(optimal_tau)

# Generate tau values and Nash equilibrium values
tau_values = np.linspace(-0.005, 3, 100)
y_nash_values = [nash_equilibrium(tau)[0] for tau in tau_values]
k_nash_values = [nash_equilibrium(tau)[1] for tau in tau_values]

# Find the intersection point
idx = np.where(np.array(k_nash_values) >= k_pre_pandemic)[0][0]
y_intersect = y_nash_values[idx]
tau_intersect = tau_values[idx]

# Plotting
plt.figure(figsize=(10, 6))

# Plot Nash equilibrium curves
plt.plot(k_nash_values, y_nash_values, 'g-', label='Nash Equilibrium (τ variation)')

# Highlight the pre-pandemic point
plt.scatter(k_pre_pandemic, y_pre_pandemic, color='blue', s=100, edgecolors='black', zorder=5, label='Pre-Pandemic')
plt.axvline(k_pre_pandemic, color='black', linestyle='--', ymin=0, ymax=(y_intersect/max(y_nash_values)))

# Highlight the optimal tau point
plt.scatter(k_optimal, y_optimal, color='purple', s=100, edgecolors='black', zorder=6, label=f'Optimal τ = {optimal_tau:.4f}')

plt.title()
plt.xlabel("Investment fraction by Speculators (k)", fontsize=14)
plt.ylabel("Investment fraction by Main Street (y)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()



from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

# Define the utility functions and Nash equilibrium
def utility_main_street(y, R_m=1.02):
    return R_m*y - 0.2*y**2

def utility_speculator(k, beta=1.2, A=1.7):
    return beta*A*k - 0.05*k**2

def nash_equilibrium(tau, delta=1, R_m=1.05, beta=1.2, A=1.7):
    # Define the system of equations for the Nash equilibrium
    def equations(p):
        y, k = p
        # Derivative of utility function for Main Street
        dy = R_m - 0.4*y - 2*tau*y
        # Derivative of utility function for Speculators
        dk = beta*A - 0.1*k - 2*tau*k
        return (dy, dk)

    # Solve the system of equations
    y_re, k_re = fsolve(equations, (1, 1))
    return y_re, k_re

# Pre-pandemic values
y_pre_pandemic = 3.2
k_pre_pandemic = 21.4

# Optimal tau value and corresponding investment levels
optimal_tau = -0.0023726119876094653
y_optimal, k_optimal = nash_equilibrium(optimal_tau)

# Generate tau values and Nash equilibrium values
tau_values = np.linspace(-0.005, 3, 100)
y_nash_values = [nash_equilibrium(tau)[0] for tau in tau_values]
k_nash_values = [nash_equilibrium(tau)[1] for tau in tau_values]

# Find the intersection point
idx = np.where(np.array(k_nash_values) >= k_pre_pandemic)[0][0]
y_intersect = y_nash_values[idx]
tau_intersect = tau_values[idx]

# Calculate the Euclidean distance between the pre-pandemic point and the optimal tau point
distance = np.sqrt((k_optimal - k_pre_pandemic)**2 + (y_optimal - y_pre_pandemic)**2)

# Plotting
plt.figure(figsize=(10, 6))

# Plot Nash equilibrium curves
plt.plot(k_nash_values, y_nash_values, 'g-', label='Nash Equilibrium (τ variation)')

# Highlight the pre-pandemic point
plt.scatter(k_pre_pandemic, y_pre_pandemic, color='blue', s=100, edgecolors='black', zorder=5, label='Pre-Pandemic')
plt.axvline(k_pre_pandemic, color='black', linestyle='--', ymin=0, ymax=(y_intersect/max(y_nash_values)))
plt.annotate(f"({k_pre_pandemic}, {y_pre_pandemic})", (k_pre_pandemic, y_pre_pandemic), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

# Highlight the optimal tau point
plt.scatter(k_optimal, y_optimal, color='purple', s=100, edgecolors='black', zorder=6, label=f'Optimal τ = {optimal_tau:.4f}')
plt.annotate(f"({k_optimal:.4f}, {y_optimal:.4f})", (k_optimal, y_optimal), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

# Add distance annotation
plt.annotate(f"Distance: {distance:.4f}", ((k_optimal + k_pre_pandemic) / 2, (y_optimal + y_pre_pandemic) / 2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='red')

plt.title("Investment Strategies and Nash Equilibrium", fontsize=14)
plt.xlabel("Investment fraction by Speculators (k)", fontsize=14)
plt.ylabel("Investment fraction by Main Street (y)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
