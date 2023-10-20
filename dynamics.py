import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
R_m = 1.05
f_R1 = 0.1
B = 1.2
c = 0.05  # Interaction term

# Define the equations for y and k for each period
def y0(A, c, c_e):
    numerator = R_m + f_R1 + c*f_R1*(10*B*A - 10*f_R1 - 10*c_e) - c_e*(1 - 10*B*A*c + 10*c*f_R1 - 10*c_e*c)
    denominator = 0.4 + (20*f_R1*c_e*c**2) - (10*c**2*c_e**2) - (10*f_R1**2*c**2)
    return numerator / denominator

def k0(A, c, c_e):
    numerator = 10*B*A + 10*f_R1 - 10*c_e + (f_R1*(10*R_m*c + 10*f_R1*c - 100*f_R1**2*c**2 - 100*c_e*c + 300*B*A*c**2*c_e) + c_e*(10*R_m*c - 110*c_e*c - 100*c_e**2*c**2))
    denominator = 0.4 + (20*f_R1*c_e*c**2) - (10*c**2*c_e**2) - (10*f_R1**2*c**2)
    return numerator / denominator

# Create a range of A and c_e values for plotting
A_values = np.linspace(1.7, 2.7, 100)
ce_values = np.linspace(0.005, 3, 100)

c_values = [0, 0.05, 0.5]

# Create a figure with 2 rows and 2 columns of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

for c in c_values:
    # Top-left: Dynamics of y0 with varying A (fixed c_e)
    y_values_A = [y0(A, c, 0.005) for A in A_values]
    axs[0, 0].plot(A_values, y_values_A, label=f'c={c}')
    axs[0, 0].set_xlabel('A values')
    axs[0, 0].set_ylabel('y0 values')
    axs[0, 0].set_title('Dynamics of y0 with varying A')
    axs[0, 0].grid(False)
    axs[0, 0].legend()

    # Top-right: Dynamics of y0 with varying c_e (fixed A)
    y_values_ce = [y0(1.7, c, ce) for ce in ce_values]
    axs[0, 1].plot(ce_values, y_values_ce, label=f'c={c}')
    axs[0, 1].set_xlabel('c_e values')
    axs[0, 1].set_ylabel('y0 values')
    axs[0, 1].set_title('Dynamics of y0 with varying c_e')
    axs[0, 1].grid(False)
    axs[0, 1].legend()

    # Bottom-left: Dynamics of k0 with varying A (fixed c_e)
    k_values_A = [k0(A, c, 0.005) for A in A_values]
    axs[1, 0].plot(A_values, k_values_A, label=f'c={c}')
    axs[1, 0].set_xlabel('A values')
    axs[1, 0].set_ylabel('k0 values')
    axs[1, 0].set_title('Dynamics of k0 with varying A')
    axs[1, 0].grid(False)
    axs[1, 0].legend()

    # Bottom-right: Dynamics of k0 with varying c_e (fixed A)
    k_values_ce = [k0(1.7, c, ce) for ce in ce_values]
    axs[1, 1].plot(ce_values, k_values_ce, label=f'c={c}')
    axs[1, 1].set_xlabel('c_e values')
    axs[1, 1].set_ylabel('k0 values')
    axs[1, 1].set_title('Dynamics of k0 with varying c_e')
    axs[1, 1].grid(False)
    axs[1, 1].legend()

# Adjust the layout to ensure the plots don't overlap
plt.tight_layout()
plt.show()
