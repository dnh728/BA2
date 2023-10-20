import sympy as sp

# Define the variables and parameters symbolically
k, y, c_val = sp.symbols('k y c')
c = c_val * y * k  # Equation for c(y, k)

# Define the parameters
β, A, w, L, R_m, f_R1, c_e, I_1i = sp.symbols('β A w L R_m f_R1 c_e I_1i')

# Define the objective function W_pre symbolically
W_pre = (β * A * k - 0.05 * k**2) + (w * L + R_m * y - 0.2 * y**2) + (f_R1 * (y + k + c) - c_e * (y + k + c) - I_1i)

# Compute the partial derivatives
partial_k = sp.diff(W_pre, k)
partial_y = sp.diff(W_pre, y)

# Set the partial derivatives to zero to get the equations for the optimal values
equation_k = sp.Eq(partial_k, 0)
equation_y = sp.Eq(partial_y, 0)

# Solve the system of equations
solutions = sp.solve((equation_k, equation_y), (k, y))

# Print the solutions
print("Solutions:", solutions)


import sympy as sp

# Define the variables
k, y = sp.symbols('k y')

# Given parameter values
β, A, w, L, R_m, f_R1, c_e, I_1i, c_val = 1.2, 1.7, 1, 1, 1.05, 0.1, 0.005, 1, 0.05
c = c_val * y * k  # Equation for c(y, k)

# Equations for optimal k and y
k_optimal = (-10.0*A*β + 25.0*R_m*c_val*c_e - 25.0*R_m*c_val*f_R1 - 25.0*c_val*c_e**2 + 50.0*c_val*c_e*f_R1 - 25.0*c_val*f_R1**2 + 10.0*c_e - 10.0*f_R1)/(25.0*c_val**2*c_e**2 - 50.0*c_val**2*c_e*f_R1 + 25.0*c_val**2*f_R1**2 - 1.0)
y_optimal = (50.0*A*c_val*c_e*β - 50.0*A*c_val*f_R1*β - 5.0*R_m - 50.0*c_val*c_e**2 + 100.0*c_val*c_e*f_R1 - 50.0*c_val*f_R1**2 + 5.0*c_e - 5.0*f_R1)/(50.0*c_val**2*c_e**2 - 100.0*c_val**2*c_e*f_R1 + 50.0*c_val**2*f_R1**2 - 2.0)

# Print the results
print("Optimal k:", k_optimal)
print("Optimal y:", y_optimal)


