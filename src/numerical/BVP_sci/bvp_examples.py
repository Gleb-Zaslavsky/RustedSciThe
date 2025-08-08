import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def lane_emden(x, y):
    return np.vstack((y[1], -2/x * y[1] - y[0]**5))

def bc(ya, yb):
    return np.array([ya[0] - 1, yb[1]])

x = np.linspace(1e-10, 2, 100)  # Avoid division by zero at x=0
y_guess = np.zeros((2, x.size))
y_guess[0] = 1 - x**2/6  # Initial guess based on Taylor expansion near 0

sol = solve_bvp(lane_emden, bc, x, y_guess)

# Exact solution
def exact_lane_emden(x):
    return (1 + x**2/3)**(-0.5)

x_plot = np.linspace(0, 2, 100)
y_num = sol.sol(x_plot)[0]
y_exact = exact_lane_emden(x_plot)

plt.figure()
plt.plot(x_plot, y_num, label='Numerical')
plt.plot(x_plot, y_exact, '--', label='Exact')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Lane-Emden Equation (n=5)")
plt.show()

# Check error
error_lm = np.max(np.abs(y_num - y_exact))
print("lane emden")
print(f"Maximum error: {error_lm:.2e}")
############################################################

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

k = 3
g = 5

def parachute(x, y):
    return np.vstack((y[1], -k*y[1]**2 + g))

def bc(ya, yb):
    return np.array([ya[0], ya[1]])

x = np.linspace(0, 0.6, 100)
y_guess = np.zeros((2, x.size))
y_guess[0] = g*x**2/2  # Initial guess ignoring drag
y_guess[1] = g*x

sol = solve_bvp(parachute, bc, x, y_guess)

# Exact solution
def exact_parachute(x):
    sqrt_gk = np.sqrt(g*k)
    return (1/k) * (np.log((np.exp(2*sqrt_gk*x) + 1)/2) - sqrt_gk*x)

x_plot = np.linspace(0, 0.6, 100)
y_num = sol.sol(x_plot)[0]
y_exact = exact_parachute(x_plot)

plt.figure()
plt.plot(x_plot, y_num, label='Numerical')
plt.plot(x_plot, y_exact, '--', label='Exact')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Parachute Equation")
plt.show()

# Check error
error_par = np.max(np.abs(y_num - y_exact))
print(f"Maximum error: {error_par:.2e}")
##########################################################################################################

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

a = 4

def exponential_bvp(x, y):
    return np.vstack((y[1], -(2/a)*(1 + 2*np.log(y[0]))*y[0]))

def bc(ya, yb):
    return np.array([ya[0] - np.exp(-1/a), yb[0] - np.exp(-1/a)])

x = np.linspace(-1, 1, 100)
y_guess = np.zeros((2, x.size))
y_guess[0] = np.exp(-x**2/a)  # Good initial guess

sol = solve_bvp(exponential_bvp, bc, x, y_guess)

# Exact solution
def exact_exponential(x):
    return np.exp(-x**2/a)

x_plot = np.linspace(-1, 1, 100)
y_num = sol.sol(x_plot)[0]
y_exact = exact_exponential(x_plot)

plt.figure()
plt.plot(x_plot, y_num, label='Numerical')
plt.plot(x_plot, y_exact, '--', label='Exact')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Exponential BVP")
plt.show()

# Check error
error_exp = np.max(np.abs(y_num - y_exact))
print(f"Maximum error EXP: {error_exp:.2e}")


print(f"Maximum error LM: {error_lm:.2e}")

print(f"Maximum error PAR: {error_par:.2e}")