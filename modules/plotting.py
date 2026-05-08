import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# symbolic variable
x = sp.symbols('x')

# define function
f = x**3 - 3*x**2 + 2

# derivative
f_prime = sp.diff(f, x)

# integral
f_integral = sp.integrate(f, x)

print("f(x) =", f)
print("f'(x) =", f_prime)
print("Integral =", f_integral)

# convert symbolic expressions to numerical functions
f_num = sp.lambdify(x, f, 'numpy')
f_prime_num = sp.lambdify(x, f_prime, 'numpy')
f_integral_num = sp.lambdify(x, f_integral, 'numpy')

# x values
xs = np.linspace(-2, 4, 500)

# plot
plt.figure(figsize=(10,6))
plt.plot(xs, f_num(xs), label='f(x)')
plt.plot(xs, f_prime_num(xs), label="f'(x)")
plt.plot(xs, f_integral_num(xs), label='Integral')

plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()