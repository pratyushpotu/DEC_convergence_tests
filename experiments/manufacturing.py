import numpy as np
import sympy as sym

#Solve for scaled 15th order bubble function
def u_ex(x, y):
    l1 = 1 - x - y / np.sqrt(3)
    l2 = x - y / np.sqrt(3)
    l3 = 2 * y / np.sqrt(3)
    return 1e8 * l1**5 * l2**5 * l3**5

#Use symbolic differentiation to compute exact gradient and exact Laplacian
x, y = sym.symbols('x y')
sqrt3 = sym.sqrt(3)
l1 = 1 - x - y / sqrt3
l2 = x - y / sqrt3
l3 = 2 * y / sqrt3
u = 1e8 * l1**5 * l2**5 * l3**5

f_expr = -sym.diff(u, x, 2) - sym.diff(u, y, 2) #f = -\Delta u
f_func = sym.lambdify((x, y), f_expr, modules="numpy")
def f_rhs(x, y):
    return f_func(x, y)


grad_x_term = sym.diff(u, x)
grad_y_term = sym.diff(u, y)

grad_x_term_func = sym.lambdify((x, y), grad_x_term, modules="numpy")
grad_y_term_func = sym.lambdify((x, y), grad_y_term, modules="numpy")

def grad_u(x, y):
    return np.array([grad_x_term_func(x, y), grad_y_term_func(x, y)])


#For k=1
u_func = sym.lambdify((x, y), u, "numpy")

def u_vector(x, y):
    return np.array([u_func(x, y), u_func(x, y)]) #Take u to be the bubble function in each component

def f_vector(x, y):
    return np.array([f_func(x, y), f_func(x, y)]) #Then, rhs is f = -\Delta u in each component

#For computing e_rho
div_u = sym.diff(u, x, 1) + sym.diff(u, y, 1)
delta_u_func = sym.lambdify((x, y), -div_u, "numpy") # delta u = -div u for k=1

def delta_u(x,y):
    return delta_u_func(x,y)

#For k=2 computing e_rho
rot_x_term = sym.diff(u, y, 1)
rot_y_term = -sym.diff(u, x, 1)

rot_x_term_func = sym.lambdify((x, y), rot_x_term, modules="numpy")
rot_y_term_func = sym.lambdify((x, y), rot_y_term, modules="numpy")

def delta2_u(x, y):
    return np.array([rot_x_term_func(x, y), rot_y_term_func(x, y)])