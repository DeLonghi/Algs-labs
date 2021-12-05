import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.linear_model import SGDRegressor

Eps = 0.0001 
K = 101
alpha = np.random.rand(1)
beta = np.random.rand(1)

mu, sigma = 0, 0.1
s = np.random.normal(0, 1, 101)


def fL(k, x):
    return alpha * x + beta + s[k]

def linear(x, a, b):
    return a * x + b

def rational(x, a, b):
    return a / (1 + b * x)

def funcToOptLin(a_b, *params):
    x, y = params
    a = a_b[0]
    b = a_b[1]
    sum = 0
    for i in range(K):
        sum += (linear(x[i], a, b) - y[i])**2
    return sum

def funcToOptRat(a_b, *params):
    x, y = params
    a = a_b[0]
    b = a_b[1]
    sum = 0
    for i in range(K):
        sum += (rational(x[i], a, b) - y[i])**2
    # print(sum)
    return sum



def form_data(f):
    y = []
    x = []
    for i in range(K):
        _x = i / 100
        x.append(_x)
        y.append(f(i, _x))
    return x, y

x_values, y = form_data(fL)


def getGradLin(a_b, *params):
    grads = np.array([])
    x, y = params
    a = a_b[0]
    b = a_b[1]
    a_grad = 0
    b_grad = 0
    for i in range(K):
        a_grad += -2 * x[i] * (-1 * x[i] * a - b + y[i])
        b_grad += -2 * (-1 * x[i] * a - b + y[i])
    grads = np.append(grads, (a_grad, b_grad))
    return grads

def getGradRat(a_b, *params):
    grads = np.array([])
    x, y = params
    a = a_b[0]
    b = a_b[1]
    a_grad = 0
    b_grad = 0
    for i in range(K):
        a_grad += (2 * (a - y[i] * (x[i] * b + 1) )) / ((x[i] * b + 1) **2)
        b_grad += -1 *(2 * x[i] * a * (a - y[i] * (x[i] * b + 1) )) / ((x[i] * b + 1) **3)
    grads = np.append(grads, (a_grad, b_grad))
    return grads

def ResToOptLin(a_b, *params):
    res = np.array([])
    x, y = params
    a = a_b[0]
    b = a_b[1]
    for i in range(K):
        res = np.append(res, (linear(x[i], a, b) - y[i])**2)
    return res


def getJacobianRat(a_b, *params):
    res = optimize.approx_fprime(a_b, funcToOptRat, [Eps, np.sqrt(200) * Eps], *params)
    return res

def getJacobianLin(a_b, *params):
    res = optimize.approx_fprime(a_b, funcToOptLin, [Eps, np.sqrt(200) * Eps], *params)
    return res

def ResToOptRat(a_b, *params):
    res = np.array([])
    x, y = params
    a = a_b[0]
    b = a_b[1]
    for i in range(K):
        res = np.append(res, (rational(x[i], a, b) - y[i])**2)
    return res

params = (x_values, y)


res_gd = optimize.minimize(funcToOptLin, [0, 0], method='BFGS', args=params, tol=Eps, options={'disp':True})
res_cg = optimize.minimize(funcToOptLin, [0, 0], method='CG', args=params, tol=Eps, options={'disp':True})
res_N_cg = optimize.minimize(funcToOptLin, [0, 0], method='Newton-CG', args=params, tol=Eps,  options={'disp':True}, jac=getGradLin)
res_lma = optimize.least_squares(ResToOptLin, [0, 0], method='lm', args=params, xtol=Eps)
x_values = np.array(x_values)


fig, axes = plt.subplots()
plt.plot(x_values, y, label='Generated data')
plt.plot(x_values, x_values * alpha + beta, label='Initial line')
plt.plot(x_values, x_values * res_gd.x[0] + res_gd.x[1], label='Gradient descent method')
plt.plot(x_values, x_values * res_cg.x[0] + res_cg.x[1], label='Conjugate gradient method')
plt.plot(x_values, x_values * res_N_cg.x[0] + res_N_cg.x[1], label='Newton method')
plt.plot(x_values, x_values * res_lma.x[0] + res_lma.x[1], label='Levenberg-Marquardt method')
axes.set_title("Linear approximation")
axes.set_xlabel('X')
axes.set_ylabel('Y')
plt.legend()
plt.grid()

res_gd = optimize.minimize(funcToOptRat, [0, 0], method='BFGS', args=params, tol=Eps, options={'disp':True})
res_cg = optimize.minimize(funcToOptRat, [0, 0], method='CG', args=params, tol=Eps, options={'disp':True})
res_N_cg = optimize.minimize(funcToOptRat, [0, 0], method='Newton-CG', args=params, tol=Eps, options={'disp':True}, jac=getJacobianRat)
res_lma = optimize.least_squares(ResToOptRat, [0, 0], method='lm', args=params)
# x_values = np.array(x_values)


fig, axes = plt.subplots()
plt.plot(x_values, y, label='Generated data')
plt.plot(x_values, x_values * alpha + beta, label='Initial line')
plt.plot(x_values, res_gd.x[0] / (1 + x_values * res_gd.x[1]), label='Gradient descent method')
plt.plot(x_values, res_gd.x[0] / (1 + x_values * res_gd.x[1]), label='Conjugate gradient method')
plt.plot(x_values, res_gd.x[0] / (1 + x_values * res_gd.x[1]), label='Newton method')
plt.plot(x_values, res_gd.x[0] / (1 + x_values * res_gd.x[1]), label='Levenberg-Marquardt method')
axes.set_title("Rational approximation")
axes.set_xlabel('X')
axes.set_ylabel('Y')
plt.legend()
plt.grid()
plt.show()