import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.linear_model import SGDRegressor

Eps = 0.001 
K = 1001

mu, sigma = 0, 0.1
s = np.random.normal(0, 1, K)


def F(x, a, b, c, d):
    return (a * x + b) / (x**2 + c * x + d)

def getY(x):
    y = []
    for i in range(K):
        _x = 3 * i / 1000
        y.append(F(_x, x[0], x[1], x[2], x[3]))
    return y


def form_data():
    y = []
    x = []
    for i in range(K):
        _x = 3 * i / 1000
        x.append(_x)
        fx = F(_x, 1, 0, -3, 2)
        if (fx < -100):
            y.append(100 + s[i])
        elif (-100 <= fx <= 100):
            y.append(fx + s[i])
        else:
            y.append(100 + s[i])
    return x, y

x_values, y = form_data()



def ResToOpt(a_b_c_d, *params):
    res = np.array([])
    x, y = params
    a = a_b_c_d[0]
    b = a_b_c_d[1]
    c = a_b_c_d[2]
    d = a_b_c_d[3]
    for i in range(K):
        res = np.append(res, (F(x[i], a, b, c, d) - y[i]))
    return res

def funcToOpt(a_b_c_d, *params):
    x, y = params
    a = a_b_c_d[0]
    b = a_b_c_d[1]
    c = a_b_c_d[2]
    d = a_b_c_d[3]
    sum = 0
    for i in range(K):
        sum += (F(x[i], a, b, c, d) - y[i])**2
    return sum



params = (x_values, y)



res_Nedler_Mead = optimize.minimize(funcToOpt, (-2, -3, -1, 1), 
                                                method='Nelder-Mead', args=params,)
print(res_Nedler_Mead)
res_lma = optimize.least_squares(ResToOpt, (-2, -3, -1, 1), method='lm', args=params)
print(res_lma)
res_diff_ev = optimize.differential_evolution(funcToOpt, bounds=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)], tol=Eps, args=params)
print(res_diff_ev)
res_ann = optimize.dual_annealing(funcToOpt, bounds=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)], args=params)
print(res_ann)

fig, axes = plt.subplots()
plt.plot(x_values, y, ".", label='Generated data')
plt.plot(x_values, getY(res_Nedler_Mead.x), label='Nelder-Mead')
plt.plot(x_values, getY(res_lma.x), label='Levenberg-Marquardt method')
plt.plot(x_values, getY(res_diff_ev.x), label='Differential evolution')
plt.plot(x_values, getY(res_ann.x), label='Simulated annealing')
axes.set_title("Rational approximation")
axes.set_xlabel('X')
axes.set_ylabel('Y')
plt.legend()
plt.show()