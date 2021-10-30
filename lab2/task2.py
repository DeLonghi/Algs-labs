import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

Eps = 0.001 
K = 101
a = random.uniform(0, 1)
b = random.uniform(0, 1)
print(a, b)

mu, sigma = 0, 0.1
s = np.random.normal(0, 1, 101)


def fL(k, x):
    return a * x + b + s[k]

def fR(k, x):
    return (b + s[k]) / (1 + a * x )

def linear(x, a, b):
    return a * x + b

def rational(x, a, b):
    return b / (1 + a * x)

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


params = (x_values, y)

res_Brute= optimize.brute(funcToOptLin, (slice(0, 1), slice(0, 1)), Ns=100, args=params)
res_Powell = optimize.minimize(funcToOptLin, (0.3, 0.5), bounds=((0,1), (0,1)), method='Powell', args=params,  options={'disp':True})
res_Nedler_Mead = optimize.minimize(funcToOptLin, (0.3, 0.5), bounds=((0,1), (0,1)),
                                                    method='Nelder-Mead', args=params, options={'xatol':Eps, 'disp':True})

x_values = np.array(x_values)

fig, axes = plt.subplots()
plt.plot(x_values, y, "o", label='Generated data')
plt.plot(x_values, x_values * a + b, label='Initial line')
plt.plot(x_values, x_values * res_Brute[0] + res_Brute[1], label='Brute method')
plt.plot(x_values, x_values * res_Powell.x[0] + res_Powell.x[1], label='Gaus method')
plt.plot(x_values, x_values * res_Nedler_Mead.x[0] + res_Nedler_Mead.x[1], label='Nedler-Mead method')
axes.set_title("Linear approximation")
axes.set_xlabel('X')
axes.set_ylabel('Y')
plt.legend()
plt.grid()


res_Brute = optimize.brute(funcToOptRat, (slice(0, 1), slice(0, 1)), Ns=1/Eps, args=params)
res_Powell = optimize.minimize(funcToOptRat, (0.3, 0.5), bounds=((0,1), (0,1)), 
                                            method='Powell', args=params, options={'disp':True})
res_Nedler_Mead = optimize.minimize(funcToOptRat, (0.3, 0.5), bounds=((0,1), (0,1)), 
                                                method='Nelder-Mead', args=params, options={'xatol':Eps, 'disp':True})

print(res_Powell.x)
print(res_Nedler_Mead.x)
print(res_Brute)


fig, axes = plt.subplots()
plt.plot(x_values, y, "o", label='Generated data')
plt.plot(x_values, x_values * a + b, label='Initial line')
plt.plot(x_values, x_values * res_Brute[0] + res_Brute[1], label='Brute method')
plt.plot(x_values, x_values * res_Powell.x[0] + res_Powell.x[1], label='Gaus method')
plt.plot(x_values, x_values * res_Nedler_Mead.x[0] + res_Nedler_Mead.x[1], label='Nedler-Mead method')
axes.set_title("Rational approximation")
axes.set_xlabel('X')
axes.set_ylabel('Y')
plt.legend()
plt.grid()
plt.show()