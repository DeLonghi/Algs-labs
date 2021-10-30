import math
import numpy
from scipy import optimize
from scipy.optimize.optimize import golden
import random

Eps = 0.001
phi = (1 + math.sqrt(5))/2
resphi = 2 - phi

def f1(x):
    return numpy.power(x, 3)

def f2(x):
    return abs(x - 0.2)

def f3(x):
    return x * math.sin(1 / x)

counter = 0

def goldenSectionSearch(f, a, c, b, absolutePrecision):
    global counter
    counter += 1
    if abs(a - b) < absolutePrecision:
        print(counter)
        return (a + b)/2
    # Create a new possible center, in the area between c and b, pushed against c
    d = c + resphi*(b - c)
    if f(d) < f(c):
        return goldenSectionSearch(f, c, d, b, absolutePrecision)
    else:
        return goldenSectionSearch(f, d, c, a, absolutePrecision)

def dichotomy(f, a, b, absolutePrecision):
    d = random.uniform(0, absolutePrecision)
    while (b- a) >= absolutePrecision:
        global counter
        counter += 1
        x1 = (a + b - d) / 2
        x2 = (a + b + d) / 2
        if (f(x1) <= f(x2)):
            b = x2
        else:
            a = x1
    print(counter)
    return (a+ b) / 2

print(" Golden-section search:")
f1_g = optimize.minimize_scalar(f1, method="golden", tol=Eps, options={'xtol': Eps})
f2_g = optimize.minimize_scalar(f2, method="golden", tol=Eps, options={'xtol': Eps})
f3_g = optimize.minimize_scalar(f3, bracket=(0.1, 1), method="golden", tol=Eps)


print("\n   Dichotomy:")
print(dichotomy(f1, 0, 1, Eps))
print(dichotomy(f2, 0, 1, Eps))
print(dichotomy(f3, 0.1, 1, Eps))

print("\n   Brute:")

f1_b = optimize.brute(f1, ((0, 1),), Ns=1/Eps, full_output=True, finish=None)
f2_b = optimize.brute(f2, ((0, 1),), Ns=1/Eps, full_output=True, finish=None)
f3_b = optimize.brute(f3, ((0.001, 1),), Ns=1/Eps, disp=True, full_output=True, finish=None)