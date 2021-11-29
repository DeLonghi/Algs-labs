from timeit import default_timer as timer
from matplotlib import pyplot as plt
import numpy as np
import string
import random


NUMBER_OF_EXPEREMENTS = 200
LETTERS = string.ascii_letters

def lcs(X, Y):
    m = len(X)
    n = len(Y)

    L = [[None]*(n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
    return L[m][n]


def measure_time(n_runs, s1, s2, linear=False):
    times = []
    for i in range(n_runs):
        times.append([])
        if linear:
            for j in range(1, NUMBER_OF_EXPEREMENTS):
                    _s1 = s1[:j]
                    _s2 = s2[:100]
                    start_time = timer()
                    lcs(_s1, _s2)
                    times[i].append(timer() - start_time)
        else:    
            for j in range(1, NUMBER_OF_EXPEREMENTS):
                    _s1 = s1[:j]
                    _s2 = s2[:j]
                    # print(_s1)
                    # print(_s2)
                    start_time = timer()
                    lcs(_s1, _s2)
                    times[i].append(timer() - start_time)

    return np.array(times, dtype=np.float64).mean(axis=0)

times = measure_time(5, ''.join(random.choice(LETTERS) for i in range(NUMBER_OF_EXPEREMENTS)),''.join(random.choice(LETTERS) for i in range(NUMBER_OF_EXPEREMENTS)), True)
fig, axes = plt.subplots()
plt.plot((np.arange(NUMBER_OF_EXPEREMENTS) * 0.00015 + 0.0001 ), label = 'Theoretical time')
plt.plot(times, label = 'Experimental time')
axes.set_title("Longest common subsequence dynamic solution")
axes.set_xlabel('n')
axes.set_ylabel('second')
plt.legend()
plt.show()