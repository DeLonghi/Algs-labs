import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
NUMBER_OF_EXPEREMENTS = 200


def maxCrossingSum(arr, l, m, h):
    sm = 0
    left_sum = -10000

    for i in range(m, l-1, -1):
        sm = sm + arr[i]

        if (sm > left_sum):
            left_sum = sm

    sm = 0
    right_sum = -1000
    for i in range(m + 1, h + 1):
        sm = sm + arr[i]

        if (sm > right_sum):
            right_sum = sm

    return max(left_sum + right_sum, left_sum, right_sum)


def maxSubArraySum(arr, l, h):
    if (l == h):
        return arr[l]

    m = (l + h) // 2

    return max(maxSubArraySum(arr, l, m),
               maxSubArraySum(arr, m+1, h),
               maxCrossingSum(arr, l, m, h))


# Driver Code

def measure_time(n_runs, arr):
    times = []
    for i in range(n_runs):
        times.append([])
        for j in range(1, NUMBER_OF_EXPEREMENTS):
            temp_arr = arr[:j + 1]
            start_time = timer()
            maxSubArraySum(temp_arr, 0, j)
            times[i].append(timer() - start_time)

    return np.array(times, dtype=np.float64).mean(axis=0)


r = np.arange(NUMBER_OF_EXPEREMENTS)
times = measure_time(5, r)

fig, axes = plt.subplots()
plt.plot(r * 0.000023 + 0.000045, label = 'Theoretical time')
plt.plot(times, label='Experimental time')
axes.set_title("Max subarray using divide and conquer algorithm")
axes.set_xlabel('n')
axes.set_ylabel('second')
plt.legend()
plt.show()