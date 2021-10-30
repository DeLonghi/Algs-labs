import numpy as np
import time;
import matplotlib.pyplot as plt
import math

N = 10000
MIN_MERGE = 32

def getVector(size):
    return np.random.rand(size)

def const(arr):
    return ((7**3 + 11) / 2 + 1) % 3
    # return 7**3

def sum(arr):
    arr_size = len(arr)
    sum = 0
    for i in range(arr_size):
        sum += arr[i]
    return sum

def prod(arr):
    arr_size = len(arr)
    prod = 1
    for i in range(arr_size):
        prod *= arr[i]
    return prod

def poly_naive(poly):
    poly_size = len(poly)
    res = 0
    for i in range(poly_size):
        res += poly[i] * 1.5**i
    return res

def horner(poly):
    poly_size = len(poly)
    result = poly[-1] 
    for i in reversed(range(poly_size - 1)):
        result = result*1.5 + poly[i]
    return result

def bubbleSort(arr):
    n = len(arr) - 1 
    for i in range(n):
        for j in range(0, n-i):
            if arr[j] > arr[j + 1] :
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return


def quicksort(arr):
    np.sort(arr, kind='quicksort')

def calcMinRun(n):

    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r
 
def insertionSort(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1
 
 
def merge(arr, l, m, r):
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(arr[l + i])
    for i in range(0, len2):
        right.append(arr[m + 1 + i])
 
    i, j, k = 0, 0, l

    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
 
        else:
            arr[k] = right[j]
            j += 1
 
        k += 1
 
    while i < len1:
        arr[k] = left[i]
        k += 1
        i += 1

    while j < len2:
        arr[k] = right[j]
        k += 1
        j += 1
 
def timSort(arr):
    n = len(arr)
    minRun = calcMinRun(n)

    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(arr, start, end)
    size = minRun
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
            if mid < right:
                merge(arr, left, mid, right)
        size = 2 * size



N_pow_1 = []
N_pow_2 = []
N_log_N = []
N_const = []



y_const = []
y_sum = []
y_poly_naive = []
y_horner = []
y_bubble_sort = []
y_quicksort_sort = []
y_timsort_sort = []


def get_timestamp(func, arr):
    t = time.time()
    func(arr)
    return time.time() - t

def init():
    for n in range(1, N):
        const_total = 0
        sum_total = 0
        poly_naive_total = 0
        horner_total = 0
        bubble_sort_total = 0
        quicksort_sort_total = 0
        timsort_sort_total = 0
        for i in range(5):
            V = getVector(n)

            v = np.copy(V)
            horner_total += get_timestamp(horner, v)


        N_pow_1.append(0.0000007 * n)
        N_pow_2.append(0.0000005 * n**2)
        N_log_N.append(0.000001 * n * math.log2(n))
        N_const.append(0.00001)

        y_const.append(const_total / 5)
        y_sum.append(sum_total / 5)
        y_poly_naive.append(poly_naive_total / 5)
        y_horner.append(horner_total / 5)
        y_bubble_sort.append(bubble_sort_total / 5)
        y_quicksort_sort.append(quicksort_sort_total / 5)
        y_timsort_sort.append(timsort_sort_total / 5)

init()


fig, axes = plt.subplots()

plt.plot(N_pow_1, label = 'Theoretical time')
plt.plot(y_horner, label = 'Experimental time')
axes.set_title("Horner method")
axes.set_xlabel('n')
axes.set_ylabel('seconds')
plt.legend()

plt.show()

# v = getVector(N)
# # print(v)
# print(poly_naive(v, N))
# print(horner(v, N))
# bubbleSort(v, N)
# print(v)
# v = getVector(N)
# quicksort(v)
# print(v)
# v = getVector(N)
# timSort(v, N)
# print(v)