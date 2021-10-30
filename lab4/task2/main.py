from anneal import SimAnneal
import matplotlib.pyplot as plt
import random


def read_coords(path):
    coords = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [float(x.replace("\n", "")) for x in line.split(" ")]
            coords.append(line)
    return coords


def generate_random_coords(num_nodes):
    return [[random.uniform(-1000, 1000), random.uniform(-1000, 1000)] for i in range(num_nodes)]


if __name__ == "__main__":
    print('HERE')
    coords = read_coords("D:\Algs labs\lab4\\task2\coord.txt")  # generate_random_coords(100)
    sa = SimAnneal(coords, stopping_iter=5000)
    sa.initial_solution()
    sa.visualize_routes()
    sa.anneal()
    sa.visualize_routes()
    sa.plot_learning()