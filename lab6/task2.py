import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

M = 10
N = 20
OBSTACLES = 40
NUMBER_OF_EXPERIMETNS = 5

g = nx.grid_2d_graph(M, N)
nodes = g.nodes
a = list(nodes)
np.random.shuffle(a)
print(a[:OBSTACLES])
g.remove_nodes_from(a[:OBSTACLES])
labels = {(x,y):(x + y * 10) for x,y in g.nodes()}
plt.figure(0)
nx.draw_networkx(g, pos={(x,y):(x,-y) for x,y in g.nodes()}, labels=labels)


for i in range(NUMBER_OF_EXPERIMETNS):
    plt.subplots()
    nx.draw_networkx(g, pos={(x,y):(x,-y) for x,y in g.nodes()}, labels=labels)
    n1 = random.choice(a[OBSTACLES:])
    n2 = random.choice(a[OBSTACLES:])

    # print(n1)
    # print(n2)

    path = nx.algorithms.shortest_paths.astar.astar_path(g, n1, n2)
    pos = {(x,y):(x,-y) for x,y in g.nodes()}
    nx.draw_networkx_nodes(g, pos=pos, nodelist=path,  node_color="red")
    # print(path)

plt.show()
