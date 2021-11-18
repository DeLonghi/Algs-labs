import numpy as np
from numpy.core.fromnumeric import size
import networkx as nx
import matplotlib.pyplot as plt

V = 100
E = 100

color_list = ["lightcoral", "gray", "lightgray", "firebrick", "red", "chocolate", "darkorange", "moccasin", "gold", "yellow", "darkolivegreen", "chartreuse", "forestgreen", "lime", "mediumaquamarine", "turquoise", "teal", "cadetblue", "dogerblue", "blue", "slateblue", "blueviolet", "magenta", "lightsteelblue"]
def convert_to_adjacency(matrix):
    start = 0
    res = []
    lst = []
    n = len(matrix)

    for i in range(n):
        res.append(lst*n)
    while start < n:
        y = matrix[start]
        for i in range(len(y)):
            if y[i] == 1:
                res[start].append(i)
        start += 1
    return res

il1 = np.triu_indices(V,1)
a = np.zeros(V ** 2).reshape(V, V)

edges = np.append(np.full(E, 1), np.zeros(int((V * V -  V) / 2 - E)))
np.random.shuffle(edges)

a[il1] = edges

a = a + a.T

print(a[0:3])

adj = convert_to_adjacency(a)
print(adj[0:3])

plt.figure(0)

g = nx.Graph(a)
pos = nx.spring_layout(g)
comp = nx.connected_components(g)
lol = 1
nx.draw(g, pos, with_labels=True)


plt.figure(1)
nx.draw(g, pos, with_labels=True)
i=0
for c in sorted(nx.connected_components(g), key=len, reverse=True):
    nx.draw_networkx_nodes(g,pos, nodelist=c,  node_color=color_list[i], label=1)
    nx.draw_networkx_labels(g, pos, {n: n for n in c})
    lol = c
    i = i + 1
nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)

plt.figure(2)

p = nx.single_source_shortest_path(g, 15)[49]
path = []
for i in range(len(p) - 1):
    path.append((p[i], p[i + 1]))
print(path)
nx.draw(g, pos, with_labels=True, width=0.1)
nx.draw_networkx_nodes(g,pos, nodelist=p,  node_color="red", label=1)
nx.draw_networkx_labels(g, pos, {n: n for n in p})
nx.draw_networkx_edges(g, pos, edgelist=path, arrows=True, width=2.0, alpha=0.5)

plt.show()