import numpy as np
from numpy.core.fromnumeric import size
import networkx as nx
import matplotlib.pyplot as plt
from timeit import default_timer as timer

V = 100
E = 250
NUMBER_OF_EXPEREMETNS = 10

il1 = np.triu_indices(V, 1)
a = np.zeros(V ** 2).reshape(V, V)

edges = np.append(np.random.randint(1, 100, size=E),
                  np.zeros(int((V * V - V) / 2 - E)))
np.random.shuffle(edges)

a[il1] = edges

a = a + a.T

g = nx.Graph(a)

times = [[],[]]
for i in range(NUMBER_OF_EXPEREMETNS):
    t1 = timer()
    nx.dijkstra_predecessor_and_distance(g, 15)
    times[0].append(timer() - t1)

    t1 = timer()
    nx.single_source_bellman_ford_path_length(g, 15)
    times[1].append(timer() - t1)

print(np.array(times[0]).mean())
print(np.array(times[1]).mean())


# pos = nx.spring_layout(g)
# comp = nx.connected_components(g)
# nx.draw(g, pos, with_labels=True)
# labels = nx.get_edge_attributes(g,'weight')
# nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
plt.show()
