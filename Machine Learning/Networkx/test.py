import networkx as nx
G = nx.Graph()
G.add_node('a')
e = [('a', 'b', 0.3), ('b', 'c', 0.9), ('a', 'c', 0.5), ('c', 'd', 1.2)]
G.add_weighted_edges_from(e)
G.add_node('1')
print(nx.dijkstra_path(G, 'a', 'd'))

# import pyvis
from pyvis.network import Network
net = Network(notebook=False, width=1000, height=600)
net.from_nx(G)
net.show("example.html")
