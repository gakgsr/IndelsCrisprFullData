import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()

G.add_edge(1,2,w=0.5, c = 'b')
G.add_edge(1,3,w=9.8, c = 'g')
edges = G.edges()
colors = [G[u][v]['c'] for u, v in edges]
weights = [G[u][v]['w'] for u, v in edges]
nx.draw(G, nx.circular_layout(G), font_size=10, node_color='y', with_labels=True, edge_color=colors)

nodes = G.nodes()
print nodes
# pos = nx.get_node_attributes(G, 'pos')
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.savefig("test.pdf")
plt.clf()

# G=nx.Graph()
# i=1
# G.add_node(i,pos=(i,i))
# G.add_node(2,pos=(2,2))
# G.add_node(3,pos=(1,0))
# G.add_edge(1,2,weight=0.5)
# G.add_edge(1,3,weight=9.8)
# pos=nx.get_node_attributes(G,'pos')
# nx.draw(G,pos)
# labels = nx.get_edge_attributes(G,'weight')
# print pos
# print labels
# nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
# plt.savefig("test.pdf")