#%%

"""
Graph Mining - ALTEGRAD - Dec 2020
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
##################

data_path = '/home/mahdo/Desktop/ALTEGRAD_LAB#4/code/datasets/CA-HepTh.txt'

G = nx.read_edgelist(data_path,comments='#', delimiter='\t')


print(len(G.nodes())) ## Number of nodes

print(sum([G.degree(node) for node in G.nodes])/2) ### Sum of degrees 




############## Task 2

##################
# your code here #
##################

n_connected = nx.connected_components(G)


L = [1 for c in sorted(nx.connected_components(G), key=len, reverse=True)]

n_connected = sum(L)

print("Number of connected components",n_connected)

if n_connected > 1:
    
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

else :
    
    largest_cc = G

print("Number of nodes gcc ", largest_cc.number_of_nodes())
print("Number of edges gcc",largest_cc.number_of_edges())

############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################

print("G max degree",np.max(degree_sequence))      
print("G min degree",np.min(degree_sequence))
print("G mean degree",np.mean(degree_sequence))
print("G median degree",np.median(degree_sequence))

############## Task 4

##################
# your code here #
##################

degree_hist = nx.degree_histogram(G)

plt.plot(degree_hist)
plt.show()

ax = plt.gca()
plt.plot(degree_hist)

plt.title('Log scale')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

############## Task 5

##################
# your code here #
##################
print("The graph transitivity aka (Global clustering coeff) is ",nx.transitivity(G))
# %%

# %%
