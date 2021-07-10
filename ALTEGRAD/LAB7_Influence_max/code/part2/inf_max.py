"""
Influence Maximization - ALTEGRAD - Jan 2021
"""

# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import load_data

def independent_cascade(G, S, p, mc):
    
    # Loops over the Monte-Carlo Simulations
    spread = list()
    for i in range(mc):
        # Simulates propagation process      
        new_active = S[:]
        A = S[:]
        while new_active:

            # For each newly active node, finds its neighbors that become activated
            new_ones = list()
            for node in new_active:
                # Determines neighbors that become infected
                neighbors = list(G.neighbors(node))
                success = np.random.uniform(0,1,len(neighbors)) < p
                new_ones += list(np.extract(success, neighbors))

            new_active = list(set(new_ones) - set(A))
            
            # Adds newly activated nodes to the set of activated nodes
            A += new_active
        
        # Appends the number of activated nodes
        spread.append(len(A))
        
    return np.mean(spread)


def greedy_algorithm(G, k, p, mc):

    S = list()
    spread = list()
    
    ############## Task 8
    
    ##################
    # your code here #
    S = list()
    spread = list()
    
    ############## Task 8

    
    ##################
    maks = 0
    for n in range(k):
        for node in list(G.nodes() - set(S)):
            sp = independent_cascade(G, S+[node], p,mc)
            if sp>maks:
                seed = node
                best = sp
        S.append(seed)
        spread.append(best)

    ##################

    return S, spread
    
            
    


# Loads network
G = load_data()

print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

############## Task 9
    
    
##################
used,spread = greedy_algorithm(G,5,0.1,100)
##################

# Visualizes the spread with respect to the size of the set
plt.plot(range(1,len(spread)+1), spread)
plt.xlabel('size of set')
plt.ylabel('expected spread')
plt.show()


# %%
