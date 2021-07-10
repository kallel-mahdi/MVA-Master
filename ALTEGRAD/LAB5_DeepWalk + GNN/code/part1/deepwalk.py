"""
Deep Learning on Graphs - ALTEGRAD - Dec 2020
"""

import numpy as np
import networkx as nx
from random import randint, choice
from gensim.models import Word2Vec


#G = nx.read_edgelist('../data/karate.edgelist',delimiter=' ',nodetype=int)

############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):
    
    walk = [node]
    
    ##################
    for i in range(walk_length):
        
        neighbours = list(G.neighbors(walk[-1]))
        next_step  = choice(neighbours)
        walk.append(next_step)

    ##################
    
    walk = [str(node) for node in walk]
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    
    ##################
    for i in range(num_walks):
        permuted_nodes = np.random.permutation(G.nodes())
        
        for node in permuted_nodes:
            
            walks.append(random_walk(G,node,walk_length))
            
    
        
    ##################
    
    return walks

# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model