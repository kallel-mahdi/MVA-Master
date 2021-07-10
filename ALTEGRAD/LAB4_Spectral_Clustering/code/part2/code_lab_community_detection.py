"""
Graph Mining - ALTEGRAD - Dec 2020
"""

import networkx as nx
import numpy as np
import scipy
from scipy.sparse.linalg import eigs
from scipy.sparse import diags
from random import randint
from sklearn.cluster import KMeans





############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    # your code here #
    ##################
    
    A = nx.linalg.graphmatrix.adjacency_matrix(G)
    n, m = A.shape
    
    D_inv = diags([1 / G.degree(node) for node in G.nodes()])
    
    
    Lrw = scipy.sparse.identity(n) - D_inv @ A
    
    eig_values,eig_vectors = eigs(Lrw,k=k,which='SR')
    
    eig_vectors = eig_vectors.real
    
    kmeans = KMeans(n_clusters=k)
    
    kmeans.fit(eig_vectors)
    
    clustering = {}
    
    for i,node in enumerate(G.nodes()):
        
        clustering[node] = kmeans.labels_[i]
    
    return clustering
    
    




############## Task 7

##################
# your code here #
##################

data_path = '/home/mahdo/Desktop/ALTEGRAD_LAB#4/code/datasets/CA-HepTh.txt'

G = nx.read_edgelist(data_path,comments='#', delimiter='\t')

largest_cc = max(nx.connected_components(G),key=len)

GCC = G.subgraph(largest_cc)

clustering = spectral_clustering(GCC,50)

#print(clustering)



############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    
    m = G.number_of_edges()
    
    clusters = set(clustering.values())
    
    modularity = 0
    
    for cluster in clusters:
        
        nodes_in_cluster = [node for node in G.nodes() if clustering[node]==cluster]
        
        subG = G.subgraph(nodes_in_cluster)
        
        lc = subG.number_of_edges()
        
        dc = 0 
        
        for node in nodes_in_cluster:
            
            dc+= G.degree(node)
    
        modularity += lc /m  - (dc / (2*m))**2
        
        
        
        
    
    return modularity



############## Task 9

##################
# your code here #
##################

print(modularity(GCC,clustering))

random_clustering = {}

for node in GCC.nodes():
    
    random_clustering[node] = np.random.randint(0,49)

print(modularity(GCC,random_clustering))