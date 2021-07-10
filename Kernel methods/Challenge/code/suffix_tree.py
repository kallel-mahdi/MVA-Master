import numpy as np

class Node:
    def __init__(self):
        
        self.children : Dict[str, Node] = {}  ### mapping from char to node
        self.value = None ### only leave nodes require a value
        self.visited = False ### used to define leave nodes later

  
class SuffixTree:

    def __init__(self,n_seq):

      self.root = Node()
      self.n_seq = n_seq

    def insert(self,key: str,idx,node=None):
        """Add key to tree"""
        
        if node == None : node = self.root

        for char in key:
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]
        ### When out of the loop, "node" is a leaf node
        ### only leaf nodes need a value
        if not node.visited : 
          node.visited = True
          node.value = np.zeros(self.n_seq)   
 
        node.value[idx] +=1


    def insert_mismatch(self,key,idx,node=None,m=1):

      """ inserts a key and it's m, mismatches into the tree"""

      if node == None : node = self.root
      ALPHABET = ['A','C','G','T'] 
      
      if m ==0 or key == '' :
        self.insert(key,idx,node)
        return   
        
      else :

        good_char = key[0]
        
        for char in ALPHABET :

            if char not in node.children:

                node.children[char]= Node()

            if char != good_char : 
              
              self.insert_mismatch(key[1:],idx,node.children[char],m-1)

            else :
              
              self.insert_mismatch(key[1:],idx,node.children[good_char],m)

    def get_leafs(self):

        """ look for leaf nodes using DFS"""

        
        to_visit = [self.root]
        leafs = []

        while len(to_visit)>0:
          node = to_visit.pop(0)
          if node.children == {}:
            leafs.append(node)

          else :
            for char,children in node.children.items():
                to_visit.append(children)

 
        return leafs
    
    def compute_kernel(self):

      
      leafs = self.get_leafs()
      node_values = np.array([leaf.value for leaf in leafs])
      K = node_values.T @ node_values
      """ return normalized K """
      D = np.diag(np.power(np.diag(K),-0.5))
      return D@K@D

    def find(self, key: str):
      """Count occurences of suffix in tree """
      node = self.root
      for char in key:
          if char in node.children:
              node = node.children[char]
          else:
              return None

      print(f'Suffix:{key},count:{node.value}')
      return node.value
    
    def __str__(self,node=None,key=''):
      
      """ print the tree leafs"""

      if node == None : node = self.root

      if node.children == {}: 
        print(f'Suffix:{key},count:{node.value}')
        return

      else :
        for char,child in  node.children.items():
          self.__str__(child,key+char)

