
from suffix_tree import SuffixTree
from tqdm import tqdm
class StringUtils:

    def build_kmers(sequence, ksize):
        kmers = []
        n_kmers = len(sequence) - ksize + 1

        for i in range(n_kmers):
            kmer = sequence[i:i + ksize]
            kmers.append(kmer)

        return kmers
      

  
def mismatch_kernel(X_seq,k,m):
    """Compute the mismatch kernel for a given dataset
    """
    tree = SuffixTree(X_seq.size)
    for idx, DNA in tqdm(enumerate(X_seq)):
        kmers = StringUtils.build_kmers(DNA, k)
        for km in kmers:
            tree.insert_mismatch(km, idx, m=m)
    
    K = tree.compute_kernel()
    
    return K
    
    
    