---
layout: post
title:  "Molecular Generation using Junction Tree VAE using PyTorch"
date: 2019-06-27
comments: True
mathjax: True
---

<h2>Introduction</h2>
We attempt to automate the design of molecules.This task involves continuous embedding and generation of molecular graphs.

We generate the molecular graphs.Our junction tree variational autoencoder generates molecular graphs in two phases:<br>
i)First generating a tree-structured scaffold over chemical substructures<br>
ii)Combining them into a molecule with a graph message passing network.

<h2>Overview</h2>
A molecular graph G is first decomposed into its junction tree TG, where each coloured node in the tree represents a substructure in the 
molecule.We then encode both the tree and graph into their latent embeddings zT and zG. To decode the molecule, we first reconstruct junction
tree from zT , and then assemble nodes in the tree back to the original molecule.
<center>{%include image.html url="\assets\img\jvae_1.png" %}</center>
<h2>Implementation</h2>
I'll be showing you how I built my Junction tree VAE in Pytorch. The dataset I used is ZINC dataset.The dataset contains smiles representation 
of molecules.

<h3>I.Data Preprocessing</h3>
(i)Import the text file into our code.

```python
with open('train1.txt') as f:
    data = [line.strip("\r\n ").split()[0] for line in f]
```

(ii)Convert each molecule to a Molecular tree.First we have to decompose the molecule to a tree.

<h4>Tree Decomposition of Molecules:</h4>
A tree decomposition maps a graph G into a junction tree by contracting certain vertices into a single node so that G becomes cycle-free.
Formally, given a graph G, a junction tree TG = (V, E, X ) is a connected labeled tree whose node set is V = {C1, · · · , Cn} and edge set 
is E.Here X is vocabulary contains only cycles (rings) and single edges.

We first find simple cycles of given graph G, and its edges not belonging to any cycles.

Two simple rings are merged together if they have more than two overlapping atoms. Each of those cycles or edges is considered as a 
cluster(clique).

Here cliques are nothing but the clusters.We will check the length of the clique and if it is more than 2 then we will check for the set of
intersection atoms in the neighbourhood list of the cluster and if intersection atom list is more than 2 we will merge them.

```python
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []
    
    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
```

Next, a cluster graph is constructed by adding edges between all intersecting clusters. Finally, we select one of its spanning trees as the 
junction tree of G.

Here csr_matrix is creating sparse matrix with given number of rows and columns and minumum_spanning tree is an inbuilt from scipy module to 
get the minimum spanning tree.

```python
    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
```

Now,after collecting cliques and edges from tree decomposition,we construct molecular tree using those cliques and edges.

<h3>II.Defining the model</h3>
Here we define our model as JTNNVAE.

```python
class JTNNVAE(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        #print(int(vocab.size()))
        self.hidden_size = int(hidden_size)
        self.latent_size = latent_size = latent_size / 2 #Tree and Mol has two vectors
        self.latent_size=int(self.latent_size)
        self.jtnn = JTNNEncoder(int(hidden_size),int(depthT), nn.Embedding(780,450))
        self.decoder = JTNNDecoder(vocab, int(hidden_size), int(latent_size), nn.Embedding(780,450))

        self.jtmpn = JTMPN(int(hidden_size), int(depthG))
        self.mpn = MPN(int(hidden_size), int(depthG))

        self.A_assm = nn.Linear(int(latent_size), int(hidden_size), bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(int(hidden_size), int(latent_size))
        self.T_var = nn.Linear(int(hidden_size), int(latent_size))
        self.G_mean = nn.Linear(int(hidden_size), int(latent_size))
        self.G_var = nn.Linear(int(hidden_size), int(latent_size))        
```
