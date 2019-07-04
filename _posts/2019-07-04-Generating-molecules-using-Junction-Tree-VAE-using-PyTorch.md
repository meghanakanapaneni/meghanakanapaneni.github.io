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
