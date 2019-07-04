---
layout: post
title:  "Molecular Generation using Junction Tree VAE using PyTorch"
date: 2019-06-27
comments: True
mathjax: True
---

We attempt to automate the design of molecules.This task involves continuous embedding and generation of molecular graphs.

We generate the molecular graphs.Our junction tree variational autoencoder generates molecular graphs in two phases:
i)First generating a tree-structured scaffold over chemical substructures
ii)Combining them into a molecule with a graph message passing network.
