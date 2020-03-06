---
title: 'Ginkgo: A high performance numerical linear algebra library'
tags:
  - linear-algebra
  - hpc
  - cuda
  - modern-c++
  - hip
  - spmv
authors:
  - name: Hartwig Anzt
    orcid: 0000-0003-2177-952X
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Terry Cojean
    orcid: 0000-0002-1560-921X
    affiliation: 1 
  - name: Thomas Gr\"{u}tzmacher 
    affiliation: 1 
  - name: Pratik Nayak 
    orcid: 0000-0002-7961-1159
    affiliation: 1 
  - name: Tobias Ribizel 
    affiliation: 1 
  - name: Yu-hsiang Tsai 
    affiliation: 1 
affiliations:
 - name: Karlsruhe Institute of Technology
   index: 1
 - name: Innovative Computing Laboratory, University of Tennessee, Knoxville 
   index: 2
date: 6th March, 2020.
bibliography: paper.bib

---

# Summary

The Ginkgo library provides a clean and user-friendly high-level
interface for high-performance sparse linear algebra on single-node manycore and
heterogeneous architectures with a focus on flexibility and software sustainability.
The library focuses on solving sparse linear systems and accommodates a large variety of
matrix formats, state-of-the-art iterative (Krylov) solvers and preconditioners, 
which make the library suitable for a variety of different scientific applications. 

# Software extensibility and sustainability.

Ginkgo is also extensible in terms of linear algebra solvers, preconditioners and
matrix formats. Ginkgo is flexible in terms of computational technologies and it
currently features CPU and GPU execution with support for OpenMP backend on the CPU 
and both CUDA and HIP backends on NVIDIA and AMD GPU's. The programming language
of choice is C++ (C++11 standard) as it enables high performance while also
providing various language features (data abstraction, generic programming and
automatic memory management) which facilitate the use and the maintenance of the
library. 

The Ginkgo library is constructed around two principal design concepts. The first
one is the class and object-oriented design which aims to provide an easy to use 
interface, common for all the devices; and the second part consists of the low level
device specific kernels. These low level kernels are optimized for the specific device
and makes use of C++ features such as templates to generate high-performance specialized
kernels for a wide variety of parameters.


# Performance and Benchmarking

A unique feature of Ginkgo is the availability of the performance data from various high performance
systems on a git repository `[@gko-data]` and its corresponding visualization on an interactive
webpage `[@gpe]`. We have also put in some effort in making benchmarking easier, within the 
Ginkgo repository using the `rapidjson` `[@rapidjson]` and  `gflags` `[@gflags]` libraries. 

Various papers on the performance Ginkgo have been published.`[@gko:spmv; ]`


# Acknowledgements


# References
