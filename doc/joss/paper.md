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
  - name: Yen-Chen Chen 
    affiliation: 4 
  - name: Goran Flegar
    orcid: 0000-0002-4154-0420
    affiliation: 3 
  - name: Fritz G{\"o}bel
    affiliation: 1 
  - name: Thomas Gr{\"u}tzmacher 
    affiliation: 1
  - name: Pratik Nayak 
    orcid: 0000-0002-7961-1159
    affiliation: 1 
  - name: Tobias Ribizel 
    affiliation: 1 
  - name: Yu-Hsiang Tsai 
    affiliation: 1 
affiliations:
 - name: Karlsruhe Institute of Technology
   index: 1
 - name: Innovative Computing Laboratory, University of Tennessee, Knoxville 
   index: 2
 - name: University of Jaume I 
   index: 3
 - name: University of Tokyo 
   index: 4
date: 6th March, 2020.
bibliography: paper.bib

---

# Summary

The Ginkgo library provides a clean and user-friendly high-level
interface for high-performance sparse linear algebra on single-node manycore and
heterogeneous architectures with a focus on flexibility and software sustainability.

The library focuses on solving sparse linear systems and accommodates a large variety
of matrix formats, state-of-the-art iterative (Krylov) solvers and preconditioners, 
which make the library suitable for a variety of scientific applications. Ginkgo
supports many architectures such as multi-threaded CPU, NVIDIA GPU's and AMD GPU's.
With the help of modern C++ 11 features such as dynamic polymorphism and lambdas, 
addition of new executor paradigms and algorithms is made easy, while the zero-cost
abstraction makes sure that the performance is unaffected by these additions. 

Ginkgo is also a part of the xSDK effort [@xsdk] which aims to provide infrastructure
for and interoperability of a collection of related and complementary software 
elements to foster rapid and efficient development of scientific applications
using High Performance Computing. Within this effort, we support interoperability with
application libraries such as `deal.ii`[@dealii] and `mfem`[@mfem]. Ginkgo provides
wrappers within these two libraries so that they can take advantage of the features of
Ginkgo. 

# Features

As sparse linear algebra is one of the main focus of Ginkgo, we provide a variety of 
sparse matrix formats such as COO, CSR, ELL, HYBRID and SELLP and provide highly tuned
Sparse Matrix Vector product (SpMV) kernels. The SpMV forms one of the key building 
blocks in an iterative solver and most of the time in an iteration is spent in this 
kernel. By providing highly tuned kernels, we can make sure that the iterative solvers
and preconditioners that use these SpMV kernels are highly performant as well. 
Additionally, we also provide various algorithms for the kernels within the SpMV
formats allowing the user to select the best choice for their applications. We also
provide conversion routines between the different formats enhancing their flexibility.

Ginkgo provides multiple iterative solution methods such as the well known Krylov
methods: Conjugate gradient (CG), Flexible Conjugate Gradient (FCG), Bi-Conjugate
Gradient (BiCG) and its stabilized version (Bi-CGSTAB), Generalized Minimal
residual method (GMRES) and more generic methods such as Iterative Refinement,
which forms the basis of many relaxation methods. It also has support for 
direct triangular solvers.

Ginkgo provides support for state of the art preconditioners such as the Block Jacobi
preconditioner with support for a version which uses adaptive precision to
automatically reduce the precision of the blocks based on their condition numbers
[@adaptive-bj] which has been shown to be very efficient for problems with a block 
structure; the ParILUT preconditioner, a generic preconditioner which is
applicable to more generic problems [@parilut].


# Software extensibility and sustainability.

Ginkgo is extensible in terms of linear algebra solvers, preconditioners and
matrix formats. With modern C++ (C++11 standard), various language features
such as data abstraction, generic programming and automatic memory management are
leveraged to enhance the performance of the library while still maintaining ease of 
use and maintenance. 

The Ginkgo library is constructed around two principal design concepts. The first
one is the class and object-oriented design which aims to provide an easy to use 
interface, common for all the devices; and the second part consists of the low level
device specific kernels. These low level kernels are optimized for the specific device
and make use of C++ features such as templates to generate high-performance 
kernels for a wide variety of parameters. 

Ginkgo adopts a rigorous approach to testing and maintenance. Using continuous
integration tools such as Gitlab-CI and Github-Actions, we make sure that 
the library builds on a range of compilers and environments. To verify that our code
quality is of the highest standards we use the Sonarcloud platform [@sonarcloud],
a code analyzer to analyze and report any code smells. To ensure correctness of
the code, we use the Google Test library [@gtest] to rigorously test each of the
device specific kernels and the core framework of the library. 


# Performance and Benchmarking

Ginkgo provides comprehensive logging facilities both in-house and with interfaces
 to external libraries such as PAPI [@papi]. This allows for detailed analysis of
 the kernels while reducing the intellectual overhead of optimizing the applications.

To enhance reproducibility from a performance perspective, we provide the performance 
results of our kernel implementations in an open source git repository [@gko-data].

A unique feature of Ginkgo is the availability of an interactive webtool Ginkgo
Performance explorer [@gpe], which can plot results from the aforementioned data 
repository. Additionally, we have also put in some effort in making benchmarking
easier, within the Ginkgo repository using the `rapidjson` [@rapidjson] and 
`gflags` [@gflags] libraries to run and generate benchmarking results for a variety 
of Ginkgo features.

# Acknowledgements
Research and software development in Ginkgo received support from the Helmholtz 
association (Impuls und Vernetzungsfond VH-NG-1241), and the US Exascale Computing 
Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office 
of Science and the National Nuclear Security Administration.

# References
