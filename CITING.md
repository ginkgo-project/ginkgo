# Citing Ginkgo                                           {#citing_ginkgo}

The main Ginkgo interface design, performance paper is
``` bibtex
@article{ginkgo-toms-2022,
title = {{Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing}},
volume = {48},
copyright = {All rights reserved},
issn = {0098-3500},
shorttitle = {Ginkgo},
url = {https://doi.org/10.1145/3480935},
doi = {10.1145/3480935},
number = {1},
urldate = {2022-02-17},
journal = {ACM Transactions on Mathematical Software},
author = {Anzt, Hartwig and Cojean, Terry and Flegar, Goran and Göbel, Fritz and Grützmacher, Thomas and Nayak, Pratik and Ribizel, Tobias and Tsai, Yuhsiang Mike and Quintana-Ortí, Enrique S.},
month = feb,
year = {2022},
keywords = {ginkgo, healthy software lifecycle, High performance computing, multi-core and manycore architectures},
pages = {2:1--2:33}
}
```

and the software itself has been reviewed and published in Journal of Open Source Software (JOSS).

```bibtex

@article{ginkgo-joss-2020,
  doi = {10.21105/joss.02260},
  url = {https://doi.org/10.21105/joss.02260},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {52},
  pages = {2260},
  author = {Hartwig Anzt and Terry Cojean and Yen-Chen Chen and Goran Flegar and Fritz G\"{o}bel and Thomas Gr\"{u}tzmacher and Pratik Nayak and Tobias Ribizel and Yu-Hsiang Tsai},
  title = {Ginkgo: A high performance numerical linear algebra library},
  journal = {Journal of Open Source Software}

```

For topical publications, please cite whichever features you use within Ginkgo:

**[Portability](#on-portability)** |
**[Software Sustainability](#on-software-sustainability)** |
**[SpMV performance](#spmv-performance)** |
**[Preconditioners](#preconditioners)** |
**[Mixed-precision](#mixed-precision-algorithms)** |
**[Sparse direct](#sparse-factorizations-and-direct-solvers)** |
**[Batched](#batched-algorithms)**


### On Portability

``` bibtex
@inproceedings{tsaiPreparingGinkgoAMD2021,
  title = {Preparing {{Ginkgo}} for {{AMD GPUs}} {\textendash} {{A Testimonial}} on {{Porting CUDA Code}} to {{HIP}}},
  booktitle = {Euro-{{Par}} 2020: {{Parallel Processing Workshops}}},
  author = {Tsai, Yuhsiang M. and Cojean, Terry and Ribizel, Tobias and Anzt, Hartwig},
  editor = {Balis, Bartosz and B. Heras, Dora and Antonelli, Laura and Bracciali, Andrea and Gruber, Thomas and {Hyun-Wook}, Jin and Kuhn, Michael and Scott, Stephen L. and Unat, Didem and Wyrzykowski, Roman},
  year = {2021},
  series = {Lecture {{Notes}} in {{Computer Science}}},
  pages = {109--121},
  publisher = {{Springer International Publishing}},
  address = {{Cham}},
  doi = {10.1007/978-3-030-71593-9_9},
  isbn = {978-3-030-71593-9},
  langid = {english},
  keywords = {CUDA,GPU,HIP,Portability}
}

@inproceedings{tsaiPortingSparseLinear2022,
  title = {Porting {{Sparse Linear Algebra}} to~{{Intel GPUs}}},
  booktitle = {Euro-{{Par}} 2021: {{Parallel Processing Workshops}}},
  author = {Tsai, Yuhsiang M. and Cojean, Terry and Anzt, Hartwig},
  editor = {Chaves, Ricardo and B. Heras, Dora and Ilic, Aleksandar and Unat, Didem and Badia, Rosa M. and Bracciali, Andrea and Diehl, Patrick and Dubey, Anshu and Sangyoon, Oh and L. Scott, Stephen and Ricci, Laura},
  year = {2022},
  series = {Lecture {{Notes}} in {{Computer Science}}},
  pages = {57--68},
  publisher = {{Springer International Publishing}},
  address = {{Cham}},
  doi = {10.1007/978-3-031-06156-1_5},
  isbn = {978-3-031-06156-1},
  langid = {english},
  keywords = {Ginkgo,Intel GPUs,Math library,oneAPI,SpMV}
}
```


### On Software Sustainability

``` bibtex
@inproceedings{anzt2019pasccb,
author = {Anzt, Hartwig and Chen, Yen-Chen and Cojean, Terry and Dongarra, Jack and Flegar, Goran and Nayak, Pratik and Quintana-Ort\'{\i}, Enrique S. and Tsai, Yuhsiang M. and Wang, Weichung},
title = {Towards Continuous Benchmarking: An Automated Performance Evaluation Framework for High Performance Software},
year = {2019},
isbn = {9781450367707},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3324989.3325719},
doi = {10.1145/3324989.3325719},
booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference},
articleno = {9},
numpages = {11},
keywords = {interactive performance visualization, healthy software lifecycle, continuous integration, automated performance benchmarking},
location = {Zurich, Switzerland},
series = {PASC ’19}
}
```


### SpMV performance

``` bibtex
@InProceedings{tsai2020amdspmv,
author="Tsai, Yuhsiang M.
and Cojean, Terry
and Anzt, Hartwig",
editor="Sadayappan, Ponnuswamy
and Chamberlain, Bradford L.
and Juckeland, Guido
and Ltaief, Hatem",
title="Sparse Linear Algebra on AMD and NVIDIA GPUs -- The Race Is On",
booktitle="High Performance Computing",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="309--327",
abstract="Efficiently processing sparse matrices is a central and performance-critical part of many scientific simulation codes. Recognizing the adoption of manycore accelerators in HPC, we evaluate in this paper the performance of the currently best sparse matrix-vector product (SpMV) implementations on high-end GPUs from AMD and NVIDIA. Specifically, we optimize SpMV kernels for the CSR, COO, ELL, and HYB format taking the hardware characteristics of the latest GPU technologies into account. We compare for 2,800 test matrices the performance of our kernels against AMD's hipSPARSE library and NVIDIA's cuSPARSE library, and ultimately assess how the GPU technologies from AMD and NVIDIA compare in terms of SpMV performance.",
isbn="978-3-030-50743-5"
}

@article{anzt2020spmv,
author = {Anzt, Hartwig and Cojean, Terry and Yen-Chen, Chen and Dongarra, Jack and Flegar, Goran and Nayak, Pratik and Tomov, Stanimire and Tsai, Yuhsiang M. and Wang, Weichung},
title = {Load-Balancing Sparse Matrix Vector Product Kernels on GPUs},
year = {2020},
issue_date = {March 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {7},
number = {1},
issn = {2329-4949},
url = {https://doi.org/10.1145/3380930},
doi = {10.1145/3380930},
journal = {ACM Trans. Parallel Comput.},
month = mar,
articleno = {2},
numpages = {26},
keywords = {irregular matrices, GPUs, Sparse Matrix Vector Product (SpMV)}
}

@inproceedings{anztEvaluatingPerformanceNVIDIA2020,
  title = {Evaluating the {{Performance}} of {{NVIDIA}}'s {{A100 Ampere GPU}} for {{Sparse}} and {{Batched Computations}}},
  booktitle = {2020 {{IEEE}}/{{ACM Performance Modeling}}, {{Benchmarking}} and {{Simulation}} of {{High Performance Computer Systems}} ({{PMBS}})},
  author = {Anzt, Hartwig and Tsai, Yuhsiang M. and Abdelfattah, Ahmad and Cojean, Terry and Dongarra, Jack},
  year = {2020},
  month = nov,
  pages = {26--38},
  doi = {10.1109/PMBS51919.2020.00009},
  urldate = {2023-12-06},
  keywords = {ginkgo}
}
```


### Preconditioners

```bibtex
@inproceedings{gobelMixedPrecisionIncomplete2021,
  title = {Mixed {{Precision Incomplete}} and {{Factorized Sparse Approximate Inverse Preconditioning}} on {{GPUs}}},
  booktitle = {Euro-{{Par}} 2021: {{Parallel Processing}}},
  author = {G{\"o}bel, Fritz and Gr{\"u}tzmacher, Thomas and Ribizel, Tobias and Anzt, Hartwig},
  editor = {Sousa, Leonel and Roma, Nuno and Tom{\'a}s, Pedro},
  year = {2021},
  series = {Lecture {{Notes}} in {{Computer Science}}},
  pages = {550--564},
  publisher = {{Springer International Publishing}},
  address = {{Cham}},
  doi = {10.1007/978-3-030-85665-6_34},
  isbn = {978-3-030-85665-6},
  langid = {english},
  keywords = {Factorized Sparse Approximate Inverse,GPUs,Incomplete Sparse Approximate Inverse,Mixed precision,Preconditioning}
}

@article{flegarAdaptivePrecisionBlockJacobi2021,
  title = {Adaptive {{Precision Block-Jacobi}} for {{High Performance Preconditioning}} in the {{Ginkgo Linear Algebra Software}}},
  author = {Flegar, Goran and Anzt, Hartwig and Cojean, Terry and {Quintana-Ort{\'i}}, Enrique S.},
  year = {2021},
  month = apr,
  journal = {ACM Transactions on Mathematical Software},
  volume = {47},
  number = {2},
  pages = {14:1--14:28},
  issn = {0098-3500},
  doi = {10.1145/3441850},
  urldate = {2023-08-08},
  keywords = {adaptive precision,block-Jacobi,GPU,Krylov solvers,preconditioning,Sparse linear algebra}
}
```


### Mixed-precision algorithms

```bibtex
@article{tsaiThreeprecisionAlgebraicMultigrid2023,
  title = {Three-Precision Algebraic Multigrid on {{GPUs}}},
  author = {Tsai, Yu-Hsiang Mike and Beams, Natalie and Anzt, Hartwig},
  year = {2023},
  month = dec,
  journal = {Future Generation Computer Systems},
  volume = {149},
  pages = {280--293},
  issn = {0167-739X},
  doi = {10.1016/j.future.2023.07.024},
  urldate = {2023-12-13},
  keywords = {Algebraic multigrid,GPUs,Mixed precision,Portability}
}

@article{aliagaCompressedBasisGMRES2022,
  title = {Compressed Basis {{GMRES}} on High-Performance Graphics Processing Units},
  author = {Aliaga, Jos{\'e} I and Anzt, Hartwig and Gr{\"u}tzmacher, Thomas and {Quintana-Ort{\'i}}, Enrique S and Tom{\'a}s, Andr{\'e}s E},
  year = {2022},
  month = aug,
  journal = {The International Journal of High Performance Computing Applications},
  pages = {10943420221115140},
  publisher = {{SAGE Publications Ltd STM}},
  issn = {1094-3420},
  doi = {10.1177/10943420221115140},
  urldate = {2023-08-06},
  langid = {english}
}

@inproceedings{aliagaBalancedCompressedCoordinate2021,
  title = {Balanced and {{Compressed Coordinate Layout}} for the {{Sparse Matrix-Vector Product}} on {{GPUs}}},
  booktitle = {Euro-{{Par}} 2020: {{Parallel Processing Workshops}}},
  author = {Aliaga, Jos{\'e} Ignacio and Anzt, Hartwig and {Quintana-Ort{\'i}}, Enrique S. and Tom{\'a}s, Andr{\'e}s E. and Tsai, Yuhsiang M.},
  editor = {Balis, Bartosz and B. Heras, Dora and Antonelli, Laura and Bracciali, Andrea and Gruber, Thomas and {Hyun-Wook}, Jin and Kuhn, Michael and Scott, Stephen L. and Unat, Didem and Wyrzykowski, Roman},
  year = {2021},
  series = {Lecture {{Notes}} in {{Computer Science}}},
  pages = {83--95},
  publisher = {{Springer International Publishing}},
  address = {{Cham}},
  doi = {10.1007/978-3-030-71593-9_7},
  isbn = {978-3-030-71593-9},
  langid = {english},
  keywords = {GPUs,High performance computing,Sparse linear algebra,Sparse matrix data layouts,Sparse matrix-vector product,spmv}
}

@article{grutzmacherUsingGinkgoMemory2023,
  title = {Using {{Ginkgo}}'s Memory Accessor for Improving the Accuracy of Memory-Bound Low Precision {{BLAS}}},
  author = {Gr{\"u}tzmacher, Thomas and Anzt, Hartwig and {Quintana-Ort{\'i}}, Enrique S.},
  year = {2023},
  journal = {Software: Practice and Experience},
  volume = {53},
  number = {1},
  pages = {81--98},
  issn = {1097-024X},
  doi = {10.1002/spe.3041},
  urldate = {2023-08-06},
  copyright = {{\textcopyright} 2021 The Authors. Software: Practice and Experience published by John Wiley \& Sons Ltd.},
  langid = {english},
  keywords = {accessor,floating-point formats,high performance,memory-bound algorithms,mixed precision,roofline model}
}
```


### Sparse factorizations and direct solvers

```bibtex
@inproceedings{ribizelParallelSymbolicCholesky2023,
  title = {Parallel {{Symbolic Cholesky Factorization}}},
  booktitle = {Proceedings of the {{SC}} '23 {{Workshops}} of {{The International Conference}} on {{High Performance Computing}}, {{Network}}, {{Storage}}, and {{Analysis}}},
  author = {Ribizel, Tobias and Anzt, Hartwig},
  year = {2023},
  month = nov,
  series = {{{SC-W}} '23},
  pages = {1721--1727},
  publisher = {{Association for Computing Machinery}},
  address = {{New York, NY, USA}},
  doi = {10.1145/3624062.3624253},
  urldate = {2023-11-12},
  isbn = {9798400707858},
  keywords = {elimination tree,GPGPU,sparse factorization,symbolic factorization}
}

@misc{swirydowiczGPUResidentSparseDirect2023,
  title = {{{GPU-Resident Sparse Direct Linear Solvers}} for {{Alternating Current Optimal Power Flow Analysis}}},
  author = {{\'S}wirydowicz, Kasia and Koukpaizan, Nicholson and Ribizel, Tobias and G{\"o}bel, Fritz and Abhyankar, Shrirang and Anzt, Hartwig and Pele{\v s}, Slaven},
  year = {2023},
  month = aug,
  number = {arXiv:2306.14337},
  eprint = {2306.14337},
  primaryclass = {cs},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2306.14337},
  urldate = {2023-11-08},
  archiveprefix = {arxiv},
  keywords = {{65F05, 65F10, 65F50, 65K10, 65Y05, 65Y10, 90C51},{Computer Science - Computational Engineering, Finance, and Science},ginkgo}
}
```


### Batched algorithms

```bibtex
@inproceedings{aggarwalBatchedSparseIterative2021,
  title = {Batched {{Sparse Iterative Solvers}} for {{Computational Chemistry Simulations}} on {{GPUs}}},
  booktitle = {2021 12th {{Workshop}} on {{Latest Advances}} in {{Scalable Algorithms}} for {{Large-Scale Systems}} ({{ScalA}})},
  author = {Aggarwal, Isha and Kashi, Aditya and Nayak, Pratik and Balos, Cody J. and Woodward, Carol S. and Anzt, Hartwig},
  year = {2021},
  month = nov,
  pages = {35--43},
  doi = {10/gn3xcg},
  copyright = {All rights reserved},
  keywords = {batched solvers,Chemistry,Computational modeling,Computer architecture,Conferences,Ginkgo,GPU,Graphics processing units,Mathematical models,Sparse linear systems}
}

@inproceedings{aggarwalPreconditionersBatchedIterative2022,
  title = {Preconditioners for~{{Batched Iterative Linear Solvers}} on~{{GPUs}}},
  booktitle = {Accelerating {{Science}} and {{Engineering Discoveries Through Integrated Research Infrastructure}} for {{Experiment}}, {{Big Data}}, {{Modeling}} and {{Simulation}}},
  author = {Aggarwal, Isha and Nayak, Pratik and Kashi, Aditya and Anzt, Hartwig},
  editor = {Doug, Kothe and Al, Geist and Pophale, Swaroop and Liu, Hong and {Parete-Koon}, Suzanne},
  year = {2022},
  series = {Communications in {{Computer}} and {{Information Science}}},
  pages = {38--53},
  publisher = {{Springer Nature Switzerland}},
  address = {{Cham}},
  doi = {10.1007/978-3-031-23606-8_3},
  copyright = {All rights reserved},
  isbn = {978-3-031-23606-8},
  langid = {english},
  keywords = {Batched preconditioners,Batched solvers,GINKGO,GPU,Sparse linear systems}
}

@inproceedings{kashiBatchedSparseIterative2022,
  title = {Batched Sparse Iterative Solvers on {{GPU}} for the Collision Operator for Fusion Plasma Simulations},
  booktitle = {2022 {{IEEE International Parallel}} and {{Distributed Processing Symposium}} ({{IPDPS}})},
  author = {Kashi, Aditya and Nayak, Pratik and Kulkarni, Dhruva and Scheinberg, Aaron and Lin, Paul and Anzt, Hartwig},
  year = {2022},
  month = may,
  pages = {157--167},
  issn = {1530-2075},
  doi = {10.1109/IPDPS53621.2022.00024},
  copyright = {All rights reserved},
  keywords = {batched solvers,fusion,Ginkgo,GPU,Graphics processing units,Iter,Iterative methods,performance portability,Plasma simulation,Production,Programming,simulation,Software architecture,Sparse linear systems,WDMApp,Xgc}
}

@inproceedings{nguyenPortingBatchedIterative2023,
  title = {Porting {{Batched Iterative Solvers}} onto {{Intel GPUs}} with {{SYCL}}},
  booktitle = {Proceedings of the {{SC}} '23 {{Workshops}} of {{The International Conference}} on {{High Performance Computing}}, {{Network}}, {{Storage}}, and {{Analysis}}},
  author = {Nguyen, Phuong and Nayak, Pratik and Anzt, Hartwig},
  year = {2023},
  month = nov,
  series = {{{SC-W}} '23},
  pages = {1048--1058},
  publisher = {{Association for Computing Machinery}},
  address = {{New York, NY, USA}},
  doi = {10.1145/3624062.3624181},
  urldate = {2023-11-11},
  copyright = {All rights reserved},
  isbn = {9798400707858},
  keywords = {Batched Linear Solvers,Intel GPUs,Performance Portability,SYCL}
}
```
