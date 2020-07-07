# Citing Ginkgo                                           {#citing_ginkgo}

The main Ginkgo paper describing Ginkgo's purpose, design and interface is
available through the following reference:

``` bibtex
@misc{anzt2020ginkgo,
    title={Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing},
    author={Hartwig Anzt and Terry Cojean and Goran Flegar and Fritz Göbel and Thomas Grützmacher and Pratik Nayak and Tobias Ribizel and Yuhsiang Mike Tsai and Enrique S. Quintana-Ortí},
    year={2020},
    eprint={2006.16852},
    archivePrefix={arXiv},
    primaryClass={cs.MS}
}
```

Multiple topical papers exist on Ginkgo and its algorithms. The following papers
can be used to cite specific aspects of the Ginkgo project.

### On Portability

``` bibtex
@misc{tsai2020amdportability,
    title={Preparing Ginkgo for AMD GPUs -- A Testimonial on Porting CUDA Code to HIP},
    author={Yuhsiang M. Tsai and Terry Cojean and Tobias Ribizel and Hartwig Anzt},
    year={2020},
    eprint={2006.14290},
    archivePrefix={arXiv},
    primaryClass={cs.MS}
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

### On SpMV performance

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
```
