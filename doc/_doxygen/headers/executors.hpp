// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @defgroup Executor Executors
 *
 * @brief A module dedicated to the implementation and usage of the executors in
 * Ginkgo.
 *
 * Below, we provide a brief introduction to executors in Ginkgo, how they have
 * been implemented, how to best make use of them and how to add new executors.
 *
 * @section exec_1 Executors in Ginkgo.
 *
 * The first step in using the Ginkgo library consists of creating an
 * executor. Executors are used to specify the location for the data of linear
 * algebra objects, and to determine where the operations will be executed.
 * Ginkgo currently supports three different executor types:
 *
 * +    @ref exec_omp specifies that the data should be stored and the
 *      associated operations executed on an OpenMP-supporting device (e.g. host
 *      CPU);
 * +    @ref exec_cuda specifies that the data should be stored and the
 *      operations executed on the NVIDIA GPU accelerator;
 * +    @ref exec_hip uses the HIP library to compile code for either NVIDIA or
 *      AMD GPU accelerator;
 * +    @ref exec_dpcpp uses the DPC++ compiler for any DPC++ supported hardware
 *      (e.g. Intel CPUs, GPU, FPGAs, ...);
 * +    @ref exec_ref executes a non-optimized reference implementation,
 *      which can be used to debug the library.
 */
