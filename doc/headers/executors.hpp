/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
 * associated operations executed on an OpenMP-supporting device (e.g. host
 * CPU);
 * +    @ref exec_cuda specifies that the data should be stored and the
 *      operations executed on the NVIDIA GPU accelerator;
 * +    @ref exec_hip uses the HIP library to compile code for either NVIDIA or
 *      AMD GPU accelerator;
 * +    @ref exec_ref executes a non-optimized reference implementation,
 *      which can be used to debug the library.
 */
