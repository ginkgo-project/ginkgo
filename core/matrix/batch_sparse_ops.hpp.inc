/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

/** @file batch_sparse_ops.hpp.inc
 *
 * Overloads of (sparse) matrix device operations for every type of
 * batch matrix. This file is meant to be included in files that define
 * batch solver kernels.
 * Device kernels for the relevant backend should be included before this file.
 */

/**
 * Matrix-vector product of one matrix out of a batch.
 * This overload corresponds to a batch CSR operator.
 */
template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE void batch_spmv_single(
    const gko::batch_csr::BatchEntry<const ValueType> &a,
    const gko::batch_dense::BatchEntry<const ValueType> &b,
    const gko::batch_dense::BatchEntry<ValueType> &c)
{
    batch_csr::spmv_ker(a, b, c);
}


/**
 * 'Advanced' matrix-vector product of one matrix out of a batch.
 * This overload corresponds to a batch CSR operator.
 */
template <typename ValueType>
GKO_ATTRIBUTES GKO_INLINE void batch_adv_spmv_single(
    const ValueType alpha, const gko::batch_csr::BatchEntry<const ValueType> &a,
    const gko::batch_dense::BatchEntry<const ValueType> &b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType> &c)
{
    batch_csr::adv_spmv_ker(alpha, a, b, beta, c);
}
