/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_CORE_SOLVER_BATCH_GMRES_KERNELS_HPP_
#define GKO_CORE_SOLVER_BATCH_GMRES_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


#include "core/log/batch_logging.hpp"

namespace gko {
namespace kernels {
namespace batch_gmres {

constexpr int max_restart = 20;

/**
 * Options controlling the batch Gmres solver.
 */
template <typename RealType>
struct BatchGmresOptions {
    int max_its;
    RealType residual_tol;
    int restart_num;
    ::gko::stop::batch::ToleranceType tol_type;
};


/**
 * Calculates the amount of in-solver storage needed by batch-Gmres.
 *
 * The calculation includes
 * multivectors (length of each vector: number of rows in system matrix) for
 * - r
 * - z
 * - w
 * - x
 * - helper
 * multivectors (length of each vector: restart ) for
 * - cs
 * - sn
 * - y
 * multivectors (length of each vector: restart + 1) for
 * - s
 * matrices for
 * - Hessenberg matrix
 * - Krylov subspace basis vectors
 * and small arrays for
 * - rhs_norms
 * - res_norms
 * - tmp_norms
 */
template <typename ValueType>
inline int local_memory_requirement(const int num_rows, const int num_rhs,
                                    const int restart_num)
{
    return (5 * num_rows * num_rhs + 3 * restart_num * num_rhs +
            (restart_num + 1) * num_rhs +
            restart_num * (restart_num + 1) * num_rhs +
            num_rows * (restart_num + 1) * num_rhs) *
               sizeof(ValueType) +
           3 * num_rhs * sizeof(typename gko::remove_complex<ValueType>);
}

#define GKO_DECLARE_BATCH_GMRES_APPLY_KERNEL(_type)                \
    void apply(std::shared_ptr<const DefaultExecutor> exec,        \
               const gko::kernels::batch_gmres::BatchGmresOptions< \
                   remove_complex<_type>>& options,                \
               const BatchLinOp* a, const BatchLinOp* precon,      \
               const matrix::BatchDense<_type>* const b,           \
               matrix::BatchDense<_type>* const x,                 \
               gko::log::BatchLogData<_type>& logdata)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_BATCH_GMRES_APPLY_KERNEL(ValueType)


}  // namespace batch_gmres


namespace omp {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace omp


namespace cuda {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace cuda


namespace reference {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace reference


namespace hip {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace hip


namespace dpcpp {
namespace batch_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_gmres
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BATCH_GMRES_KERNELS_HPP_
