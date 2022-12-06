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

#ifndef GKO_CORE_SOLVER_BATCH_DIRECT_KERNELS_HPP_
#define GKO_CORE_SOLVER_BATCH_DIRECT_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/log/batch_logging.hpp"


namespace gko {
namespace kernels {
namespace batch_direct {


#define GKO_DECLARE_BATCH_DIRECT_APPLY_KERNEL(_type)        \
    void apply(std::shared_ptr<const DefaultExecutor> exec, \
               matrix::BatchDense<_type> *const a,          \
               matrix::BatchDense<_type> *const b,          \
               gko::log::BatchLogData<_type> &logdata)

#define GKO_DECLARE_BATCH_DIRECT_SCALE_AND_COPY(_type)                \
    void scale_and_copy(std::shared_ptr<const DefaultExecutor> exec,  \
                        const matrix::BatchDense<_type> *scaling_vec, \
                        const matrix::BatchDense<_type> *orig,        \
                        matrix::BatchDense<_type> *scaled)


#define GKO_DECLARE_ALL_AS_TEMPLATES                  \
    template <typename ValueType>                     \
    GKO_DECLARE_BATCH_DIRECT_APPLY_KERNEL(ValueType); \
    template <typename ValueType>                     \
    GKO_DECLARE_BATCH_DIRECT_SCALE_AND_COPY(ValueType)


}  // namespace batch_direct


namespace omp {
namespace batch_direct {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_direct
}  // namespace omp


namespace cuda {
namespace batch_direct {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_direct
}  // namespace cuda


namespace reference {
namespace batch_direct {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_direct
}  // namespace reference


namespace hip {
namespace batch_direct {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_direct
}  // namespace hip


namespace dpcpp {
namespace batch_direct {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_direct
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_BATCH_DIRECT_KERNELS_HPP_
