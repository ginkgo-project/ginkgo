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

#ifndef GKO_CORE_REORDER_RCM_KERNELS_HPP_
#define GKO_CORE_REORDER_RCM_KERNELS_HPP_


#include <ginkgo/core/reorder/rcm.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_RCM_GET_PERMUTATION_KERNEL(IndexType)                     \
    void get_permutation(std::shared_ptr<const DefaultExecutor> exec,         \
                         IndexType num_vertices, const IndexType *row_ptrs,   \
                         const IndexType *col_idxs, const IndexType *degrees, \
                         IndexType *permutation, IndexType *inv_permutation,  \
                         gko::reorder::starting_strategy strategy)

#define GKO_DECLARE_RCM_GET_DEGREE_OF_NODES_KERNEL(IndexType)             \
    void get_degree_of_nodes(std::shared_ptr<const DefaultExecutor> exec, \
                             IndexType num_vertices,                      \
                             const IndexType *row_ptrs, IndexType *degrees)

#define GKO_DECLARE_ALL_AS_TEMPLATES                       \
    template <typename IndexType>                          \
    GKO_DECLARE_RCM_GET_DEGREE_OF_NODES_KERNEL(IndexType); \
    template <typename IndexType>                          \
    GKO_DECLARE_RCM_GET_PERMUTATION_KERNEL(IndexType)


namespace omp {
namespace rcm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace rcm
}  // namespace omp


namespace cuda {
namespace rcm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace rcm
}  // namespace cuda


namespace hip {
namespace rcm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace rcm
}  // namespace hip


namespace dpcpp {
namespace rcm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace rcm
}  // namespace dpcpp


namespace reference {
namespace rcm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace rcm
}  // namespace reference


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_REORDER_RCM_KERNELS_HPP_
