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

#ifndef GKO_CORE_MULTIGRID_AMGX_PGM_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_AMGX_PGM_KERNELS_HPP_


#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


namespace gko {
namespace kernels {
namespace amgx_pgm {


#define GKO_DECLARE_AMGX_PGM_MATCH_EDGE_KERNEL(IndexType)        \
    void match_edge(std::shared_ptr<const DefaultExecutor> exec, \
                    const Array<IndexType> &strongest_neighbor,  \
                    Array<IndexType> &agg)

#define GKO_DECLARE_AMGX_PGM_COUNT_UNAGG_KERNEL(IndexType)        \
    void count_unagg(std::shared_ptr<const DefaultExecutor> exec, \
                     const Array<IndexType> &agg, IndexType *num_unagg)

#define GKO_DECLARE_AMGX_PGM_RENUMBER_KERNEL(IndexType)        \
    void renumber(std::shared_ptr<const DefaultExecutor> exec, \
                  Array<IndexType> &agg, IndexType *num_agg)

#define GKO_DECLARE_AMGX_PGM_FIND_STRONGEST_NEIGHBOR(ValueType, IndexType) \
    void find_strongest_neighbor(                                          \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::Csr<ValueType, IndexType> *weight_mtx,               \
        const matrix::Diagonal<ValueType> *diag, Array<IndexType> &agg,    \
        Array<IndexType> &strongest_neighbor)

#define GKO_DECLARE_AMGX_PGM_ASSIGN_TO_EXIST_AGG(ValueType, IndexType)  \
    void assign_to_exist_agg(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::Csr<ValueType, IndexType> *weight_mtx,            \
        const matrix::Diagonal<ValueType> *diag, Array<IndexType> &agg, \
        Array<IndexType> &intermediate_agg)

#define GKO_DECLARE_AMGX_PGM_GENERATE(ValueType, IndexType)                 \
    void amgx_pgm_generate(std::shared_ptr<const DefaultExecutor> exec,     \
                           const matrix::Csr<ValueType, IndexType> *source, \
                           const Array<IndexType> &agg,                     \
                           matrix::Csr<ValueType, IndexType> *coarse)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                    \
    template <typename IndexType>                                       \
    GKO_DECLARE_AMGX_PGM_MATCH_EDGE_KERNEL(IndexType);                  \
    template <typename IndexType>                                       \
    GKO_DECLARE_AMGX_PGM_COUNT_UNAGG_KERNEL(IndexType);                 \
    template <typename IndexType>                                       \
    GKO_DECLARE_AMGX_PGM_RENUMBER_KERNEL(IndexType);                    \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_AMGX_PGM_FIND_STRONGEST_NEIGHBOR(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_AMGX_PGM_ASSIGN_TO_EXIST_AGG(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_AMGX_PGM_GENERATE(ValueType, IndexType)


}  // namespace amgx_pgm


namespace omp {
namespace amgx_pgm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace amgx_pgm
}  // namespace omp


namespace cuda {
namespace amgx_pgm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace amgx_pgm
}  // namespace cuda


namespace reference {
namespace amgx_pgm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace amgx_pgm
}  // namespace reference


namespace hip {
namespace amgx_pgm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace amgx_pgm
}  // namespace hip


namespace dpcpp {
namespace amgx_pgm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace amgx_pgm
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_AMGX_PGM_KERNELS_HPP_
