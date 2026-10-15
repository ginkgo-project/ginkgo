// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_PAR_ILUT_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_PAR_ILUT_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/par_ilut.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/device_views.hpp>

#include "core/base/kernel_declaration.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL(ValueType, IndexType) \
    void add_candidates(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        matrix::view::csr<const ValueType, const IndexType> lu,          \
        matrix::view::csr<const ValueType, const IndexType> a,           \
        matrix::view::csr<const ValueType, const IndexType> l,           \
        matrix::view::csr<const ValueType, const IndexType> u,           \
        matrix::CsrBuilder<ValueType, IndexType>* l_new_builder,         \
        matrix::CsrBuilder<ValueType, IndexType>* u_new_builder)

#define GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS_KERNEL(ValueType, IndexType) \
    void compute_l_u_factors(                                                \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        matrix::view::csr<const ValueType, const IndexType> a,               \
        matrix::view::csr<ValueType, IndexType> l,                           \
        matrix::view::coo<const ValueType, const IndexType> l_coo,           \
        matrix::view::csr<ValueType, IndexType> u,                           \
        matrix::view::coo<const ValueType, const IndexType> u_coo,           \
        matrix::view::csr<ValueType, IndexType> u_csc)

#define GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL(ValueType, IndexType)     \
    void threshold_select(                                                     \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        matrix::view::csr<const ValueType, const IndexType> m, IndexType rank, \
        array<ValueType>& tmp, array<remove_complex<ValueType>>& tmp2,         \
        remove_complex<ValueType>& threshold)

#define GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL(ValueType, IndexType) \
    void threshold_filter(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        matrix::view::csr<const ValueType, const IndexType> m,             \
        remove_complex<ValueType> threshold,                               \
        matrix::CsrBuilder<ValueType, IndexType>* m_out_builder,           \
        matrix::Coo<ValueType, IndexType>* m_out_coo, bool lower)

#define GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL(ValueType,         \
                                                            IndexType)         \
    void threshold_filter_approx(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        matrix::view::csr<const ValueType, const IndexType> m, IndexType rank, \
        array<ValueType>& tmp, remove_complex<ValueType>& threshold,           \
        matrix::CsrBuilder<ValueType, IndexType>* m_out_builder,               \
        matrix::Coo<ValueType, IndexType>* m_out_coo)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                      \
    constexpr auto sampleselect_searchtree_height = 8;                    \
    constexpr auto sampleselect_oversampling = 4;                         \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(par_ilut_factorization,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_ILUT_KERNELS_HPP_
