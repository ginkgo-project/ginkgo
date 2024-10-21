// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#if GINKGO_BUILD_SYCL

#include <ginkgo/core/log/batch_logger.hpp>

#include "../../batch_cg_settings.hpp"
#include "../../batch_criteria.hpp"
#include "../../batch_identity.hpp"
#include "../../batch_logger.hpp"
#include "../../batch_multi_vector.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_template {
namespace batch_cg {


template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op* mat, multi_vector_view<const ValueType> b,
    multi_vector_view<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
    GKO_NOT_IMPLEMENTED;


}  // namespace batch_cg
}  // namespace batch_template
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#else


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_template {
namespace batch_cg {


template <typename ValueType, typename Op>
void apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const kernels::batch_cg::settings<remove_complex<ValueType>>& options,
    const Op mat, batch::multi_vector::uniform_batch<const ValueType> b,
    batch::multi_vector::uniform_batch<ValueType> x,
    batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
    GKO_NOT_IMPLEMENTED;


}  // namespace batch_cg
}  // namespace batch_template
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif
