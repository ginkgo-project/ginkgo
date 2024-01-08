// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_PRECONDITIONER_BATCH_PRECONDITIONERS_HPP_
#define GKO_DPCPP_PRECONDITIONER_BATCH_PRECONDITIONERS_HPP_


#include <ginkgo/core/matrix/batch_identity.hpp>


#include "core/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_preconditioner {


#include "dpcpp/preconditioner/batch_identity.hpp.inc"


}  // namespace batch_preconditioner
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_PRECONDITIONER_BATCH_PRECONDITIONERS_HPP_
