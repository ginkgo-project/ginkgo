// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/components/bitvector.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace bitvector {


template GKO_DECLARE_BITVECTOR_FROM_SORTED_INDICES(gko::int32*);
template GKO_DECLARE_BITVECTOR_FROM_SORTED_INDICES(gko::int64*);
template GKO_DECLARE_BITVECTOR_FROM_SORTED_INDICES(const gko::int32*);
template GKO_DECLARE_BITVECTOR_FROM_SORTED_INDICES(const gko::int64*);


}  // namespace bitvector
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
