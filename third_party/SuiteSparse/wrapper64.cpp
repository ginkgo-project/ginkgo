// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

using std::int32_t;
using std::int64_t;
using std::size_t;
using std::sqrt;
#define ASSERT(expr) assert(expr)

namespace gko {
namespace experimental {
namespace reorder {
namespace suitesparse_wrapper {


#include "AMD/Source/amd_l2.c"
#include "AMD/Source/amd_l_defaults.c"
#include "AMD/Source/amd_l_post_tree.c"
#include "AMD/Source/amd_l_postorder.c"


}  // namespace suitesparse_wrapper
}  // namespace reorder
}  // namespace experimental
}  // namespace gko
