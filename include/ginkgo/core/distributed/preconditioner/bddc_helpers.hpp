// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if GINKGO_BUILD_MPI


namespace gko {
namespace experimental {
namespace distributed {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


enum class dof_type { inner, inactive, face, edge, vertex };

enum class scaling_type { stiffness, deluxe };

}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif
