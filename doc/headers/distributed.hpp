// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @defgroup distributed Distributed
 *
 * @brief A module dedicated to the implementation and usage of the
 * distributed linear algebra objects in Ginkgo.
 *
 * These types partition their rows over the ranks of an MPI communicator, so
 * that each rank stores and operates on only its own part of the data.
 *
 * @ingroup LinOp
 */
