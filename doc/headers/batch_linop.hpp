// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @defgroup BatchLinOp Batched Linear Operators
 *
 * @brief A module dedicated to the implementation and usage of the batched
 * linear operators in Ginkgo.
 *
 * A batched linear operator applies the same operation to a large number of
 * small, independent systems at once. Every item of a batch shares the same
 * dimensions and sparsity pattern, and no data is exchanged between items,
 * which is what allows the whole batch to be solved in a single kernel.
 */
