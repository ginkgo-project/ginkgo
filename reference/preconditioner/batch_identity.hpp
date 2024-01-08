// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_IDENTITY_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_IDENTITY_HPP_


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace host {
namespace batch_preconditioner {


/**
 * Identity preconditioner for batch solvers. Enables unpreconditioned solves
 * by performing a copy of the preconditioned vector to the un-preconditioned
 * vector.
 */
template <typename ValueType>
class Identity final {
public:
    using value_type = ValueType;

    /**
     * The size of the work vector required in case of static allocation.
     */
    static constexpr int work_size = 0;

    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    static int dynamic_work_size(int, int) { return 0; }

    /**
     * Sets the input and generates the identity preconditioner.(Nothing needs
     * to be actually generated.)
     */
    template <typename batch_item_type>
    void generate(size_type, const batch_item_type&, ValueType* const)
    {}

    /**
     * Applies the preconditioner to the vector. For the identity
     * preconditioner, this is equivalent to a copy.
     */
    void apply(const gko::batch::multi_vector::batch_item<const ValueType>& r,
               const gko::batch::multi_vector::batch_item<ValueType>& z) const
    {
        for (int i = 0; i < r.num_rows; i++) {
            z.values[i * z.stride] = r.values[i * r.stride];
        }
    }
};


}  // namespace batch_preconditioner
}  // namespace host
}  // namespace kernels
}  // namespace gko


#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_IDENTITY_HPP_
