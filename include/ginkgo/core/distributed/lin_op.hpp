// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_LIN_OP_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_LIN_OP_HPP_


#include <memory>
#include <type_traits>
#include <utility>


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/distributed/polymorphic_object.hpp>


namespace gko {
namespace experimental {


/**
 * This mixin does the same as EnableLinOp, but for concrete
 * types that are derived from distributed::DistributedBase.
 *
 * @see EnableLinOp.
 *
 * @tparam ConcreteLinOp  the concrete LinOp which is being implemented that
 *                        is derived from distributed::DistributedBase
 *                        [CRTP parameter]
 * @tparam PolymorphicBase  parent of ConcreteLinOp in the polymorphic
 *                          hierarchy, has to be a subclass of LinOp
 *
 * @ingroup LinOp
 */
template <typename ConcreteLinOp, typename PolymorphicBase = LinOp>
class EnableDistributedLinOp
    : public EnableDistributedPolymorphicObject<ConcreteLinOp, PolymorphicBase>,
      public EnablePolymorphicAssignment<ConcreteLinOp> {
public:
    using EnableDistributedPolymorphicObject<
        ConcreteLinOp, PolymorphicBase>::EnableDistributedPolymorphicObject;

    const ConcreteLinOp* apply(ptr_param<const LinOp> b,
                               ptr_param<LinOp> x) const
    {
        PolymorphicBase::apply(b, x);
        return self();
    }

    ConcreteLinOp* apply(ptr_param<const LinOp> b, ptr_param<LinOp> x)
    {
        PolymorphicBase::apply(b, x);
        return self();
    }

    const ConcreteLinOp* apply(ptr_param<const LinOp> alpha,
                               ptr_param<const LinOp> b,
                               ptr_param<const LinOp> beta,
                               ptr_param<LinOp> x) const
    {
        PolymorphicBase::apply(alpha, b, beta, x);
        return self();
    }

    ConcreteLinOp* apply(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b,
                         ptr_param<const LinOp> beta, ptr_param<LinOp> x)
    {
        PolymorphicBase::apply(alpha, b, beta, x);
        return self();
    }

protected:
    GKO_ENABLE_SELF(ConcreteLinOp);
};


}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_LIN_OP_HPP_
