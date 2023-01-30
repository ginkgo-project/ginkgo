/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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

    const ConcreteLinOp* apply(pointer_param<const LinOp> b,
                               pointer_param<LinOp> x) const
    {
        PolymorphicBase::apply(b, x);
        return self();
    }

    ConcreteLinOp* apply(pointer_param<const LinOp> b, pointer_param<LinOp> x)
    {
        PolymorphicBase::apply(b, x);
        return self();
    }

    const ConcreteLinOp* apply(pointer_param<const LinOp> alpha,
                               pointer_param<const LinOp> b,
                               pointer_param<const LinOp> beta,
                               pointer_param<LinOp> x) const
    {
        PolymorphicBase::apply(alpha, b, beta, x);
        return self();
    }

    ConcreteLinOp* apply(pointer_param<const LinOp> alpha,
                         pointer_param<const LinOp> b,
                         pointer_param<const LinOp> beta,
                         pointer_param<LinOp> x)
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
