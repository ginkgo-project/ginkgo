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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_POLYMORPHIC_OBJECT_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_POLYMORPHIC_OBJECT_HPP_


#include <memory>
#include <type_traits>


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/distributed/base.hpp>


namespace gko {
namespace experimental {


/**
 * This mixin does the same as EnablePolymorphicObject, but for concrete
 * types that are derived from distributed::DistributedBase.
 *
 * @see EnablePolymporphicObject.
 *
 * The following is a minimal example of a distributed PolymorphicObject:
 *
 * ```c++
 * struct MyObject : EnableDistributedPolymorphicObject<MyObject>,
 *                   distributed::DistributedBase {
 *     MyObject(std::shared_ptr<const Executor> exec, mpi::communicator comm)
 *         : EnableDistributedPolymorphicObject<MyObject>(std::move(exec)),
 *           distributed::DistributedBase(std::move(comm))
 *     {}
 * };
 * ```
 *
 * @tparam ConcreteObject  the concrete type which is being implemented that
 *                         is derived from distributed::DistributedBase
 *                         [CRTP parameter]
 * @tparam PolymorphicBase  parent of ConcreteObject in the polymorphic
 *                          hierarchy, has to be a subclass of polymorphic
 *                          object
 */
template <typename ConcreteObject, typename PolymorphicBase = PolymorphicObject>
class EnableDistributedPolymorphicObject
    : public EnableAbstractPolymorphicObject<ConcreteObject, PolymorphicBase> {
protected:
    using EnableAbstractPolymorphicObject<
        ConcreteObject, PolymorphicBase>::EnableAbstractPolymorphicObject;

    std::unique_ptr<PolymorphicObject> create_default_impl(
        std::shared_ptr<const Executor> exec) const override
    {
        return std::unique_ptr<ConcreteObject>{
            new ConcreteObject(exec, self()->get_communicator())};
    }

    PolymorphicObject* copy_from_impl(const PolymorphicObject* other) override
    {
        as<ConvertibleTo<ConcreteObject>>(other)->convert_to(self());
        return this;
    }

    PolymorphicObject* copy_from_impl(
        std::unique_ptr<PolymorphicObject> other) override
    {
        as<ConvertibleTo<ConcreteObject>>(other.get())->move_to(self());
        return this;
    }

    PolymorphicObject* move_from_impl(PolymorphicObject* other) override
    {
        as<ConvertibleTo<ConcreteObject>>(other)->move_to(self());
        return this;
    }

    PolymorphicObject* move_from_impl(
        std::unique_ptr<PolymorphicObject> other) override
    {
        as<ConvertibleTo<ConcreteObject>>(other.get())->move_to(self());
        return this;
    }

    PolymorphicObject* clear_impl() override
    {
        *self() =
            ConcreteObject{self()->get_executor(), self()->get_communicator()};
        return this;
    }

private:
    GKO_ENABLE_SELF(ConcreteObject);
};


}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_POLYMORPHIC_OBJECT_HPP_
