/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_POLYMORPHIC_OBJECT_HPP_
#define GKO_CORE_BASE_POLYMORPHIC_OBJECT_HPP_


#include "core/base/executor.hpp"
#include "core/base/polymorphic_object_interfaces.hpp"
#include "core/base/utils.hpp"


namespace gko {


/**
 * A PolymorphicObject is the abstract base for all "heavy" objects in Ginkgo
 * that behave polymorphically.
 *
 * It defines the basic utilities for all such objects such as copying, moving,
 * cloning and clearing the object. It takes into account that such objects are
 * most likely dynamically allocated, being managed by smart pointers, and use
 * polymorphically. It also takes into account that their data can be allocated
 * on different executors, and that they can be copied between those executors.
 *
 * Users that wish to add new implementations of polymorphic objects should look
 * into the EnablePolymorphicObject mixin, which provides sensible default
 * implementation of all methods of the PolymorphicObject.
 *
 * @see EnablePolymorphicObject
 */
class PolymorphicObject {
public:
    virtual ~PolymorphicObject() = default;

    // preserve the executor of the object
    PolymorphicObject &operator=(const PolymorphicObject &other)
    {
        return *this;
    }

    /**
     * Creates a new object of the same dynamic type as this object.
     *
     * This is the polymorphic equivalent of the _executor default constructor_
     * `decltype(*this)(exec);`.
     *
     * @param exec the executor where the foundation object will be created
     *
     * @return  a polymorphic object of the same type as this
     */
    std::unique_ptr<PolymorphicObject> create_foundation(
        std::shared_ptr<const Executor> exec) const
    {
        return std::unique_ptr<PolymorphicObject>(
            this->create_foundation_impl(std::move(exec)));
    }

    /**
     * Creates a new object of the same dynamic type as this object.
     *
     * This is a shorthand of create_foundation(std::shared_ptr<const Executor>)
     * that uses the executor of this object to construct the new object.
     *
     * @return  a polymorphic object of the same type as this
     */
    std::unique_ptr<PolymorphicObject> create_foundation() const
    {
        return this->create_foundation(exec_);
    }

    /**
     * Creates a clone of the object.
     *
     * This is the polymorphic equivalent of the _executor copy constructor_
     * `decltype(*this)(exec, this)`.
     *
     * @param exec the executor where the clone will be created
     *
     * @return A clone of the LinOp.
     */
    std::unique_ptr<PolymorphicObject> clone(
        std::shared_ptr<const Executor> exec) const
    {
        auto new_op = this->create_foundation(exec);
        new_op->copy_from(this);
        return new_op;
    }

    /**
     * Creates a clone of the object.
     *
     * This is a shorthand of clone(std::shared_ptr<const Executor>) that uses
     * the executor of this object to construct the new object.
     *
     * @return A clone of the LinOp.
     */
    std::unique_ptr<PolymorphicObject> clone() const
    {
        return this->clone(exec_);
    }

    /**
     * Copies another object to this object.
     *
     * This is the polymorphic equivalent of the copy assignment operator.
     *
     * @see copy_from_impl(const PolymorphicObject *)
     *
     * @param other  the object to copy
     *
     * @return this
     */
    PolymorphicObject *copy_from(const PolymorphicObject *other)
    {
        return this->copy_from_impl(other);
    }

    /**
     * Moves another object to this object.
     *
     * This is the polymorphic equivalent of the move assignment operator.
     *
     * @see copy_from_impl(std::unique_ptr<PolymorphicObject>)
     *
     * @param other  the object to move from
     *
     * @return this
     */
    PolymorphicObject *copy_from(std::unique_ptr<PolymorphicObject> other)
    {
        return this->copy_from_impl(std::move(other));
    }

    /**
     * Transforms the object back to its foundation state.
     *
     * Equivalent to `this->copy_from(this->create_foundation())`.
     *
     * @see clear_impl()
     *
     * @return this
     */
    PolymorphicObject *clear() { return this->clear_impl(); }

    /**
     * Returns the Executor of the object.
     *
     * @return Executor of the object
     */
    std::shared_ptr<const Executor> get_executor() const noexcept
    {
        return exec_;
    }

protected:
    /**
     * Creates a new polymorphic object on the requested executor.
     *
     * @param exec  executor where the object will be create.
     *
     * @note This method is defined as protected since a polymorphic object
     *       should not be created using their constructor directly, but by
     *       creating an std::unique_ptr to it. Defining the constructor as
     *       protected keeps these access rights when inheriting the constructor
     *       (`using PolymorphicObject::PolymorphicObject;`).
     */
    explicit PolymorphicObject(std::shared_ptr<const Executor> exec)
        : exec_{exec}
    {}

    /**
     * Implementers of PolymorphicObject should implement this function instead
     * of `create_foundation()`.
     *
     * This allows implementations of `create_foundation()` which preserve the
     * the static type of the returned object.
     *
     * @see EnablePolymorphicObject
     *
     * @param exec the executor where the foundation will be created
     *
     * @return  a polymorphic object of the same type as this
     */
    virtual PolymorphicObject *create_foundation_impl(
        std::shared_ptr<const Executor> exec) const = 0;

    /**
     * Implementers of PolymorphicObject should implement this function instead
     * of `copy_from(const PolymorphicObject *)`.
     *
     * This allows implementations of `copy_from(const PolymorphicObject *)`
     * which preserve the static type of the returned object.
     *
     * @see EnablePolymorphicObject
     *
     * @param other  the object to copy
     *
     * @return this
     */
    virtual PolymorphicObject *copy_from_impl(
        const PolymorphicObject *other) = 0;

    /**
     * Implementers of PolymorphicObject should implement this function instead
     * of `copy_from(std::unique_ptr<PolymorphicObject>)`.
     *
     * This allows implementations of
     * `copy_from(std::unique_ptr<PolymorphicObject>)` which preserve the static
     * type of the returned object.
     *
     * @see EnablePolymorphicObject
     *
     * @param other  the object to move from
     *
     * @return this
     */
    virtual PolymorphicObject *copy_from_impl(
        std::unique_ptr<PolymorphicObject> other) = 0;

    /**
     * Implementers of PolymorphicObject should implement this function instead
     * of `clear()`.
     *
     * This allows implementations of `clear()` which preserve the static type
     * of the returned object.
     *
     * @see EnablePolymorphicObject
     *
     * @return this
     */
    virtual PolymorphicObject *clear_impl() = 0;

private:
    std::shared_ptr<const Executor> exec_;
};


template <typename ConcreteObject, typename PolymorphicBase = PolymorphicObject>
class EnableAbstractPolymorphicObject : public PolymorphicBase {
public:
    using PolymorphicBase::PolymorphicBase;

    EnableAbstractPolymorphicObject &operator=(
        const EnableAbstractPolymorphicObject &) = default;

    EnableAbstractPolymorphicObject &operator=(
        EnableAbstractPolymorphicObject &&) = default;

    std::unique_ptr<ConcreteObject> create_foundation(
        std::shared_ptr<const Executor> exec) const
    {
        return std::unique_ptr<ConcreteObject>{static_cast<ConcreteObject *>(
            this->create_foundation_impl(std::move(exec)))};
    }

    std::unique_ptr<ConcreteObject> create_foundation() const
    {
        return this->create_foundation(this->get_executor());
    }

    std::unique_ptr<ConcreteObject> clone(
        std::shared_ptr<const Executor> exec) const
    {
        auto new_op = this->create_foundation(exec);
        new_op->copy_from(this);
        return new_op;
    }

    std::unique_ptr<ConcreteObject> clone() const
    {
        return this->clone(this->get_executor());
    }

    ConcreteObject *copy_from(const PolymorphicObject *other)
    {
        return static_cast<ConcreteObject *>(this->copy_from_impl(other));
    }

    ConcreteObject *copy_from(std::unique_ptr<PolymorphicObject> other)
    {
        return static_cast<ConcreteObject *>(
            this->copy_from_impl(std::move(other)));
    }

    ConcreteObject *clear()
    {
        return static_cast<ConcreteObject *>(this->clear_impl());
    }
};


/**
 * The EnablePolymorphicObject mixin provides a default implementation of
 * PolymorphicObject for a concrete polymorphic object.
 *
 * Think of it as an extension of default constructor/destructor/assignment
 * operators. In more detail, this mixin implements all PolymorphicObject's
 * pure virtual methods by using the objects constructors and assignment
 * operators. In addition, it hides appropriate PolymorphicObject's default
 * non-virtual methods with variants that use `ConcreteObject` as return type
 * instead of the generic PolymorphicObject. This simplifies the management of
 * polymorphic objects, as calls like `object->clone()` will return an object
 * with the same static type as the source object.
 *
 * As a result, creating a new PolymorphicObject requires that the implementer
 * only inherits from `EnablePolymorphicObject` and implement an _executor
 * default constructor_ to obtain a fully functional PolymorphicObject:
 *
 * ```c++
 * struct MyObject : EnablePolymorphicObject<MyObject> {
 *     MyObject(std::shared_ptr<const Executor> exec)
 *         : EnablePolymorphicObject<MyObject>(std::move(exec))
 *     {}
 * };
 * ```
 *
 * Consequently, when implementing new polymorphic objects, users are encouraged
 * to use this class as the base implementation, and override (or hide) some of
 * its methods if necessary.
 *
 * @tparam ConcreteObject  the concrete object for which the PolymorphicObject
 *                         interface is to be implemented
 * @tparam PolymorphicBase  the direct base class of ConcreteObject (has to be
 *                          a subclass of PolymorphicObject)
 */
template <typename ConcreteObject, typename PolymorphicBase = PolymorphicObject>
class EnablePolymorphicObject
    : public EnableAbstractPolymorphicObject<ConcreteObject, PolymorphicBase> {
public:
    using EnableAbstractPolymorphicObject<
        ConcreteObject, PolymorphicBase>::EnableAbstractPolymorphicObject;

    EnablePolymorphicObject &operator=(const EnablePolymorphicObject &) =
        default;

    EnablePolymorphicObject &operator=(EnablePolymorphicObject &&) = default;

protected:
    GKO_ENABLE_SELF(ConcreteObject);

    PolymorphicObject *create_foundation_impl(
        std::shared_ptr<const Executor> exec) const override
    {
        return new ConcreteObject(exec);
    }

    PolymorphicObject *copy_from_impl(const PolymorphicObject *other) override
    {
        as<ConvertibleTo<ConcreteObject>>(other)->convert_to(self());
        return this;
    }

    PolymorphicObject *copy_from_impl(
        std::unique_ptr<PolymorphicObject> other) override
    {
        as<ConvertibleTo<ConcreteObject>>(other.get())->move_to(self());
        return this;
    }

    PolymorphicObject *clear_impl() override
    {
        *self() = ConcreteObject{this->get_executor()};
        return this;
    }
};


}  // namespace gko


#endif  // GKO_CORE_BASE_POLYMORPHIC_OBJECT_HPP_
