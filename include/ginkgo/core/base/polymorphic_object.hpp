/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_POLYMORPHIC_OBJECT_HPP_
#define GKO_CORE_BASE_POLYMORPHIC_OBJECT_HPP_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {


/**
 * A PolymorphicObject is the abstract base for all "heavy" objects in Ginkgo
 * that behave polymorphically.
 *
 * It defines the basic utilities (copying moving, cloning, clearing the
 * objects) for all such objects. It takes into account that these objects are
 * dynamically allocated, managed by smart pointers, and used polymorphically.
 * Additionally, it assumes their data can be allocated on different executors,
 * and that they can be copied between those executors.
 *
 * @note Most of the public methods of this class should not be overridden
 *       directly, and are thus not virtual. Instead, there are equivalent
 *       protected methods (ending in <method_name>_impl) that should be
 *       overriden instead. This allows polymorphic objects to implement default
 *       behavior around virtual methods (parameter checking, type casting).
 *
 * @see EnablePolymorphicObject if you wish to implement a concrete polymorphic
 *      object and have sensible defaults generated automatically.
 *      EnableAbstractPolymorphicObject if you wish to implement a new abstract
 *      polymorphic object, and have the return types of the methods updated to
 *      your type (instead of having them return PolymorphicObject).
 */
class PolymorphicObject : public log::EnableLogging<PolymorphicObject> {
public:
    virtual ~PolymorphicObject()
    {
        this->template log<log::Logger::polymorphic_object_deleted>(exec_.get(),
                                                                    this);
    }

    // preserve the executor of the object
    PolymorphicObject &operator=(const PolymorphicObject &) { return *this; }

    /**
     * Creates a new "default" object of the same dynamic type as this object.
     *
     * This is the polymorphic equivalent of the _executor default constructor_
     * `decltype(*this)(exec);`.
     *
     * @param exec  the executor where the object will be created
     *
     * @return a polymorphic object of the same type as this
     */
    std::unique_ptr<PolymorphicObject> create_default(
        std::shared_ptr<const Executor> exec) const
    {
        this->template log<log::Logger::polymorphic_object_create_started>(
            exec_.get(), this);
        auto created = this->create_default_impl(std::move(exec));
        this->template log<log::Logger::polymorphic_object_create_completed>(
            exec_.get(), this, created.get());
        return created;
    }

    /**
     * Creates a new "default" object of the same dynamic type as this object.
     *
     * This is a shorthand for create_default(std::shared_ptr<const Executor>)
     * that uses the executor of this object to construct the new object.
     *
     * @return a polymorphic object of the same type as this
     */
    std::unique_ptr<PolymorphicObject> create_default() const
    {
        return this->create_default(exec_);
    }

    /**
     * Creates a clone of the object.
     *
     * This is the polymorphic equivalent of the _executor copy constructor_
     * `decltype(*this)(exec, this)`.
     *
     * @param exec  the executor where the clone will be created
     *
     * @return A clone of the LinOp.
     */
    std::unique_ptr<PolymorphicObject> clone(
        std::shared_ptr<const Executor> exec) const
    {
        auto new_op = this->create_default(exec);
        new_op->copy_from(this);
        return new_op;
    }

    /**
     * Creates a clone of the object.
     *
     * This is a shorthand for clone(std::shared_ptr<const Executor>) that uses
     * the executor of this object to construct the new object.
     *
     * @return A clone of the LinOp.
     */
    std::unique_ptr<PolymorphicObject> clone() const
    {
        return this->clone(exec_);
    }

    /**
     * Copies another object into this object.
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
        this->template log<log::Logger::polymorphic_object_copy_started>(
            exec_.get(), other, this);
        auto copied = this->copy_from_impl(other);
        this->template log<log::Logger::polymorphic_object_copy_completed>(
            exec_.get(), other, this);
        return copied;
    }

    /**
     * Moves another object into this object.
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
        this->template log<log::Logger::polymorphic_object_copy_started>(
            exec_.get(), other.get(), this);
        auto copied = this->copy_from_impl(std::move(other));
        this->template log<log::Logger::polymorphic_object_copy_completed>(
            exec_.get(), other.get(), this);
        return copied;
    }

    /**
     * Transforms the object into its default state.
     *
     * Equivalent to `this->copy_from(this->create_default())`.
     *
     * @see clear_impl() when implementing this method
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
    // This method is defined as protected since a polymorphic object should not
    // be created using their constructor directly, but by creating an
    // std::unique_ptr to it. Defining the constructor as protected keeps these
    // access rights when inheriting the constructor.
    /**
     * Creates a new polymorphic object on the requested executor.
     *
     * @param exec  executor where the object will be created
     */
    explicit PolymorphicObject(std::shared_ptr<const Executor> exec)
        : exec_{std::move(exec)}
    {}

    /**
     * Implementers of PolymorphicObject should override this function instead
     * of create_default().
     *
     * @param exec  the executor where the object will be created
     *
     * @return a polymorphic object of the same type as this
     */
    virtual std::unique_ptr<PolymorphicObject> create_default_impl(
        std::shared_ptr<const Executor> exec) const = 0;

    /**
     * Implementers of PolymorphicObject should implement this function instead
     * of copy_from(const PolymorphicObject *).
     *
     * @param other  the object to copy
     *
     * @return this
     */
    virtual PolymorphicObject *copy_from_impl(
        const PolymorphicObject *other) = 0;

    /**
     * Implementers of PolymorphicObject should implement this function instead
     * of copy_from(std::unique_ptr<PolymorphicObject>).
     *
     * @param other  the object to move from
     *
     * @return this
     */
    virtual PolymorphicObject *copy_from_impl(
        std::unique_ptr<PolymorphicObject> other) = 0;

    /**
     * Implementers of PolymorphicObject should implement this function instead
     * of clear().
     *
     * @return this
     */
    virtual PolymorphicObject *clear_impl() = 0;

private:
    std::shared_ptr<const Executor> exec_;
};


/**
 * This mixin inherits from (a subclass of) PolymorphicObject and provides a
 * base implementation of a new abstract object.
 *
 * It uses method hiding to update the parameter and return types from
 * `PolymorphicObject to `AbstractObject` wherever it makes sense.
 * As opposed to EnablePolymorphicObject, it does not implement
 * PolymorphicObject's virtual methods.
 *
 * @tparam AbstractObject  the abstract class which is being implemented
 *                         [CRTP parameter]
 * @tparam PolymorphicBase  parent of AbstractObject in the polymorphic
 *                          hierarchy, has to be a subclass of polymorphic
 *                          object
 *
 * @see EnablePolymorphicObject for creating a concrete subclass of
 *      PolymorphicObject.
 */
template <typename AbstactObject, typename PolymorphicBase = PolymorphicObject>
class EnableAbstractPolymorphicObject : public PolymorphicBase {
public:
    using PolymorphicBase::PolymorphicBase;

    std::unique_ptr<AbstactObject> create_default(
        std::shared_ptr<const Executor> exec) const
    {
        return std::unique_ptr<AbstactObject>{static_cast<AbstactObject *>(
            this->create_default_impl(std::move(exec)).release())};
    }

    std::unique_ptr<AbstactObject> create_default() const
    {
        return this->create_default(this->get_executor());
    }

    std::unique_ptr<AbstactObject> clone(
        std::shared_ptr<const Executor> exec) const
    {
        auto new_op = this->create_default(exec);
        new_op->copy_from(this);
        return new_op;
    }

    std::unique_ptr<AbstactObject> clone() const
    {
        return this->clone(this->get_executor());
    }

    AbstactObject *copy_from(const PolymorphicObject *other)
    {
        return static_cast<AbstactObject *>(this->copy_from_impl(other));
    }

    AbstactObject *copy_from(std::unique_ptr<PolymorphicObject> other)
    {
        return static_cast<AbstactObject *>(
            this->copy_from_impl(std::move(other)));
    }

    AbstactObject *clear()
    {
        return static_cast<AbstactObject *>(this->clear_impl());
    }
};


/**
 * This macro implements the `self()` method, which is a shortcut for
 * `static_cast<_type *>(this)`.
 *
 * It also provides a constant version overload. It is often useful when
 * implementing mixins which depend on the type of the affected object, in which
 * case the type is set to the affected object (i.e. the CRTP parameter).
 */
#define GKO_ENABLE_SELF(_type)                                    \
    _type *self() noexcept { return static_cast<_type *>(this); } \
                                                                  \
    const _type *self() const noexcept                            \
    {                                                             \
        return static_cast<const _type *>(this);                  \
    }


/**
 * ConvertibleTo interface is used to mark that the implementer can be converted
 * to the object of ResultType.
 *
 * This interface is used to enable conversions between polymorphic objects.
 * To mark that an object of type `U` can be converted to an object of type `V`,
 * `U` should implement ConvertibleTo<V>.
 * Then, the implementation of `PolymorphicObject::copy_from` automatically
 * generated by `EnablePolymorphicObject` mixin will use RTTI to figure out that
 * `U` implements the interface and convert it using the convert_to / move_to
 * methods of the interface.
 *
 * As an example, the following function:
 *
 * ```c++
 * void my_function(const U *u, V *v) {
 *     v->copy_from(u);
 * }
 * ```
 *
 * will convert object `u` to object `v` by checking that `u` can be dynamically
 * casted to `ConvertibleTo\<V\>`, and calling
 * ConvertibleTo\<V\>::convert_to(V*)` to do the actual conversion.
 *
 * In case `u` is passed as a unique_ptr, call to `convert_to` will be replaced
 * by a call to `move_to` and trigger move semantics.
 *
 * @tparam ResultType  the type to which the implementer can be converted to,
 *                     has to be a subclass of PolymorphicObject
 */
template <typename ResultType>
class ConvertibleTo {
public:
    using result_type = ResultType;

    virtual ~ConvertibleTo() = default;

    /**
     * Converts the implementer to an object of type result_type.
     *
     * @param result  the object used to store the result of the conversion
     */
    virtual void convert_to(result_type *result) const = 0;

    /**
     * Converts the implementer to an object of type result_type by moving data
     * from this object.
     *
     * This method is used when the implementer is a temporary object, and move
     * semantics can be used.
     *
     * @param result  the object used to emplace the result of the conversion
     *
     * @note ConvertibleTo::move_to can be implemented by simply calling
     *       ConvertibleTo::convert_to. However, this operation can often be
     *       optimized by exploiting the fact that implementer's data can be
     *       moved to the result.
     */
    virtual void move_to(result_type *result) = 0;
};


namespace detail {


template <typename R, typename T>
std::unique_ptr<R, std::function<void(R *)>> copy_and_convert_to_impl(
    std::shared_ptr<const Executor> exec, T *obj)
{
    auto obj_as_r = dynamic_cast<R *>(obj);
    if (obj_as_r != nullptr && obj->get_executor() == exec) {
        return {obj_as_r, [](R *) {}};
    } else {
        auto copy = R::create(exec);
        as<ConvertibleTo<xstd::decay_t<R>>>(obj)->convert_to(lend(copy));
        return {copy.release(), std::default_delete<R>{}};
    }
}


}  // namespace detail


/**
 * Converts the object to R and places it on Executor exec.
 *
 * If the object is already of the requested type and on the requested executor,
 * the copy and conversion is avoided and a reference to the original object is
 * returned instead.
 *
 * @tparam R  the type to which the object should be converted
 * @tparam T  the type of the input object
 *
 * @param exec  the executor where the result should be placed
 * @param obj  the object that should be converted
 *
 * @return a unique pointer (with dynamically bound deleter) to the converted
 *         object
 */
template <typename R, typename T>
std::unique_ptr<R, std::function<void(R *)>> copy_and_convert_to(
    std::shared_ptr<const Executor> exec, T *obj)
{
    return detail::copy_and_convert_to_impl<R>(std::move(exec), obj);
}


/**
 * @copydoc copy_and_convert_to(std::shared_ptr<const Executor>, T*)
 *
 * @note This is a version of the function which adds the const qualifier to the
 *       result if the input had the same qualifier.
 */
template <typename R, typename T>
std::unique_ptr<const R, std::function<void(const R *)>> copy_and_convert_to(
    std::shared_ptr<const Executor> exec, const T *obj)
{
    return detail::copy_and_convert_to_impl<const R>(std::move(exec), obj);
}


/**
 * This mixin inherits from (a subclass of) PolymorphicObject and provides a
 * base implementation of a new concrete polymorphic object.
 *
 * The mixin changes parameter and return types of appropriate public methods of
 * PolymorphicObject in the same way EnableAbstractPolymorphicObject does.
 * In addition, it also provides default implementations of PolymorphicObject's
 * vritual methods by using the _executor default constructor_ and the
 * assignment operator of ConcreteObject. Consequently, the following is a
 * minimal example of PolymorphicObject:
 *
 * ```c++
 * struct MyObject : EnablePolymorphicObject<MyObject> {
 *     MyObject(std::shared_ptr<const Executor> exec)
 *         : EnablePolymorphicObject<MyObject>(std::move(exec))
 *     {}
 * };
 * ```
 *
 * In a way, this mixin can be viewed as an extension of default
 * constructor/destructor/assignment operators.
 *
 * @note  This mixin does not enable copying the polymorphic object to the
 *        object of the same type (i.e. it does not implement the
 *        ConvertibleTo<ConcreteObject> interface). To enable a default
 *        implementation of this interface see the EnablePolymorphicAssignment
 *        mixin.
 *
 * @tparam ConcreteObject  the concrete type which is being implemented
 *                         [CRTP parameter]
 * @tparam PolymorphicBase  parent of ConcreteObject in the polymorphic
 *                          hierarchy, has to be a subclass of polymorphic
 *                          object
 */
template <typename ConcreteObject, typename PolymorphicBase = PolymorphicObject>
class EnablePolymorphicObject
    : public EnableAbstractPolymorphicObject<ConcreteObject, PolymorphicBase> {
protected:
    using EnableAbstractPolymorphicObject<
        ConcreteObject, PolymorphicBase>::EnableAbstractPolymorphicObject;

    std::unique_ptr<PolymorphicObject> create_default_impl(
        std::shared_ptr<const Executor> exec) const override
    {
        return std::unique_ptr<ConcreteObject>{new ConcreteObject(exec)};
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

private:
    GKO_ENABLE_SELF(ConcreteObject);
};


/**
 * This mixin is used to enable a default PolymorphicObject::copy_from()
 * implementation for objects that have implemented conversions between them.
 *
 * The requirement is that there is either a conversion constructor from
 * `ConcreteType` in `ResultType`, or a conversion operator to `ResultType` in
 * `ConcreteType`.
 *
 * @tparam ConcreteType  the concrete type from which the copy_from is being
 *                       enabled [CRTP parameter]
 * @tparam ResultType  the type to which copy_from is being enabled
 */
template <typename ConcreteType, typename ResultType = ConcreteType>
class EnablePolymorphicAssignment : public ConvertibleTo<ResultType> {
public:
    using result_type = ResultType;

    void convert_to(result_type *result) const override { *result = *self(); }

    void move_to(result_type *result) override { *result = std::move(*self()); }

private:
    GKO_ENABLE_SELF(ConcreteType);
};


/**
 * This mixin implements a static `create()` method on `ConcreteType` that
 * dynamically allocates the memory, uses the passed-in arguments to construct
 * the object, and returns an std::unique_ptr to such an object.
 *
 * @tparam ConcreteObject  the concrete type for which `create()` is being
 *                         implemented [CRTP parameter]
 */
template <typename ConcreteType>
class EnableCreateMethod {
public:
    template <typename... Args>
    static std::unique_ptr<ConcreteType> create(Args &&... args)
    {
        return std::unique_ptr<ConcreteType>(
            new ConcreteType(std::forward<Args>(args)...));
    }
};


}  // namespace gko


#endif  // GKO_CORE_BASE_POLYMORPHIC_OBJECT_HPP_
