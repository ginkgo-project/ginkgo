// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_TEMPORARY_CONVERSION_HPP_
#define GKO_PUBLIC_CORE_BASE_TEMPORARY_CONVERSION_HPP_


#include <memory>
#include <tuple>
#include <type_traits>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
namespace detail {


/**
 * @internal
 *
 * A convert_back_deleter is a type of deleter that converts and copies the data
 * to an internally referenced object before performing the deletion.
 *
 * The deleter will use the `convert_to` method to perform the conversion, and
 * then delete the passed object using the `delete` keyword. This kind of
 * deleter is useful when temporarily converting an object with the intent of
 * converting it back once it goes out of scope.
 *
 * There is also a specialization for constant objects that does not perform the
 * conversion, since a constant object couldn't have been changed.
 *
 * @tparam CopyType  the type of converted object being deleted
 * @tparam OrigType  the type of converted object to which the data will be
 *                   converted back
 */
template <typename CopyType, typename OrigType>
class convert_back_deleter {
    static_assert(std::is_base_of_v<ConvertibleTo<OrigType>, CopyType>,
                  "The CopyType must be convertible to OrigType");

public:
    using pointer = CopyType*;
    using original_pointer = OrigType*;

    /**
     * Creates a new deleter object.
     *
     * @param original  the origin object to which the data will be converted
     *                  back before deletion
     */
    convert_back_deleter(original_pointer original) : original_{original} {}

    /**
     * Deletes the object.
     *
     * @param ptr  pointer to the object being deleted
     */
    void operator()(pointer ptr) const
    {
        ptr->convert_to(original_);
        delete ptr;
    }

private:
    original_pointer original_;
};

// specialization for constant objects, no need to convert back something that
// cannot change
template <typename CopyType, typename OrigType>
class convert_back_deleter<const CopyType, OrigType> {
public:
    using pointer = const CopyType*;
    using original_pointer = OrigType*;
    convert_back_deleter(original_pointer) {}

    void operator()(pointer ptr) const { delete ptr; }
};


/**
 * @internal
 *
 * Helper type to create an empty object of TargetType that will be used
 * as the target object of a convert_to call. This can be specialized to
 * gain more control on how a TargetType object has to be created.
 *
 * The default implementation assumes that the TargetType object can be created
 * just by calling the create method on the executor.
 *
 * @tparam TargetType  The type an object shall be converted to
 */
template <typename TargetType>
struct conversion_target_helper {
    /**
     * Creates an empty object on the same executor as source.
     * *
     * @tparam SourceType  The type of the source object for the conversion
     * @param source  The source object for the conversion
     * @return  An unique_ptr of TargetType on the same executor as source.
     */
    template <typename SourceType>
    static std::unique_ptr<TargetType> create_empty(const SourceType* source)
    {
        return TargetType::create(source->get_executor());
    }
};


}  // namespace detail


/**
 * A temporary_conversion is a special smart pointer-like object that is
 * designed to hold an object temporarily converted to another format.
 *
 * After the temporary_conversion goes out of scope, the stored object will
 * be converted back to its original format. This class is optimized to
 * avoid copies if the object is already in the correct format, in which
 * case it will just hold a reference to that object, without performing the
 * conversion.
 *
 * @tparam T  the type of object held in the temporary_conversion
 */
template <typename T>
class temporary_conversion {
    // std::function deleter allows to decide the (type of) deleter at
    // runtime
    using handle_type = std::unique_ptr<T, std::function<void(T*)>>;

    template <typename OtherT>
    friend class temporary_conversion;

public:
    using value_type = T;
    using pointer = T*;
    using lin_op_type =
        std::conditional_t<std::is_const<T>::value, const LinOp, LinOp>;

    /**
     * Static assertion to check if a conversion from Other to T is possible.
     *
     * @note Succeeding this check can still lead to runtime exceptions, if
     *       the to convert from object is a nullptr, if its dynamic type is not
     *       T and T is not const and T is not convertible to Other.
     *
     * @tparam Other The type to convert from
     */
    template <typename Other>
    void static static_assert_compatibility()
    {
        using DecayT = std::decay_t<T>;
        using DecayOther = std::decay_t<Other>;
        static_assert(std::is_same_v<T, Other> ||
                      std::is_base_of_v<DecayOther, DecayT> ||
                      std::is_base_of_v<DecayT, DecayOther> ||
                      (std::is_base_of_v<ConvertibleTo<DecayT>, DecayOther> &&
                       (std::is_const_v<T> ||
                        std::is_base_of_v<ConvertibleTo<DecayOther>, DecayT>)));
    }

    /**
     * Dynamic assertion to check if a conversion from Other to T is possible.
     *
     * @tparam Other Type to convert from
     *
     * @param other the object to convert from
     *
     * @throws NotSupported if:
     *                      - other is nullptr, or
     *                      - other is not of static or dynamic type T and:
     *                        - other is not of dynamic type ConvertibleTo<T>,
     *                          or
     *                        - T is not const qualified and T is not derived
     *                          from ConvertibleTo<Other>
     */
    template <typename Other>
    void static dynamic_assert_compatibility(Other* other)
    {
        if (!other) {
            GKO_NOT_SUPPORTED(other);
        }
        if constexpr (std::is_same_v<T, Other>) {
            return;
        }
        if (dynamic_cast<T*>(other)) {
            return;
        }
        using DecayT = std::decay_t<T>;
        using DecayOther = std::decay_t<Other>;
        if (!dynamic_cast<const ConvertibleTo<DecayT>*>(other)) {
            GKO_NOT_SUPPORTED(other);
        }
        if constexpr (!std::is_const_v<T> &&
                      !std::is_base_of_v<ConvertibleTo<DecayOther>, DecayT>) {
            GKO_NOT_SUPPORTED(other);
        }
    }

    /**
     * Create a temporary conversion from a bare pointer.
     *
     * @tparam OrigT Type of the pointer to convert from, see
     *               static_assert_compatibility for requirements.
     *
     * @param orig Object to convert from
     *
     * @throws NotSupported see dynamic_assert_compatibility
     *
     * @return Temporary conversion of orig_ptr to type T
     */
    template <typename OrigT>
    static temporary_conversion create(OrigT* orig)
    {
        static_assert_compatibility<OrigT>();
        dynamic_assert_compatibility(orig);
        if constexpr (std::is_same_v<T, OrigT>) {
            return handle_type{orig, null_deleter<T>{}};
        }
        if (auto p = dynamic_cast<T*>(orig)) {
            return {handle_type{p, null_deleter<T>{}}};
        }
        using DecayT = std::decay_t<T>;
        using DecayOrig = std::decay_t<OrigT>;
        if constexpr (std::is_const_v<T> ||
                      std::is_base_of_v<ConvertibleTo<DecayOrig>, DecayT>) {
            auto converted =
                detail::conversion_target_helper<DecayT>::create_empty(orig);
            as<ConvertibleTo<DecayT>>(orig)->convert_to(converted);
            return {handle_type(converted.release(),
                                detail::convert_back_deleter<T, OrigT>{orig})};
        }
        GKO_NOT_SUPPORTED(orig);  // unreachable
    }

    /**
     * Create a temporary conversion that also owns orig.
     *
     * This create method takes ownership of the input pointer. It can be
     * useful, when chaining conversion. For example if the conversion A -> C
     * isn't implemented, but A -> B and B -> C are, then this can simplify the
     * conversion by just chaining the two.
     *
     * When the temporary conversion is deleted, orig will be deleted as well.
     *
     * @tparam OrigT Type of the pointer to convert from, see
     *               static_assert_compatibility for requirements.
     *
     * @param orig Object to convert from and take ownership of
     *
     * @throws NotSupported see dynamic_assert_compatibility
     *
     * @return Temporary conversion of orig_ptr to type T
     */
    template <typename OrigT>
    static temporary_conversion create(std::unique_ptr<OrigT> orig)
    {
        static_assert_compatibility<OrigT>();
        dynamic_assert_compatibility(orig.get());
        std::function<void(OrigT*)> deleter = orig.get_deleter();
        auto orig_ptr = orig.release();
        if constexpr (std::is_same_v<T, OrigT>) {
            return handle_type{orig_ptr, deleter};
        }
        if (auto p = dynamic_cast<T*>(orig_ptr)) {
            return {
                handle_type{p, [orig_ptr, deleter](T*) { deleter(orig_ptr); }}};
        }
        using DecayT = std::decay_t<T>;
        using DecayOrig = std::decay_t<OrigT>;
        if constexpr (std::is_const_v<T> ||
                      std::is_base_of_v<ConvertibleTo<DecayOrig>, DecayT>) {
            auto converted =
                detail::conversion_target_helper<DecayT>::create_empty(
                    orig_ptr);
            as<ConvertibleTo<DecayT>>(orig_ptr)->convert_to(converted);
            return {handle_type(converted.release(),
                                [orig_ptr, orig_deleter = deleter](T* ptr) {
                                    auto deleter =
                                        detail::convert_back_deleter<T, OrigT>{
                                            orig_ptr};
                                    deleter(ptr);
                                    orig_deleter(orig_ptr);
                                })};
        }
        GKO_NOT_SUPPORTED(orig_ptr);  // unreachable
    }

    /**
     * Create a temporary conversion for a base type T from an object of a
     * derived type.
     *
     * @tparam Derived A derived type of T
     *
     * @param derived Object to convert from
     *
     * @return Temporary conversion of derived to type T
     */
    template <typename Derived, typename = std::enable_if_t<std::is_base_of_v<
                                    std::decay_t<T>, std::decay_t<Derived>>>>
    static temporary_conversion create_from_derived(
        temporary_conversion<Derived>&& derived)
    {
        auto handle = std::move(derived).empty_out();
        return {handle_type{handle.release(),
                            [deleter = handle.get_deleter()](T* ptr) {
                                deleter(dynamic_cast<Derived*>(ptr));
                            }}};
    }

    /**
     * Create a temporary conversion for a derived type T from an object of a
     * base type.
     *
     * @tparam Base A base type of T
     *
     * @param base Object to convert from
     * @throws NotSupported if the dynamic type of base is not T
     *
     * @throws NotSupported if the dynamic type of base is not T
     *
     * @return Temporary conversion of base to type T
     */
    template <typename Base, typename = std::enable_if_t<std::is_base_of_v<
                                 std::decay_t<Base>, std::decay_t<T>>>>
    static temporary_conversion create_from_base(
        temporary_conversion<Base>&& base)
    {
        if (!dynamic_cast<T*>(base.get())) {
            GKO_NOT_SUPPORTED(base);
        }
        auto handle = std::move(base).empty_out();
        return {handle_type{dynamic_cast<T*>(handle.release()),
                            [deleter = handle.get_deleter()](T* ptr) {
                                deleter(static_cast<Base*>(ptr));
                            }}};
    }

    /**
     * Returns the object held by temporary_conversion.
     *
     * @return the object held by temporary_conversion
     */
    T* get() const { return handle_.get(); }

    /**
     * Calls a method on the underlying object.
     *
     * @return the underlying object
     */
    T* operator->() const { return handle_.get(); }

private:
    temporary_conversion(handle_type handle) : handle_{std::move(handle)} {}

    handle_type empty_out() && { return std::move(handle_); }

    handle_type handle_;
};


/**
 * Performs polymorphic type conversion of a temporary_conversion.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the temporary_conversion to the object which should be converted.
 *
 * @throw NotSupported if the conversion is not supported
 *
 * @return If successful, returns a temporary_conversion to the subtype. The
 *         ownership is transferred to the result.
 */
template <typename T, typename U>
temporary_conversion<T> as(temporary_conversion<U>&& obj)
{
    if (!dynamic_cast<T*>(obj.get())) {
        GKO_NOT_SUPPORTED(obj.get());
    }
    return temporary_conversion<T>::create_from_base(std::move(obj));
}

/**
 * Const correct overload of as(temporary_conversion<U>&&)
 */
template <typename T, typename U>
temporary_conversion<const T> as(temporary_conversion<const U>&& obj)
{
    if (!dynamic_cast<const T*>(obj.get())) {
        GKO_NOT_SUPPORTED(obj.get());
    }
    return temporary_conversion<const T>::create_from_base(std::move(obj));
}


namespace detail {


// For backwards compatibility
using gko::temporary_conversion;


}  // namespace detail
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_TEMPORARY_CONVERSION_HPP_
