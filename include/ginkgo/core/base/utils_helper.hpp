// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_UTILS_HELPER_HPP_
#define GKO_PUBLIC_CORE_BASE_UTILS_HELPER_HPP_


#include <functional>
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


#ifndef NDEBUG
#include <cstdio>
#endif  // NDEBUG


namespace gko {


class Executor;


/**
 * This class is used for function parameters in the place of raw pointers.
 * Pointer parameters should be used for everything that does not involve
 * transfer of ownership. It can be converted to from raw pointers, shared
 * pointers and unique pointers of the specified type or any derived type. This
 * allows functions to be called without having to use gko::lend or calling
 * .get() for every pointer argument. It probably has no use outside of function
 * parameters, as it is immutable.
 *
 * @tparam T  the pointed-to type
 */
template <typename T>
class ptr_param {
public:
    /** Initializes the ptr_param from a raw pointer. */
    ptr_param(T* ptr) : ptr_{ptr} {}

    /** Initializes the ptr_param from a shared_ptr. */
    template <typename U,
              std::enable_if_t<std::is_base_of<T, U>::value>* = nullptr>
    ptr_param(const std::shared_ptr<U>& ptr) : ptr_param{ptr.get()}
    {}

    /** Initializes the ptr_param from a unique_ptr. */
    template <typename U, typename Deleter,
              std::enable_if_t<std::is_base_of<T, U>::value>* = nullptr>
    ptr_param(const std::unique_ptr<U, Deleter>& ptr) : ptr_param{ptr.get()}
    {}

    /** Initializes the ptr_param from a ptr_param of a derived type. */
    template <typename U,
              std::enable_if_t<std::is_base_of<T, U>::value>* = nullptr>
    ptr_param(const ptr_param<U>& ptr) : ptr_param{ptr.get()}
    {}

    ptr_param(const ptr_param&) = default;

    ptr_param(ptr_param&&) = default;

    /** @return a reference to the underlying pointee. */
    T& operator*() const { return *ptr_; }

    /** @return the underlying pointer. */
    T* operator->() const { return ptr_; }

    /** @return the underlying pointer. */
    T* get() const { return ptr_; }

    /** @return true iff the underlying pointer is non-null. */
    explicit operator bool() const { return ptr_; }

    ptr_param& operator=(const ptr_param&) = delete;

    ptr_param& operator=(ptr_param&&) = delete;

private:
    T* ptr_;
};


namespace detail {


template <typename T>
using pointee =
    std::remove_reference_t<decltype(*std::declval<std::decay_t<T>>())>;


template <typename T, typename = void>
struct is_clonable_impl : std::false_type {};

template <typename T>
struct is_clonable_impl<T, xstd::void_t<decltype(std::declval<T>().clone())>>
    : std::true_type {};

template <typename T>
constexpr bool is_clonable()
{
    return is_clonable_impl<std::decay_t<T>>::value;
}


template <typename T, typename = void>
struct is_clonable_to_impl : std::false_type {};

template <typename T>
struct is_clonable_to_impl<
    T, xstd::void_t<decltype(std::declval<T>().clone(
           std::declval<std::shared_ptr<const Executor>>()))>>
    : std::true_type {};

template <typename T>
constexpr bool is_clonable_to()
{
    return is_clonable_to_impl<std::decay_t<T>>::value;
}


template <typename T>
struct have_ownership_impl : std::false_type {};

template <typename T, typename Deleter>
struct have_ownership_impl<std::unique_ptr<T, Deleter>> : std::true_type {};

template <typename T>
struct have_ownership_impl<std::shared_ptr<T>> : std::true_type {};

template <typename T>
using have_ownership_s = have_ownership_impl<std::decay_t<T>>;

template <typename T>
constexpr bool have_ownership()
{
    return have_ownership_s<T>::value;
}


template <typename Pointer>
using cloned_type =
    std::unique_ptr<typename std::remove_cv<pointee<Pointer>>::type>;


template <typename Pointer>
using shared_type = std::shared_ptr<pointee<Pointer>>;


}  // namespace detail


/**
 * Creates a unique clone of the object pointed to by `p`.
 *
 * The pointee (i.e. `*p`) needs to have a clone method that returns a
 * std::unique_ptr in order for this method to work.
 *
 * @tparam Pointer  type of pointer to the object (plain or smart pointer)
 *
 * @param p  a pointer to the object
 *
 * @note The difference between this function and directly calling
 *       LinOp::clone() is that this one preserves the static type of the
 *       object.
 */
template <typename Pointer>
inline detail::cloned_type<Pointer> clone(const Pointer& p)
{
    static_assert(detail::is_clonable<detail::pointee<Pointer>>(),
                  "Object is not clonable");
    return detail::cloned_type<Pointer>(
        static_cast<typename std::remove_cv<detail::pointee<Pointer>>::type*>(
            p->clone().release()));
}


/**
 * Creates a unique clone of the object pointed to by `p` on Executor `exec`.
 *
 * The pointee (i.e. `*p`) needs to have a clone method that takes an
 * executor and returns a std::unique_ptr in order for this method to work.
 *
 * @tparam Pointer  type of pointer to the object (plain or smart pointer)
 *
 * @param exec  the executor where the cloned object should be stored
 * @param p  a pointer to the object
 *
 * @note The difference between this function and directly calling
 *       LinOp::clone() is that this one preserves the static type of the
 *       object.
 */
template <typename Pointer>
inline detail::cloned_type<Pointer> clone(std::shared_ptr<const Executor> exec,
                                          const Pointer& p)
{
    static_assert(detail::is_clonable_to<detail::pointee<Pointer>>(),
                  "Object is not clonable");
    return detail::cloned_type<Pointer>(
        static_cast<typename std::remove_cv<detail::pointee<Pointer>>::type*>(
            p->clone(std::move(exec)).release()));
}


/**
 * Marks the object pointed to by `p` as shared.
 *
 * Effectively converts a pointer with ownership to std::shared_ptr.
 *
 * @tparam OwningPointer  type of pointer with ownership to the object
 *                        (has to be a smart pointer)
 *
 * @param p  a pointer to the object. It must be a temporary or explicitly
 *           marked movable (rvalue reference).
 *
 * @note The original pointer `p` becomes invalid after this call.
 */
template <typename OwningPointer>
inline detail::shared_type<OwningPointer> share(OwningPointer&& p)
{
    static_assert(detail::have_ownership<OwningPointer>(),
                  "OwningPointer does not have ownership of the object");
    static_assert(std::is_rvalue_reference<decltype(p)>::value,
                  "p must be an rvalue for this function to work");
    return detail::shared_type<OwningPointer>(std::move(p));
}


/**
 * Marks that the object pointed to by `p` can be given to the callee.
 *
 * Effectively calls `std::move(p)`.
 *
 * @tparam OwningPointer  type of pointer with ownership to the object
 *                        (has to be a smart pointer)
 *
 * @param p  a pointer to the object
 *
 * @note The original pointer `p` becomes invalid after this call.
 */
template <typename OwningPointer>
inline typename std::remove_reference<OwningPointer>::type&& give(
    OwningPointer&& p)
{
    static_assert(detail::have_ownership<OwningPointer>(),
                  "OwningPointer does not have ownership of the object");
    return std::move(p);
}


/**
 * Returns a non-owning (plain) pointer to the object pointed to by `p`.
 *
 * @tparam Pointer  type of pointer to the object (plain or smart pointer)
 *
 * @param p  a pointer to the object
 *
 * @note This is the overload for owning (smart) pointers, that behaves the
 *       same as calling .get() on the smart pointer.
 */
template <typename Pointer>
GKO_DEPRECATED("no longer necessary, just pass the object without lend")
inline typename std::enable_if<detail::have_ownership_s<Pointer>::value,
                               detail::pointee<Pointer>*>::type
    lend(const Pointer& p)
{
    return p.get();
}

/**
 * Returns a non-owning (plain) pointer to the object pointed to by `p`.
 *
 * @tparam Pointer  type of pointer to the object (plain or smart pointer)
 *
 * @param p  a pointer to the object
 *
 * @note This is the overload for non-owning (plain) pointers, that just
 *       returns `p`.
 */
template <typename Pointer>
GKO_DEPRECATED("no longer necessary, just pass the object without lend")
inline typename std::enable_if<!detail::have_ownership_s<Pointer>::value,
                               detail::pointee<Pointer>*>::type
    lend(const Pointer& p)
{
    return p;
}


/**
 * Performs polymorphic type conversion.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the object which should be converted
 *
 * @return If successful, returns a pointer to the subtype, otherwise throws
 *         NotSupported.
 */
template <typename T, typename U>
inline std::decay_t<T>* as(U* obj)
{
    if (auto p = dynamic_cast<std::decay_t<T>*>(obj)) {
        return p;
    } else {
        throw NotSupported(__FILE__, __LINE__,
                           std::string{"gko::as<"} +
                               name_demangling::get_type_name(typeid(T)) + ">",
                           name_demangling::get_type_name(typeid(*obj)));
    }
}

/**
 * Performs polymorphic type conversion.
 *
 * This is the constant version of the function.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the object which should be converted
 *
 * @return If successful, returns a pointer to the subtype, otherwise throws
 *         NotSupported.
 */
template <typename T, typename U>
inline const std::decay_t<T>* as(const U* obj)
{
    if (auto p = dynamic_cast<const std::decay_t<T>*>(obj)) {
        return p;
    } else {
        throw NotSupported(__FILE__, __LINE__,
                           std::string{"gko::as<"} +
                               name_demangling::get_type_name(typeid(T)) + ">",
                           name_demangling::get_type_name(typeid(*obj)));
    }
}


/**
 * Performs polymorphic type conversion on a ptr_param.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the object which should be converted
 *
 * @return If successful, returns a pointer to the subtype, otherwise throws
 *         NotSupported.
 */
template <typename T, typename U>
inline std::decay_t<T>* as(ptr_param<U> obj)
{
    return as<T>(obj.get());
}

/**
 * Performs polymorphic type conversion.
 *
 * This is the constant version of the function.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the object which should be converted
 *
 * @return If successful, returns a pointer to the subtype, otherwise throws
 *         NotSupported.
 */
template <typename T, typename U>
inline const std::decay_t<T>* as(ptr_param<const U> obj)
{
    return as<T>(obj.get());
}


/**
 * Performs polymorphic type conversion of a unique_ptr.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the unique_ptr to the object which should be converted.
 *             If successful, it will be reset to a nullptr.
 *
 * @return If successful, returns a unique_ptr to the subtype, otherwise throws
 *         NotSupported.
 */
template <typename T, typename U>
inline std::unique_ptr<std::decay_t<T>> as(std::unique_ptr<U>&& obj)
{
    if (auto p = dynamic_cast<std::decay_t<T>*>(obj.get())) {
        obj.release();
        return std::unique_ptr<std::decay_t<T>>{p};
    } else {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           name_demangling::get_type_name(typeid(*obj)));
    }
}


/**
 * Performs polymorphic type conversion of a shared_ptr.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the shared_ptr to the object which should be converted.
 *
 * @return If successful, returns a shared_ptr to the subtype, otherwise throws
 *         NotSupported. This pointer shares ownership with the input pointer.
 */
template <typename T, typename U>
inline std::shared_ptr<std::decay_t<T>> as(std::shared_ptr<U> obj)
{
    auto ptr = std::dynamic_pointer_cast<std::decay_t<T>>(obj);
    if (ptr) {
        return ptr;
    } else {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           name_demangling::get_type_name(typeid(*obj)));
    }
}


/**
 * Performs polymorphic type conversion of a shared_ptr.
 *
 * This is the constant version of the function.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the shared_ptr to the object which should be converted.
 *
 * @return If successful, returns a shared_ptr to the subtype, otherwise throws
 *         NotSupported. This pointer shares ownership with the input pointer.
 */
template <typename T, typename U>
inline std::shared_ptr<const std::decay_t<T>> as(std::shared_ptr<const U> obj)
{
    auto ptr = std::dynamic_pointer_cast<const std::decay_t<T>>(obj);
    if (ptr) {
        return ptr;
    } else {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           name_demangling::get_type_name(typeid(*obj)));
    }
}


/**
 * This is a deleter that does not delete the object.
 *
 * It is useful where the object has been allocated elsewhere and will be
 * deleted manually.
 */
template <typename T>
class null_deleter {
public:
    using pointer = T*;

    /**
     * Deletes the object.
     *
     * @param ptr  pointer to the object being deleted
     */
    void operator()(pointer) const noexcept {}
};

// a specialization for arrays
template <typename T>
class null_deleter<T[]> {
public:
    using pointer = T[];

    void operator()(pointer) const noexcept {}
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_UTILS_HELPER_HPP_
