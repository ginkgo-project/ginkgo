/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


namespace detail {


template <typename T>
struct pointee_impl {};

template <typename T>
struct pointee_impl<T *> {
    using type = T;
};

template <typename T, typename Deleter>
struct pointee_impl<std::unique_ptr<T, Deleter>> {
    using type = T;
};

template <typename T>
struct pointee_impl<std::shared_ptr<T>> {
    using type = T;
};

template <typename T>
using pointee = typename pointee_impl<typename std::decay<T>::type>::type;


template <typename T, typename = void>
struct is_clonable_impl : std::false_type {};

template <typename T>
struct is_clonable_impl<T, xstd::void_t<decltype(std::declval<T>().clone())>>
    : std::true_type {};

template <typename T>
constexpr bool is_clonable()
{
    return is_clonable_impl<typename std::decay<T>::type>::value;
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
    return is_clonable_to_impl<typename std::decay<T>::type>::value;
}


template <typename T>
struct have_ownership_impl : std::false_type {};

template <typename T, typename Deleter>
struct have_ownership_impl<std::unique_ptr<T, Deleter>> : std::true_type {};

template <typename T>
struct have_ownership_impl<std::shared_ptr<T>> : std::true_type {};

template <typename T>
using have_ownership_s = have_ownership_impl<typename std::decay<T>::type>;

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
inline detail::cloned_type<Pointer> clone(const Pointer &p)
{
    static_assert(detail::is_clonable<detail::pointee<Pointer>>(),
                  "Object is not clonable");
    return detail::cloned_type<Pointer>(
        static_cast<typename std::remove_cv<detail::pointee<Pointer>>::type *>(
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
                                          const Pointer &p)
{
    static_assert(detail::is_clonable_to<detail::pointee<Pointer>>(),
                  "Object is not clonable");
    return detail::cloned_type<Pointer>(
        static_cast<typename std::remove_cv<detail::pointee<Pointer>>::type *>(
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
 * @param p  a pointer to the object
 *
 * @note The original pointer `p` becomes invalid after this call.
 */
template <typename OwningPointer>
inline detail::shared_type<OwningPointer> share(OwningPointer &&p)
{
    static_assert(detail::have_ownership<OwningPointer>(),
                  "OwningPointer does not have ownership of the object");
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
inline typename std::remove_reference<OwningPointer>::type &&give(
    OwningPointer &&p)
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
inline typename std::enable_if<detail::have_ownership_s<Pointer>::value,
                               detail::pointee<Pointer> *>::type
lend(const Pointer &p)
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
inline typename std::enable_if<!detail::have_ownership_s<Pointer>::value,
                               detail::pointee<Pointer> *>::type
lend(const Pointer &p)
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
inline typename std::decay<T>::type *as(U *obj)
{
    if (auto p = dynamic_cast<typename std::decay<T>::type *>(obj)) {
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
inline const typename std::decay<T>::type *as(const U *obj)
{
    if (auto p = dynamic_cast<const typename std::decay<T>::type *>(obj)) {
        return p;
    } else {
        throw NotSupported(__FILE__, __LINE__,
                           std::string{"gko::as<"} +
                               name_demangling::get_type_name(typeid(T)) + ">",
                           name_demangling::get_type_name(typeid(*obj)));
    }
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
inline std::unique_ptr<typename std::decay<T>::type> as(
    std::unique_ptr<U> &&obj)
{
    if (auto p = dynamic_cast<typename std::decay<T>::type *>(obj.get())) {
        obj.release();
        return std::unique_ptr<typename std::decay<T>::type>{p};
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
inline std::shared_ptr<typename std::decay<T>::type> as(std::shared_ptr<U> obj)
{
    auto ptr = std::dynamic_pointer_cast<typename std::decay<T>::type>(obj);
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
inline std::shared_ptr<const typename std::decay<T>::type> as(
    std::shared_ptr<const U> obj)
{
    auto ptr =
        std::dynamic_pointer_cast<const typename std::decay<T>::type>(obj);
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
    using pointer = T *;

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
