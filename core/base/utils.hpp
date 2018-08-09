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

#ifndef GKO_CORE_BASE_UTILS_HPP_
#define GKO_CORE_BASE_UTILS_HPP_


#include "core/base/exception.hpp"
#include "core/base/std_extensions.hpp"
#include "core/base/types.hpp"


#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>


#ifndef NDEBUG
#include <cstdio>
#endif  // NDEBUG


namespace gko {


class Executor;


namespace detail {


template <typename T>
struct pointee_impl {
};

template <typename T>
struct pointee_impl<T *> {
    using type = T;
};

template <typename T>
struct pointee_impl<std::unique_ptr<T>> {
    using type = T;
};

template <typename T>
struct pointee_impl<std::shared_ptr<T>> {
    using type = T;
};

template <typename T>
using pointee = typename pointee_impl<typename std::decay<T>::type>::type;


template <typename T, typename = void>
struct is_clonable_impl : std::false_type {
};

template <typename T>
struct is_clonable_impl<T, xstd::void_t<decltype(std::declval<T>().clone())>>
    : std::true_type {
};

template <typename T>
constexpr bool is_clonable()
{
    return is_clonable_impl<typename std::decay<T>::type>::value;
}


template <typename T, typename = void>
struct is_clonable_to_impl : std::false_type {
};

template <typename T>
struct is_clonable_to_impl<
    T, xstd::void_t<decltype(std::declval<T>().clone(
           std::declval<std::shared_ptr<const Executor>>()))>>
    : std::true_type {
};

template <typename T>
constexpr bool is_clonable_to()
{
    return is_clonable_to_impl<typename std::decay<T>::type>::value;
}


template <typename T>
struct have_ownership_impl : std::false_type {
};

template <typename T>
struct have_ownership_impl<std::unique_ptr<T>> : std::true_type {
};

template <typename T>
struct have_ownership_impl<std::shared_ptr<T>> : std::true_type {
};

template <typename T>
constexpr bool have_ownership()
{
    return have_ownership_impl<typename std::decay<T>::type>::value;
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
 * @
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
inline typename std::enable_if<detail::have_ownership<Pointer>(),
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
inline typename std::enable_if<!detail::have_ownership<Pointer>(),
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
        throw NotSupported(__FILE__, __LINE__, __func__, typeid(obj).name());
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
        throw NotSupported(__FILE__, __LINE__, __func__, typeid(obj).name());
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


/**
 * A copy_back_deleter is a type of deleter that copies the data to an
 * internally referenced object before performing the deletion.
 *
 * The deleter will use the `copy_from` method to perform the copy, and then
 * delete the passed object using the `delete` keyword. This kind of deleter is
 * useful when temporarily copying an object with the intent of copying it back
 * once it goes out of scope.
 *
 * There is also a specialization for constant objects that does not perform the
 * copy, since a constant object couldn't have been changed.
 *
 * @tparam T  the type of object being deleted
 */
template <typename T>
class copy_back_deleter {
public:
    using pointer = T *;

    /**
     * Creates a new deleter object.
     *
     * @param original  the origin object where the data will be copied before
     *                  deletion
     */
    copy_back_deleter(pointer original) : original_{original} {}

    /**
     * Deletes the object.
     *
     * @param ptr  pointer to the object being deleted
     */
    void operator()(pointer ptr) const
    {
        original_->copy_from(ptr);
        delete ptr;
    }

private:
    pointer original_;
};


// specialization for constant objects, no need to copy back something that
// cannot change
template <typename T>
class copy_back_deleter<const T> {
public:
    using pointer = const T *;
    copy_back_deleter(pointer original) : original_{original} {}

    void operator()(pointer ptr) const { delete ptr; }

private:
    pointer original_;
};


/**
 * A temporary_clone is a special smart pointer-like object that is designed to
 * hold an object temporarily copied to another executor.
 *
 * After the temporary_clone goes out of scope, the stored object will be copied
 * back to its original location. This class is optimized to avoid copies if the
 * object is already on the correct executor, in which case it will just hold a
 * reference to that object, without performing the copy.
 *
 * @tparam T  the type of object held in the temporary_clone
 */
template <typename T>
class temporary_clone {
public:
    using value_type = T;
    using pointer = T *;

    /**
     * Creates a temporary_clone.
     *
     * @param exec  the executor where the clone will be created
     * @param ptr  a pointer to the object of which the clone will be created
     */
    explicit temporary_clone(std::shared_ptr<const Executor> exec, pointer ptr)
    {
        if (ptr->get_executor() == exec) {
            // just use the object we already have
            handle_ = handle_type(ptr, [](pointer) {});
        } else {
            // clone the object to the new executor and make sure it's copied
            // back before we delete it
            handle_ = handle_type(gko::clone(std::move(exec), ptr).release(),
                                  copy_back_deleter<T>(ptr));
        }
    }

    /**
     * Returns the object held by temporary_clone.
     *
     * @return the object held by temporary_clone
     */
    T *get() const { return handle_.get(); }

private:
    // std::function deleter allows to decide the (type of) deleter at runtime
    using handle_type = std::unique_ptr<T, std::function<void(T *)>>;

    handle_type handle_;
};


/**
 * Creates a temporary_clone.
 *
 * This is a helper function which avoids the need to explicitly specify the
 * type of the object, as would be the case if using the constructor of
 * temporary_clone.
 *
 * @param exec  the executor where the clone will be created
 * @param ptr  a pointer to the object of which the clone will be created
 */
template <typename T>
temporary_clone<T> make_temporary_clone(std::shared_ptr<const Executor> exec,
                                        T *ptr)
{
    return temporary_clone<T>(std::move(exec), ptr);
}


#if defined(__CUDA_ARCH__) && defined(__APPLE__)

#ifdef NDEBUG
#define GKO_ASSERT(condition) ((void)0)
#else  // NDEBUG
// Poor man's assertions on GPUs for MACs. They won't terminate the program
// but will at least print something on the screen
#define GKO_ASSERT(condition)                                               \
    ((condition)                                                            \
         ? ((void)0)                                                        \
         : ((void)printf("%s: %d: %s: Assertion `" #condition "' failed\n", \
                         __FILE__, __LINE__, __func__)))
#endif  // NDEBUG

#else  // defined(__CUDA_ARCH__) && defined(__APPLE__)

// Handle assertions normally on other systems
#define GKO_ASSERT(condition) assert(condition)

#endif  // defined(__CUDA_ARCH__) && defined(__APPLE__)


}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
