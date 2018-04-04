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


#include "core/base/exception_helpers.hpp"
#include "core/base/types.hpp"


#include <memory>
#include <type_traits>


namespace gko {
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
struct pointee_impl<std::weak_ptr<T>> {
    using type = T;
};

template <typename T>
using pointee = typename pointee_impl<typename std::decay<T>::type>::type;


template <typename T, typename = void>
struct is_clonable_impl : std::false_type {
};

template <typename T>
struct is_clonable_impl<T, decltype(std::declval<T>().clone())>
    : std::true_type {
};

template <typename T>
constexpr bool is_clonable()
{
    return is_clonable_impl<typename std::decay<T>::type>::value;
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
struct have_ownership_impl<std::weak_ptr<T>> : std::true_type {
};

template <typename T>
constexpr bool have_ownership()
{
    return have_ownership_impl<typename std::remove_cv<T>::type>::value;
}


template <typename Pointer>
using cloned_type =
    std::unique_ptr<typename std::remove_cv<pointee<Pointer>>::type>;


template <typename Pointer>
using shared_type = std::shared_ptr<pointee<Pointer>>;


}  // namespace detail


template <typename Pointer>
inline detail::cloned_type<Pointer> clone(const Pointer &p)
{
    static_assert(detail::is_clonable<detail::pointee<Pointer>>(),
                  "Object is not clonable");
    return detail::cloned_type<Pointer>(p->clone().release());
}


template <typename Pointer>
inline detail::shared_type<Pointer> share(Pointer &&p)
{
    static_assert(detail::have_ownership<Pointer>(),
                  "Pointer does not have ownership of the object");
    return detail::shared_type<Pointer>(std::move(p));
}


template <typename Pointer>
inline typename std::remove_reference<Pointer>::type &&give(Pointer &&p)
{
    static_assert(detail::have_ownership<Pointer>(),
                  "Pointer does not have ownership of the object");
    return std::move(p);
}


template <typename Pointer>
inline typename std::enable_if<detail::have_ownership<Pointer>(),
                               detail::pointee<Pointer> *>::type
lend(const Pointer &p)
{
    return p.get();
}

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
        throw NOT_SUPPORTED(obj);
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
        throw NOT_SUPPORTED(obj);
    }
}


}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
