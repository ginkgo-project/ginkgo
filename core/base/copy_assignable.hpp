// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_COPY_ASSIGNABLE_HPP_
#define GKO_CORE_BASE_COPY_ASSIGNABLE_HPP_


#include <new>
#include <type_traits>
#include <utility>

#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace detail {


/**
 * Helper class to make a type copy assignable.
 *
 * This class wraps an object of a type that has a copy constructor, but not a
 * copy assignment. This is most often the case for lambdas.
 *
 * The object is stored as raw bytes and this class declares none of its own
 * special member functions, so its implicit copy assignment copies those
 * bytes. That is what makes it copy assignable and trivially copyable at the
 * same time, and the latter is required for anything passed to a device
 * kernel, whose arguments are transferred by copying their bytes. For the same
 * reason the wrapped object must not be referenced through a pointer member,
 * which would still point to the host object after such a transfer. Both
 * properties are asserted in core/test/base/iterator_factory.cpp.
 *
 * @internal needs to be unconditional; picking the storage with
 *           std::is_copy_assignable runs into a gcc 16 bug:
 *           asking it about a closure type changes std::is_trivially_copyable
 *           from true to false.
 *
 * @tparam T  a trivially copyable and trivially destructible type with a copy
 *            constructor, which includes any lambda that only captures such
 *            types
 */
template <typename T>
class copy_assignable {
    static_assert(std::is_trivially_copyable<T>::value &&
                      std::is_trivially_destructible<T>::value,
                  "The wrapped type needs to be trivially copyable and "
                  "trivially destructible. A lambda qualifies as long as "
                  "everything it captures does, so capture values, indices or "
                  "pointers instead of containers or smart pointers.");

public:
    copy_assignable() = default;

    GKO_ATTRIBUTES copy_assignable(const T& obj)
    {
        ::new (static_cast<void*>(buf)) T(obj);
    }

    template <typename... Args>
    GKO_ATTRIBUTES decltype(auto) operator()(Args&&... args) const
    {
        return get()(std::forward<Args>(args)...);
    }

    GKO_ATTRIBUTES const T& get() const
    {
        return *reinterpret_cast<const T*>(buf);
    }

    GKO_ATTRIBUTES T& get() { return *reinterpret_cast<T*>(buf); }

private:
    //!< the bytes of the wrapped object
    alignas(T) unsigned char buf[sizeof(T)];
};


}  // namespace detail
}  // namespace gko

#endif  // GKO_CORE_BASE_COPY_ASSIGNABLE_HPP_
