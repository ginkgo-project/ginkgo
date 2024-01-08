// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_COPY_ASSIGNABLE_HPP_
#define GKO_CORE_BASE_COPY_ASSIGNABLE_HPP_


#include <vector>


namespace gko {
namespace detail {


template <typename T, typename = void>
class copy_assignable;


/**
 * Helper class to make a type copy assignable.
 *
 * This class wraps an object of a type that has a copy constructor, but not
 * a copy assignment. This is most often the case for lambdas. The wrapped
 * object can then be copy assigned, by relying on the copy constructor.
 *
 * @tparam T  type with a copy constructor
 */
template <typename T>
class copy_assignable<
    T, typename std::enable_if<std::is_copy_constructible<T>::value>::type> {
public:
    copy_assignable() = default;

    copy_assignable(const copy_assignable& other)
    {
        if (this != &other) {
            *this = other;
        }
    }

    copy_assignable(copy_assignable&& other) noexcept
    {
        if (this != &other) {
            *this = std::move(other);
        }
    }

    copy_assignable(const T& obj) : obj_{new (buf)(T)(obj)} {}

    copy_assignable(T&& obj) : obj_{new (buf)(T)(std::move(obj))} {}

    copy_assignable& operator=(const copy_assignable& other)
    {
        if (this != &other) {
            if (obj_) {
                obj_->~T();
            }
            obj_ = new (buf)(T)(*other.obj_);
        }
        return *this;
    }

    copy_assignable& operator=(copy_assignable&& other) noexcept
    {
        if (this != &other) {
            if (obj_) {
                obj_->~T();
            }
            obj_ = new (buf)(T)(std::move(*other.obj_));
        }
        return *this;
    }

    ~copy_assignable()
    {
        if (obj_) {
            obj_->~T();
        }
    }

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const
    {
        return (*obj_)(std::forward<Args>(args)...);
    }

    T const& get() const { return *obj_; }

    T& get() { return *obj_; }

private:
    //!< Store wrapped object on the stack, should use std::optional in c++17
    T* obj_{};
    alignas(T) unsigned char buf[sizeof(T)];
};


}  // namespace detail
}  // namespace gko

#endif  // GKO_CORE_BASE_COPY_ASSIGNABLE_HPP_
