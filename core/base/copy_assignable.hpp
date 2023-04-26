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
    copy_assignable() : obj_{{}} {}
    copy_assignable(const copy_assignable& other) = default;
    copy_assignable(copy_assignable&& other) noexcept = default;

    copy_assignable(const T& obj) : obj_{obj} {}
    copy_assignable(T&& obj) : obj_{std::move(obj)} {}

    copy_assignable& operator=(const copy_assignable& other)
    {
        obj_.clear();
        obj_.emplace_back(other.get());
        return *this;
    }
    copy_assignable& operator=(copy_assignable&& other) noexcept = default;

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const
    {
        return obj_[0](std::forward<Args>(args)...);
    }

    T const& get() const { return obj_[0]; }
    T& get() { return obj_[0]; }

private:
    //!< Store wrapped object in a container that has an emplace function
    std::vector<T> obj_;
};


}  // namespace detail
}  // namespace gko

#endif  // GKO_CORE_BASE_COPY_ASSIGNABLE_HPP_
