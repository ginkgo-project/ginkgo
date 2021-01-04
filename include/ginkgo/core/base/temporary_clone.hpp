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

#ifndef GKO_PUBLIC_CORE_BASE_TEMPORARY_CLONE_HPP_
#define GKO_PUBLIC_CORE_BASE_TEMPORARY_CLONE_HPP_


#include <functional>
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/utils_helper.hpp>


namespace gko {
namespace detail {


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


template <typename T>
struct temporary_clone_helper {
    static std::unique_ptr<T> create(std::shared_ptr<const Executor> exec,
                                     T *ptr)
    {
        return gko::clone(std::move(exec), ptr);
    }
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
        if (ptr->get_executor()->memory_accessible(exec)) {
            // just use the object we already have
            handle_ = handle_type(ptr, null_deleter<T>());
        } else {
            // clone the object to the new executor and make sure it's copied
            // back before we delete it
            handle_ = handle_type(
                temporary_clone_helper<T>::create(std::move(exec), ptr)
                    .release(),
                copy_back_deleter<T>(ptr));
        }
    }

    /**
     * Returns the object held by temporary_clone.
     *
     * @return the object held by temporary_clone
     */
    T *get() const { return handle_.get(); }

    /**
     * Calls a method on the underlying object.
     *
     * @return the underlying object
     */
    T *operator->() const { return handle_.get(); }

private:
    // std::function deleter allows to decide the (type of) deleter at runtime
    using handle_type = std::unique_ptr<T, std::function<void(T *)>>;

    handle_type handle_;
};


}  // namespace detail


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
detail::temporary_clone<T> make_temporary_clone(
    std::shared_ptr<const Executor> exec, T *ptr)
{
    return detail::temporary_clone<T>(std::move(exec), ptr);
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_TEMPORARY_CLONE_HPP_
