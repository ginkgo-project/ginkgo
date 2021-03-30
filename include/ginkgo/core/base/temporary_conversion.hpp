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
public:
    using pointer = CopyType *;
    using original_pointer = OrigType *;

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
class convert_back_deleter<const CopyType, const OrigType> {
public:
    using pointer = const CopyType *;
    using original_pointer = const OrigType *;
    convert_back_deleter(original_pointer) {}

    void operator()(pointer ptr) const { delete ptr; }
};


/**
 * @internal
 *
 * Helper type that attempts to statically find the dynamic type of a given
 * LinOp from a list of ConversionCandidates and, on the first match, converts
 * it to TargetType with an appropriate convert_back_deleter.
 *
 * @tparam ConversionCandidates  list of potential dynamic types of the input
 *                               object to be checked.
 */
template <typename... ConversionCandidates>
struct conversion_helper {
    /** Dispatch convert_impl with the ConversionCandidates list */
    template <typename TargetType, typename MaybeConstLinOp>
    static std::unique_ptr<TargetType, std::function<void(TargetType *)>>
    convert(MaybeConstLinOp *obj)
    {
        return convert_impl<TargetType, MaybeConstLinOp,
                            ConversionCandidates...>(obj);
    }

    /**
     * Attempts to cast obj from the first ConversionCandidate and convert it to
     * TargetType with a matching convert_back_deleter. If the cast fails,
     * recursively tries the remaining conversion candidates.
     */
    template <typename TargetType, typename MaybeConstLinOp,
              typename FirstCandidate, typename... TrailingCandidates>
    static std::unique_ptr<TargetType, std::function<void(TargetType *)>>
    convert_impl(MaybeConstLinOp *obj)
    {
        // make candidate_type conditionally const based on whether obj is const
        using candidate_type =
            std::conditional_t<std::is_const<MaybeConstLinOp>::value,
                               const FirstCandidate, FirstCandidate>;
        candidate_type *cast_obj{};
        if ((cast_obj = dynamic_cast<candidate_type *>(obj))) {
            // if the cast is successful, obj is of dynamic type candidate_type
            // so we can convert from this type to TargetType
            auto converted = TargetType::create(obj->get_executor());
            cast_obj->convert_to(converted.get());
            // Make sure ConvertibleTo<TargetType> is available and symmetric
            static_assert(
                std::is_base_of<ConvertibleTo<std::remove_cv_t<TargetType>>,
                                FirstCandidate>::value,
                "ConvertibleTo not implemented");
            static_assert(std::is_base_of<ConvertibleTo<FirstCandidate>,
                                          TargetType>::value,
                          "ConvertibleTo not symmetric");
            return {converted.release(),
                    convert_back_deleter<TargetType, candidate_type>{cast_obj}};
        } else {
            // else try the remaining candidates
            return conversion_helper<TrailingCandidates...>::template convert<
                TargetType>(obj);
        }
    }
};

template <>
struct conversion_helper<> {
    template <typename T, typename MaybeConstLinOp>
    static std::unique_ptr<T, std::function<void(T *)>> convert(
        MaybeConstLinOp *obj)
    {
        // return nullptr if no previous candidates matched
        return {nullptr, null_deleter<T>{}};
    }
};


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
public:
    using value_type = T;
    using pointer = T *;
    using lin_op_type =
        std::conditional_t<std::is_const<T>::value, const LinOp, LinOp>;

    /**
     * Create a temporary conversion for a non-temporary LinOp.
     *
     * @tparam ConversionCandidates  list of potential dynamic types of ptr to
     *                               try out for converting ptr to type T.
     */
    template <typename... ConversionCandidates>
    static temporary_conversion create(lin_op_type *ptr)
    {
        T *cast_ptr{};
        if ((cast_ptr = dynamic_cast<T *>(ptr))) {
            return handle_type{cast_ptr, null_deleter<T>{}};
        } else {
            return conversion_helper<ConversionCandidates...>::template convert<
                T>(ptr);
        }
    }

    /**
     * Returns the object held by temporary_conversion.
     *
     * @return the object held by temporary_conversion
     */
    T *get() const { return handle_.get(); }

    /**
     * Calls a method on the underlying object.
     *
     * @return the underlying object
     */
    T *operator->() const { return handle_.get(); }

    /**
     * Returns if the conversion was successful.
     */
    explicit operator bool() { return static_cast<bool>(handle_); }

private:
    // std::function deleter allows to decide the (type of) deleter at
    // runtime
    using handle_type = std::unique_ptr<T, std::function<void(T *)>>;

    temporary_conversion(handle_type handle) : handle_{std::move(handle)} {}

    handle_type handle_;
};


}  // namespace detail
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_TEMPORARY_CONVERSION_HPP_
