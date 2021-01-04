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

#ifndef GKO_CORE_BASE_UTILS_HPP_
#define GKO_CORE_BASE_UTILS_HPP_


#include <ginkgo/core/base/utils.hpp>


#include <memory>
#include <type_traits>


#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {


template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE ValueType checked_load(const ValueType *p,
                                                 IndexType i, IndexType size,
                                                 ValueType sentinel)
{
    return i < size ? p[i] : sentinel;
}


}  // namespace kernels


namespace detail {


template <typename Dest>
struct conversion_sort_helper {};

template <typename ValueType, typename IndexType>
struct conversion_sort_helper<matrix::Csr<ValueType, IndexType>> {
    using mtx_type = matrix::Csr<ValueType, IndexType>;
    template <typename Source>
    static std::unique_ptr<mtx_type> get_sorted_conversion(
        std::shared_ptr<const Executor> &exec, Source *source)
    {
        auto editable_mtx = mtx_type::create(exec);
        as<ConvertibleTo<mtx_type>>(source)->convert_to(lend(editable_mtx));
        editable_mtx->sort_by_column_index();
        return editable_mtx;
    }
};


template <typename Dest, typename Source>
std::unique_ptr<Dest, std::function<void(Dest *)>> convert_to_with_sorting_impl(
    std::shared_ptr<const Executor> &exec, Source *obj, bool skip_sorting)
{
    if (skip_sorting) {
        return copy_and_convert_to<Dest>(exec, obj);
    } else {
        using decay_dest = std::decay_t<Dest>;
        auto sorted_mtx =
            detail::conversion_sort_helper<decay_dest>::get_sorted_conversion(
                exec, obj);
        return {sorted_mtx.release(), std::default_delete<Dest>()};
    }
}

template <typename Dest, typename Source>
std::shared_ptr<Dest> convert_to_with_sorting_impl(
    std::shared_ptr<const Executor> &exec, std::shared_ptr<Source> obj,
    bool skip_sorting)
{
    if (skip_sorting) {
        return copy_and_convert_to<Dest>(exec, obj);
    } else {
        using decay_dest = std::decay_t<Dest>;
        auto sorted_mtx =
            detail::conversion_sort_helper<decay_dest>::get_sorted_conversion(
                exec, obj.get());
        return {std::move(sorted_mtx)};
    }
}


}  // namespace detail


/**
 * @internal
 *
 * Helper function that converts the given matrix to the Dest format with
 * additional sorting if requested.
 *
 * If the given matrix was already sorted, is on the same executor and with a
 * dynamic type of `Dest`, the same pointer is returned with an empty
 * deleter.
 * In all other cases, a new matrix is created, which stores the converted
 * matrix.
 *
 * @tparam Dest  the type to which the object should be converted
 * @tparam Source  the type of the source object
 *
 * @param exec  the executor where the result should be placed
 * @param obj  the source object that should be converted
 * @param skip_sorting  indicator if the resulting matrix should be sorted or
 *                      not
 */
template <typename Dest, typename Source>
std::unique_ptr<Dest, std::function<void(Dest *)>> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, Source *obj, bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<Dest>(exec, obj, skip_sorting);
}

/**
 * @copydoc convert_to_with_sorting(std::shared_ptr<const Executor>,
 * Source *, bool)
 *
 * @note This version adds the const qualifier for the result since the input is
 *       also const
 */
template <typename Dest, typename Source>
std::unique_ptr<const Dest, std::function<void(const Dest *)>>
convert_to_with_sorting(std::shared_ptr<const Executor> exec, const Source *obj,
                        bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<const Dest>(exec, obj,
                                                            skip_sorting);
}

/**
 * @copydoc convert_to_with_sorting(std::shared_ptr<const Executor>,
 * Source *, bool)
 *
 * @note This version has a unique_ptr as the source instead of a plain pointer
 */
template <typename Dest, typename Source>
std::unique_ptr<Dest, std::function<void(Dest *)>> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, const std::unique_ptr<Source> &obj,
    bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<Dest>(exec, obj.get(),
                                                      skip_sorting);
}

/**
 * @internal
 *
 * Helper function that converts the given matrix to the Dest format with
 * additional sorting if requested.
 *
 * If the given matrix was already sorted, is on the same executor and with a
 * dynamic type of `Dest`, the same pointer is returned.
 * In all other cases, a new matrix is created, which stores the converted
 * matrix.
 *
 * @tparam Dest  the type to which the object should be converted
 * @tparam Source  the type of the source object
 *
 * @param exec  the executor where the result should be placed
 * @param obj  the source object that should be converted
 * @param skip_sorting  indicator if the resulting matrix should be sorted or
 *                      not
 */
template <typename Dest, typename Source>
std::shared_ptr<Dest> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, std::shared_ptr<Source> obj,
    bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<Dest>(exec, obj, skip_sorting);
}

/**
 * @copydoc convert_to_with_sorting(std::shared_ptr<const Executor>,
 * std::shared_ptr<Source>, bool)
 *
 * @note This version adds the const qualifier for the result since the input is
 *       also const
 */
template <typename Dest, typename Source>
std::shared_ptr<const Dest> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, std::shared_ptr<const Source> obj,
    bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<const Dest>(exec, obj,
                                                            skip_sorting);
}


}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
