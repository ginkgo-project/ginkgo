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

#ifndef GKO_PUBLIC_CORE_BASE_NATIVE_TYPE_HPP_
#define GKO_PUBLIC_CORE_BASE_NATIVE_TYPE_HPP_

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
/**
 * @brief This namespace contains views on the internal data of selected
 *        Ginkgo types.
 *
 * @ingroup layout
 */
namespace layout {

/**
 * A view of gko::device_matrix_data.
 *
 * @tparam ValueType  The value type.
 * @tparam IndexType  The index type.
 * @tparam array_mapper  The mapping for gko::array
 *
 * @ingroup layout
 */
template <typename ValueType, typename IndexType, typename array_mapper>
struct device_matrix_data {
    using index_array = typename array_mapper::template type<IndexType>;
    using value_array = typename array_mapper::template type<ValueType>;

    static device_matrix_data map(std::shared_ptr<const Executor> exec,
                                  size_type num_elems, IndexType* row_idxs,
                                  IndexType* col_idxs, ValueType* values)
    {
        return {
            array_mapper::map(gko::make_array_view(exec, num_elems, row_idxs)),
            array_mapper::map(gko::make_array_view(exec, num_elems, col_idxs)),
            array_mapper::map(gko::make_array_view(exec, num_elems, values))};
    }

    index_array row_idxs;
    index_array col_idxs;
    value_array values;
};


}  // namespace layout


namespace detail {


/**
 * Mapper for Ginkgo type T.
 *
 * @internal Has to be specialized for any supported type. Usually that
 *           includes specializing for T and const T.
 *
 * @tparam T  The Ginkgo type to map
 * @tparam am  The mapper for gko::array
 * @tparam dm  The mapper for gko::matrix::Dense
 */
template <typename T, typename am, typename dm>
struct native;


/**
 * Mapper for Ginkgo array of a specific value type
 *
 * @tparam ValueType  Value type of the array
 * @tparam am  The value type independent mapper for gko::array
 * @tparam dm  Unused
 */
template <typename ValueType, typename am, typename dm>
struct native<array<ValueType>, am, dm> : public am {
    using type = typename am::template type<ValueType>;
};

/**
 * Mapper for a const Ginkgo array of a specific value type
 *
 * @tparam ValueType  Value type of the array
 * @tparam am  The value type independent mapper for gko::array
 * @tparam dm  Unused
 */
template <typename ValueType, typename am, typename dm>
struct native<const array<ValueType>, am, dm> : public am {
    using type = typename am::template type<const ValueType>;
};


/**
 * Mapper for Ginkgo dense matrix of a specific value type
 *
 * @tparam ValueType  Value type of the array
 * @tparam am  unused
 * @tparam dm  The value type independent mapper for gko::matrix::Dense
 */
template <typename ValueType, typename am, typename dm>
struct native<matrix::Dense<ValueType>, am, dm> : public dm {
    using type = typename dm::template type<ValueType>;
};

/**
 * Mapper for a const Ginkgo dense matrix of a specific value type
 *
 * @tparam ValueType  Value type of the array
 * @tparam am  unused
 * @tparam dm  The value type independent mapper for gko::matrix::Dense
 */
template <typename ValueType, typename am, typename dm>
struct native<const matrix::Dense<ValueType>, am, dm> : public dm {
    using type = typename dm::template type<const ValueType>;
};


template <typename ValueType, typename IndexType, typename array_mapper,
          typename dense_mapper>
struct native<device_matrix_data<ValueType, IndexType>, array_mapper,
              dense_mapper> {
    using type = layout::device_matrix_data<ValueType, IndexType, array_mapper>;

    static type map(device_matrix_data<ValueType, IndexType>& md)
    {
        return type::map(md.get_executor(), md.get_num_elems(),
                         md.get_row_idxs(), md.get_col_idxs(), md.get_values());
    }
};

template <typename ValueType, typename IndexType, typename array_mapper,
          typename dense_mapper>
struct native<const device_matrix_data<ValueType, IndexType>, array_mapper,
              dense_mapper> {
    using type = layout::device_matrix_data<ValueType, IndexType, array_mapper>;

    static type map(const device_matrix_data<ValueType, IndexType>& md)
    {
        return type::map(md.get_executor(), md.get_num_elems(),
                         const_cast<IndexType>(md.get_const_row_idxs()),
                         const_cast<IndexType>(md.get_const_col_idxs()),
                         const_cast<ValueType>(md.get_values()));
    }
};


}  // namespace detail


/**
 * Class for mapping Ginkgo types to types that are native to other frameworks.
 *
 * This uses mapping types for gko::array and gko::matrix::Dense to build
 * up more complex types. The description of the supported types are part
 * of the layout namespace.
 *
 * Array mapper concept:
 * A class AM satisfies the array mapper concept if it provides the type alias
 * - `AM::template type<T>`
 * and the functions
 * - `static type<T> map(gko::array<T>&)`
 * - `static type<const T> map(const gko::array<T>&)`
 * - `static type<T> map(gko::array<T>&&)`
 * The mapped object should be a non-owning view of the passed in array.
 *
 * Dense mapper concept:
 * A class DM satisfies the dense mapper concept if it provides the type alias
 * - `DM::template type<T>`
 * and the functions
 * - `static type<T> map(gko::matrix::Dense<T>&)`
 * - `static type<const T> map(const gko::matrix::Dense<T>&)`
 * - `static type<T> map(gko::matrix::Dense<T>&&)`
 * The mapped object should be a non-owning view of the passed in matrix.
 *
 * Usage:
 * Provided some implementations of the array and dense mapper concept, a
 * concrete mapper class can be created by
 * ```
 * using my_mapper = gko::native<my_array_mapper, my_dense_mapper>;
 * ```
 * This mapper can than be used to map ginkgo objects to other representations
 * that are using the replacements for array and dense as defined by the
 * `my_array_mapper` and `my_dense_mapper`:
 * ```
 * gko::device_matrix_data<> md(exec, size);
 * auto my_md = my_mapper::map(md);
 * // handle my_md.row_idxs, my_md.col_idxs, my_md.values
 * ```
 *
 * @tparam array_mapper
 * @tparam dense_mapper
 */
template <typename array_mapper, typename dense_mapper>
struct native {
private:
    template <typename T>
    using sanitize = typename std::remove_pointer_t<std::remove_reference_t<T>>;

    template <typename T>
    using native_impl =
        typename detail::native<sanitize<T>, array_mapper, dense_mapper>;

public:
    template <typename T>
    using type = typename native_impl<T>::type;

    template <typename T>
    static type<T> map(T&& input)
    {
        return native_impl<T>::map(input);
    }

    template <typename T>
    static type<T> map(T* input)
    {
        return map(*input);
    }

    template <typename T>
    static type<T> map(std::shared_ptr<T>& input)
    {
        return map(*input);
    }

    template <typename T>
    static type<T> map(std::unique_ptr<T>& input)
    {
        return map(*input);
    }
};

}  // namespace gko
#endif  // GKO_PUBLIC_CORE_BASE_NATIVE_TYPE_HPP_
