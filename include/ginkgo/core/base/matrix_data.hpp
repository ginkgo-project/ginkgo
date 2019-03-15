/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, Karlsruhe Institute of Technology
Copyright (c) 2017-2019, Universitat Jaume I
Copyright (c) 2017-2019, University of Tennessee
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

#ifndef GKO_CORE_BASE_MATRIX_DATA_HPP_
#define GKO_CORE_BASE_MATRIX_DATA_HPP_


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>


#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>


namespace gko {


namespace detail {


// internal structure used to get around explicit constructors in std::tuple
template <typename ValueType, typename IndexType>
struct input_triple {
    IndexType row;
    IndexType col;
    ValueType val;
};


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex<ValueType>(), ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return dist(gen);
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex<ValueType>(), ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return ValueType(dist(gen), dist(gen));
}


}  // namespace detail


/**
 * This structure is used as an intermediate data type to store a sparse matrix.
 *
 * The matrix is stored as a sequence of nonzero elements, where each element is
 * a triple of the form (row_index, column_index, value).
 *
 * @note All Ginkgo functions returning such a structure will return the
 *       nonzeros sorted in row-major order.
 * @note All Ginkgo functions that take this structure as input expect that the
 *       nonzeros are sorted in row-major order.
 * @note This structure is not optimized for usual access patterns and it can
 *       only exist on the CPU. Thus, it should only be used for utility
 *       functions which do not have to be optimized for performance.
 *
 * @tparam ValueType  type of matrix values stored in the structure
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <typename ValueType = default_precision, typename IndexType = int32>
struct matrix_data {
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Type used to store nonzeros.
     */
    struct nonzero_type {
        nonzero_type() = default;

        nonzero_type(index_type r, index_type c, value_type v)
            : row(r), column(c), value(v)
        {}

#define GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(_op)                \
    bool operator _op(const nonzero_type &other) const          \
    {                                                           \
        return std::tie(this->row, this->column, this->value)   \
            _op std::tie(other.row, other.column, other.value); \
    }

        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(==);
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(!=);
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(<);
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(>);
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(<=);
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(>=);

#undef GKO_DEFINE_DEFAULT_COMPARE_OPERATOR

        index_type row;
        index_type column;
        value_type value;
    };

    /**
     * Initializes a matrix filled with the specified value.
     *
     * @param size_  dimensions of the matrix
     * @param value  value used to fill the elements of the matrix
     */
    matrix_data(dim<2> size_ = dim<2>{}, ValueType value = zero<ValueType>())
        : size{size_}
    {
        if (value == zero<ValueType>()) {
            return;
        }
        for (size_type row = 0; row < size[0]; ++row) {
            for (size_type col = 0; col < size[1]; ++col) {
                nonzeros.emplace_back(row, col, value);
            }
        }
    }

    /**
     * Initializes a matrix with random values from the specified distribution.
     *
     * @tparam RandomDistribution  random distribution type
     * @tparam RandomEngine  random engine type
     *
     * @param size_  dimensions of the matrix
     * @param dist  random distribution of the elements of the matrix
     * @param engine  random engine used to generate random values
     */
    template <typename RandomDistribution, typename RandomEngine>
    matrix_data(dim<2> size_, RandomDistribution &&dist, RandomEngine &&engine)
        : size{size_}
    {
        for (size_type row = 0; row < size[0]; ++row) {
            for (size_type col = 0; col < size[1]; ++col) {
                const auto value =
                    detail::get_rand_value<ValueType>(dist, engine);
                if (value != zero<ValueType>()) {
                    nonzeros.emplace_back(row, col, value);
                }
            }
        }
    }

    /**
     * List-initializes the structure from a matrix of values.
     *
     * @param values  a 2D braced-init-list of matrix values.
     */
    matrix_data(std::initializer_list<std::initializer_list<ValueType>> values)
        : size{values.size(), 0}
    {
        for (size_type row = 0; row < values.size(); ++row) {
            const auto row_data = begin(values)[row];
            size[1] = std::max(size[1], row_data.size());
            for (size_type col = 0; col < row_data.size(); ++col) {
                const auto &val = begin(row_data)[col];
                if (val != zero<ValueType>()) {
                    nonzeros.emplace_back(row, col, val);
                }
            }
        }
    }

    /**
     * Initializes the structure from a list of nonzeros.
     *
     * @param size_  dimensions of the matrix
     * @param nonzeros_  list of nonzero elements
     */
    matrix_data(
        dim<2> size_,
        std::initializer_list<detail::input_triple<ValueType, IndexType>>
            nonzeros_)
        : size{size_}, nonzeros()
    {
        nonzeros.reserve(nonzeros_.size());
        for (const auto &elem : nonzeros_) {
            nonzeros.emplace_back(elem.row, elem.col, elem.val);
        }
    }

    /**
     * Initializes a matrix out of a matrix block via duplication.
     *
     * @param size  size of the block-matrix (in blocks)
     * @param diag_block  matrix block used to fill the complete matrix
     */
    matrix_data(dim<2> size_, const matrix_data &block)
        : size{size_ * block.size}
    {
        nonzeros.reserve(size_[0] * size_[1] * block.nonzeros.size());
        for (size_type row = 0; row < size_[0]; ++row) {
            for (size_type col = 0; col < size_[1]; ++col) {
                for (const auto &elem : block.nonzeros) {
                    nonzeros.emplace_back(row * block.size[0] + elem.row,
                                          col * block.size[1] + elem.column,
                                          elem.value);
                }
            }
        }
        this->ensure_row_major_order();
    }

    /**
     * Initializes a matrix from a range.
     *
     * @tparam Accessor  accessor type of the input range
     *
     * @param data  range used to initialize the matrix
     */
    template <typename Accessor>
    matrix_data(const range<Accessor> &data)
        : size{data.length(0), data.length(1)}
    {
        for (gko::size_type row = 0; row < size[0]; ++row) {
            for (gko::size_type col = 0; col < size[1]; ++col) {
                if (data(row, col) != zero<ValueType>()) {
                    nonzeros.emplace_back(row, col, data(row, col));
                }
            }
        }
    }

    /**
     * Initializes a diagonal matrix.
     *
     * @param size_  dimensions of the matrix
     * @param value  value used to fill the elements of the matrix
     *
     * @return the diagonal matrix
     */
    static matrix_data diag(dim<2> size_, ValueType value)
    {
        matrix_data res(size_);
        if (value != zero<ValueType>()) {
            const auto num_nnz = std::min(size_[0], size_[1]);
            res.nonzeros.reserve(num_nnz);
            for (size_type i = 0; i < num_nnz; ++i) {
                res.nonzeros.emplace_back(i, i, value);
            }
        }
        return res;
    }

    /**
     * Initializes a diagonal matrix using a list of diagonal elements.
     *
     * @param size_  dimensions of the matrix
     * @param nonzeros_  list of diagonal elements
     *
     * @return the diagonal matrix
     */
    static matrix_data diag(dim<2> size_,
                            std::initializer_list<ValueType> nonzeros_)
    {
        matrix_data res(size_);
        res.nonzeros.reserve(nonzeros_.size());
        int pos = 0;
        for (auto value : nonzeros_) {
            res.nonzeros.emplace_back(pos, pos, value);
            ++pos;
        }
        return res;
    }

    /**
     * Initializes a block-diagonal matrix.
     *
     * @param size_  the size of the matrix
     * @param diag_block  matrix used to fill diagonal blocks
     *
     * @return the block-diagonal matrix
     */
    static matrix_data diag(dim<2> size_, const matrix_data &block)
    {
        matrix_data res(size_ * block.size);
        const auto num_blocks = std::min(size_[0], size_[1]);
        res.nonzeros.reserve(num_blocks * block.nonzeros.size());
        for (size_type b = 0; b < num_blocks; ++b) {
            for (const auto &elem : block.nonzeros) {
                res.nonzeros.emplace_back(b * block.size[0] + elem.row,
                                          b * block.size[1] + elem.column,
                                          elem.value);
            }
        }
        return res;
    }

    /**
     * Initializes a block-diagonal matrix from a list of diagonal blocks.
     *
     * @tparam ForwardIterator  type of list iterator
     *
     * @param begin  the first iterator of the list
     * @param end  the last iterator of the list
     *
     * @return the block-diagonal matrix with diagonal blocks set to the blocks
     *         between begin (inclusive) and end (exclusive)
     */
    template <typename ForwardIterator>
    static matrix_data diag(ForwardIterator begin, ForwardIterator end)
    {
        matrix_data res(std::accumulate(
            begin, end, dim<2>{}, [](dim<2> s, const matrix_data &d) {
                return dim<2>{s[0] + d.size[0], s[1] + d.size[1]};
            }));

        size_type row_offset{};
        size_type col_offset{};
        for (auto it = begin; it != end; ++it) {
            for (const auto &elem : it->nonzeros) {
                res.nonzeros.emplace_back(row_offset + elem.row,
                                          col_offset + elem.column, elem.value);
            }
            row_offset += it->size[0];
            col_offset += it->size[1];
        }

        return res;
    }

    /**
     * Initializes a block-diagonal matrix from a list of diagonal blocks.
     *
     * @param blocks  a list of blocks to initialize from
     *
     * @return the block-diagonal matrix with diagonal blocks set to the blocks
     *         passed in blocks
     */
    static matrix_data diag(std::initializer_list<matrix_data> blocks)
    {
        return diag(begin(blocks), end(blocks));
    }

    /**
     * Initializes a random dense matrix with a specific condition number.
     *
     * The matrix is generated by applying a series of random Hausholder
     * reflectors to a diagonal matrix with diagonal entries uniformly
     * distributed between `sqrt(condition_number)` and
     * `1/sqrt(condition_number)`.
     *
     * @tparam RandomDistribution  the type of the random distribution
     * @tparam RandomEngine  the type of the random engine
     *
     * @param size  number of rows and columns of the matrix
     * @param condition_number  condition number of the matrix
     * @param dist  random distribution used to generate reflectors
     * @param engine  random engine used to generate reflectors
     * @param num_reflectors  number of reflectors to apply from each side
     *
     * @return the dense matrix with the specified condition number
     */
    template <typename RandomDistribution, typename RandomEngine>
    static matrix_data cond(size_type size,
                            remove_complex<ValueType> condition_number,
                            RandomDistribution &&dist, RandomEngine &&engine,
                            size_type num_reflectors)
    {
        using range = range<accessor::row_major<ValueType, 2>>;
        std::vector<ValueType> mtx_data(size * size, zero<ValueType>());
        std::vector<ValueType> ref_data(size);
        std::vector<ValueType> work(size);
        range matrix(mtx_data.data(), size, size, size);
        range reflector(ref_data.data(), size, 1u, 1u);

        initialize_diag_with_cond(condition_number, matrix);
        for (size_type i = 0; i < num_reflectors; ++i) {
            generate_random_reflector(dist, engine, reflector);
            reflect_domain(reflector, matrix, work.data());
            generate_random_reflector(dist, engine, reflector);
            reflect_range(reflector, matrix, work.data());
        }
        return matrix;
    }

    /**
     * Initializes a random dense matrix with a specific condition number.
     *
     * The matrix is generated by applying a series of random Hausholder
     * reflectors to a diagonal matrix with diagonal entries uniformly
     * distributed between `sqrt(condition_number)` and
     * `1/sqrt(condition_number)`.
     *
     * This version of the function applies `size - 1` reflectors to each side
     * of the diagonal matrix.
     *
     * @tparam RandomDistribution  the type of the random distribution
     * @tparam RandomEngine  the type of the random engine
     *
     * @param size  number of rows and columns of the matrix
     * @param condition_number  condition number of the matrix
     * @param dist  random distribution used to generate reflectors
     * @param engine  random engine used to generate reflectors
     *
     * @return the dense matrix with the specified condition number
     */
    template <typename RandomDistribution, typename RandomEngine>
    static matrix_data cond(size_type size,
                            remove_complex<ValueType> condition_number,
                            RandomDistribution &&dist, RandomEngine &&engine)
    {
        return cond(size, condition_number,
                    std::forward<RandomDistribution>(dist),
                    std::forward<RandomEngine>(engine), size - 1);
    }

    /**
     * Size of the matrix.
     */
    dim<2> size;

    /**
     * A vector of tuples storing the non-zeros of the matrix.
     *
     * The first two elements of the tuple are the row index and the column
     * index of a matrix element, and its third element is the value at that
     * position.
     */
    std::vector<nonzero_type> nonzeros;

    /**
     * Sorts the nonzero vector so the values follow row-major order.
     */
    void ensure_row_major_order()
    {
        std::sort(
            begin(nonzeros), end(nonzeros), [](nonzero_type x, nonzero_type y) {
                return std::tie(x.row, x.column) < std::tie(y.row, y.column);
            });
    }

private:
    template <typename Accessor>
    static void initialize_diag_with_cond(
        remove_complex<ValueType> condition_number,
        const range<Accessor> &matrix)
    {
        using sigma_type = remove_complex<ValueType>;
        const auto size = matrix.length(0);
        const auto min_sigma = one(condition_number) / sqrt(condition_number);
        const auto max_sigma = sqrt(condition_number);

        matrix = zero(matrix);
        for (gko::size_type i = 0; i < size; ++i) {
            matrix(i, i) = max_sigma * static_cast<sigma_type>(size - i - 1) /
                               static_cast<sigma_type>(size - 1) +
                           min_sigma * static_cast<sigma_type>(i) /
                               static_cast<sigma_type>(size - 1);
        }
    }

    template <typename RandomDistribution, typename RandomEngine,
              typename Accessor>
    static void generate_random_reflector(RandomDistribution &&dist,
                                          RandomEngine &&engine,
                                          const range<Accessor> &reflector)
    {
        for (gko::size_type i = 0; i < reflector.length(0); ++i) {
            reflector(i, 0) = detail::get_rand_value<ValueType>(dist, engine);
        }
    }

    template <typename Accessor>
    static void reflect_domain(const range<Accessor> &reflector,
                               const range<Accessor> &matrix,
                               ValueType *work_data)
    {
        const auto two = one<ValueType>() + one<ValueType>();
        range<accessor::row_major<ValueType, 2>> work(work_data,
                                                      matrix.length(0), 1u, 1u);
        work = mmul(matrix, reflector);
        const auto ct_reflector = conj(transpose(reflector));
        const auto scale = two / mmul(ct_reflector, reflector)(0, 0);
        matrix = matrix - scale * mmul(work, ct_reflector);
    }

    template <typename Accessor>
    static void reflect_range(const range<Accessor> &reflector,
                              const range<Accessor> &matrix,
                              ValueType *work_data)
    {
        const auto two = one<ValueType>() + one<ValueType>();
        range<accessor::row_major<ValueType, 2>> work(
            work_data, 1u, matrix.length(0), matrix.length(0));
        const auto ct_reflector = conj(transpose(reflector));
        work = mmul(ct_reflector, matrix);
        const auto scale = two / mmul(ct_reflector, reflector)(0, 0);
        matrix = matrix - scale * mmul(reflector, work);
    }
};


}  // namespace gko


#endif  // GKO_CORE_BASE_MATRIX_DATA_HPP_
