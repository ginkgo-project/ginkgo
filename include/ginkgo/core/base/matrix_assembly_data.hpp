// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_MATRIX_ASSEMBLY_DATA_HPP_
#define GKO_PUBLIC_CORE_BASE_MATRIX_ASSEMBLY_DATA_HPP_


#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <unordered_map>


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
namespace detail {


template <typename IndexType>
struct symbolic_nonzero_hash {
    symbolic_nonzero_hash() = default;

    explicit symbolic_nonzero_hash(size_type num_cols) noexcept
        : num_cols_{num_cols}
    {}

    std::size_t operator()(std::pair<IndexType, IndexType> nnz) const noexcept
    {
        return static_cast<std::size_t>(nnz.first) * num_cols_ + nnz.second;
    }

    size_type num_cols_;
};


}  // namespace detail


/**
 * This structure is used as an intermediate type to assemble a sparse matrix.
 *
 * The matrix is stored as a set of nonzero elements, where each element is
 * a triplet of the form (row_index, column_index, value).
 *
 * New values can be added by using the matrix_assembly_data::add_value or
 * matrix_assembly_data::set_value
 *
 * @tparam ValueType  type of matrix values stored in the structure
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class matrix_assembly_data {
public:
    using value_type = ValueType;
    using index_type = IndexType;

    explicit matrix_assembly_data(dim<2> size)
        : size_{size},
          nonzeros_(0, detail::symbolic_nonzero_hash<index_type>(size_[1]))
    {}

    /**
     * Sets the matrix value at (row, col).
     * If there is an existing value, it will be set to the sum of the
     * existing and new value, otherwise the value will be inserted.
     *
     * @param row  the row where the value should be added
     * @param col  the column where the value should be added
     * @param val  the value to be added to (row, col)
     */
    void add_value(index_type row, index_type col, value_type val)
    {
        auto ind = std::make_pair(row, col);
        nonzeros_[ind] += val;
    }

    /**
     * Sets the matrix value at (row, col).
     * If there is an existing value, it will be overwritten by the new value.
     *
     * @param row  the row index
     * @param col  the column index
     * @param val  the value to be written to (row, col)
     */
    void set_value(index_type row, index_type col, value_type val)
    {
        auto ind = std::make_pair(row, col);
        nonzeros_[ind] = val;
    }

    /**
     * Gets the matrix value at (row, col).
     *
     * @param row  the row index
     * @param col  the column index
     * @return the value at (row, col) or 0 if it doesn't exist.
     */
    value_type get_value(index_type row, index_type col)
    {
        const auto it = nonzeros_.find(std::make_pair(row, col));
        if (it == nonzeros_.end()) {
            return zero<value_type>();
        } else {
            return it->second;
        }
    }

    /**
     * Returns true iff the matrix contains an entry at (row, col).
     *
     * @param row  the row index
     * @param col  the column index
     * @return true if the value at (row, col) exists, false otherwise
     */
    bool contains(index_type row, index_type col)
    {
        return nonzeros_.find(std::make_pair(row, col)) != nonzeros_.end();
    }

    /** @return the dimensions of the matrix being assembled */
    dim<2> get_size() const noexcept { return size_; }

    /** @return the number of non-zeros in the (partially) assembled matrix */
    size_type get_num_stored_elements() const noexcept
    {
        return nonzeros_.size();
    }

    /**
     * @return a matrix_data instance containing the assembled non-zeros in
     * row-major order to be used by all matrix formats.
     */
    matrix_data<ValueType, IndexType> get_ordered_data() const
    {
        using output_type = matrix_data<ValueType, IndexType>;
        using nonzero_type = typename output_type::nonzero_type;
        using entry_type =
            std::pair<std::pair<index_type, index_type>, value_type>;
        output_type data{size_};
        data.nonzeros.reserve(nonzeros_.size());
        std::transform(nonzeros_.begin(), nonzeros_.end(),
                       std::back_inserter(data.nonzeros), [](entry_type entry) {
                           return nonzero_type{entry.first.first,
                                               entry.first.second,
                                               entry.second};
                       });
        data.sort_row_major();
        return data;
    }

private:
    /**
     * Size of the matrix.
     */
    dim<2> size_;

    /**
     * An unordered map storing the non-zeros of the matrix.
     *
     * The keys of the elements in the map are the row index and the column
     * index of a matrix element, and its value is the value at that
     * position.
     */
    std::unordered_map<std::pair<index_type, index_type>, value_type,
                       detail::symbolic_nonzero_hash<index_type>>
        nonzeros_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_MATRIX_ASSEMBLY_DATA_HPP_
