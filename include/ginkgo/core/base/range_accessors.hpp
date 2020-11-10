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

#ifndef GKO_PUBLIC_CORE_BASE_RANGE_ACCESSORS_HPP_
#define GKO_PUBLIC_CORE_BASE_RANGE_ACCESSORS_HPP_


#include <array>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/range_accessor_helper.hpp>
#include <ginkgo/core/base/range_accessor_references.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
/**
 * @brief The accessor namespace.
 *
 * @ingroup accessor
 */
namespace accessor {


/**
 * A row_major accessor is a bridge between a range and the row-major memory
 * layout.
 *
 * You should never try to explicitly create an instance of this accessor.
 * Instead, supply it as a template parameter to a range, and pass the
 * constructor parameters for this class to the range (it will forward it to
 * this class).
 *
 * @warning The current implementation is incomplete, and only allows for
 *          2-dimensional ranges.
 *
 * @tparam ValueType  type of values this accessor returns
 * @tparam Dimensionality  number of dimensions of this accessor (has to be
 * 2)
 */
template <typename ValueType, size_type Dimensionality>
class row_major {
public:
    friend class range<row_major>;

    static_assert(Dimensionality == 2,
                  "This accessor is only implemented for matrices");

    /**
     * Type of values returned by the accessor.
     */
    using value_type = ValueType;

    /**
     * Type of underlying data storage.
     */
    using data_type = value_type *;

    /**
     * Number of dimensions of the accessor.
     */
    static constexpr size_type dimensionality = 2;

    using const_accessor = row_major<const ValueType, Dimensionality>;

protected:
    /**
     * Creates a row_major accessor.
     *
     * @param data  pointer to the block of memory containing the data
     * @param num_row  number of rows of the accessor
     * @param num_cols  number of columns of the accessor
     * @param stride  distance (in elements) between starting positions of
     *                consecutive rows (i.e. `data + i * stride` points to
     *                the `i`-th row)
     */
    GKO_ATTRIBUTES constexpr explicit row_major(data_type data,
                                                size_type num_rows,
                                                size_type num_cols,
                                                size_type stride)
        : data{data}, lengths{num_rows, num_cols}, stride{stride}
    {}

public:
    /**
     * Creates a row_major range which contains a read-only version of the
     * current accessor.
     *
     * @returns  a row major range which is read-only.
     */
    GKO_ATTRIBUTES constexpr range<const_accessor> to_const() const
    {
        return range<const_accessor>{data, lengths[0], lengths[1], stride};
    }

    /**
     * Returns the data element at position (row, col)
     *
     * @param row  row index
     * @param col  column index
     *
     * @return data element at (row, col)
     */
    GKO_ATTRIBUTES constexpr value_type &operator()(size_type row,
                                                    size_type col) const
    {
        return GKO_ASSERT(row < lengths[0]), GKO_ASSERT(col < lengths[1]),
               data[row * stride + col];
    }

    /**
     * Returns the sub-range spanning the range (rows, cols)
     *
     * @param rows  row span
     * @param cols  column span
     *
     * @return sub-range spanning the range (rows, cols)
     */
    GKO_ATTRIBUTES constexpr range<row_major> operator()(const span &rows,
                                                         const span &cols) const
    {
        return GKO_ASSERT(rows.is_valid()), GKO_ASSERT(cols.is_valid()),
               GKO_ASSERT(rows <= span{lengths[0]}),
               GKO_ASSERT(cols <= span{lengths[1]}),
               range<row_major>(data + rows.begin * stride + cols.begin,
                                rows.end - rows.begin, cols.end - cols.begin,
                                stride);
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @return length in dimension `dimension`
     */
    GKO_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return dimension < 2 ? lengths[dimension] : 1;
    }

    /**
     * Copies data from another accessor
     *
     * @warning Do not use this function since it is not optimized for a
     *          specific executor. It will always be performed sequentially.
     *          Please write an optimized version (adjusted to the architecture)
     *          by iterating through the values yourself.
     *
     * @tparam OtherAccessor  type of the other accessor
     *
     * @param other  other accessor
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        for (size_type i = 0; i < lengths[0]; ++i) {
            for (size_type j = 0; j < lengths[1]; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
    }

    /**
     * Reference to the underlying data.
     */
    const data_type data;

    /**
     * An array of dimension sizes.
     */
    const std::array<const size_type, dimensionality> lengths;

    /**
     * Distance between consecutive rows.
     */
    const size_type stride;
};


/**
 * The reduced_row_major class allows a storage format that is different from
 * the arithmetic format (which is returned from the brace operator).
 * As storage, the StorageType is used.
 *
 * This accessor uses row-major access. For example for three dimensions,
 * neighboring z coordinates are next to each other in memory, followed by y
 * coordinates and then x coordinates.
 *
 * @tparam Dimensionality  The number of dimensions managed by this accessor
 *
 * @tparam ArithmeticType  Value type used for arithmetic operations and
 *                         for in- and output
 *
 * @tparam StorageType  Value type used for storing the actual value to memory
 *
 * @note  This class only manages the accesses and not the memory itself.
 */
template <int Dimensionality, typename ArithmeticType, typename StorageType>
class reduced_row_major {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;
    static constexpr size_type dimensionality{Dimensionality};
    static constexpr bool is_const{std::is_const<storage_type>::value};
    using const_accessor =
        reduced_row_major<dimensionality, arithmetic_type, const storage_type>;

    static_assert(Dimensionality >= 1,
                  "Dimensionality must be a positive number!");

    friend class range<reduced_row_major>;

protected:
    using reference_type =
        reference_class::reduced_storage<arithmetic_type, storage_type>;

    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param size  multidimensional size of the memory
     * @param stride  stride array used for memory accesses
     */
    GKO_ATTRIBUTES constexpr reduced_row_major(
        storage_type *storage, gko::dim<dimensionality> size,
        const std::array<const size_type, dimensionality - 1> &stride)
        : storage_{storage}, size_{size}, stride_{stride}
    {}

    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param size  multidimensional size of the memory
     * @param stride  stride array used for memory accesses
     */
    GKO_ATTRIBUTES constexpr reduced_row_major(
        storage_type *storage, gko::dim<dimensionality> size,
        std::array<const size_type, dimensionality - 1> &&stride)
        : storage_{storage}, size_{size}, stride_{stride}
    {}

    /**
     * Creates the accessor for an already allocated storage space.
     * It is assumed that all accesses are without padding.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param size  multidimensional size of the memory
     */
    GKO_ATTRIBUTES constexpr reduced_row_major(storage_type *storage,
                                               dim<dimensionality> size)
        : reduced_row_major{storage, size,
                            helper::compute_stride_array<const size_type>(size)}
    {}

    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param size  multidimensional size of the memory
     * @param strides  strides used for memory accesses
     */
    template <typename... Strides>
    GKO_ATTRIBUTES constexpr reduced_row_major(storage_type *storage,
                                               gko::dim<dimensionality> size,
                                               Strides &&... strides)
        : storage_{storage},
          size_{size},
          stride_{std::forward<Strides>(strides)...}
    {
        static_assert(sizeof...(Strides) + 1 == dimensionality,
                      "Number of provided Strides must be dimensionality - 1!");
    }

    /**
     * Creates an empty accessor (pointing nowhere with an empty size)
     */
    GKO_ATTRIBUTES constexpr reduced_row_major()
        : reduced_row_major{nullptr, {0, 0, 0}}
    {}

public:
    /**
     * Creates a reduced_row_major range which contains a read-only version of
     * the current accessor.
     *
     * @returns  a reduced_row_major major range which is read-only.
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr range<const_accessor> to_const() const
    {
        return range<const_accessor>{storage_, size_, stride_};
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @returns  length in dimension `dimension`
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type length(
        size_type dimension) const
    {
        return dimension < dimensionality ? size_[dimension] : 1;
    }

    /**
     * Copies data from another accessor
     *
     * @warning Do not use this function since it is not optimized for a
     *          specific executor. It will always be performed sequentially.
     *          Please write an optimized version (adjusted to the architecture)
     *          by iterating through the values yourself.
     *
     * @tparam OtherAccessor  type of the other accessor
     *
     * @param other  other accessor
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        helper::multidim_for_each(size_, [&](auto... indices) {
            (*this)(indices...) = other(indices...);
        });
    }

    /**
     * Returns the stored value for the given indices. If the storage is const,
     * a value is returned, otherwise, a reference is returned.
     *
     * @param indices  indices which value is supposed to access
     *
     * @returns  the stored value if the accessor is const (if the storage type
     *           is const), or a reference if the accessor is non-const
     */
    template <typename... Indices>
    GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<
        helper::are_all_integral<Indices...>::value,
        std::conditional_t<is_const, arithmetic_type, reference_type>>
    operator()(Indices &&... indices) const
    {
        return reference_type{this->storage_ +
                              compute_index(std::forward<Indices>(indices)...)};
    }

    /**
     * Returns a sub-range spinning the current range (x1_span, x2_span, ...)
     *
     * @param spans  span for the indices
     *
     * @returns a sub-range for the given spans.
     */
    template <typename... SpanTypes>
    GKO_ATTRIBUTES constexpr std::enable_if_t<
        helper::are_span_compatible<SpanTypes...>::value,
        range<reduced_row_major>>
    operator()(SpanTypes &&... spans) const
    {
        return helper::validate_spans(size_, spans...),
               range<reduced_row_major>{
                   storage_ + compute_index((span{spans}.begin)...),
                   dim<dimensionality>{
                       (span{spans}.end - span{spans}.begin)...},
                   stride_};
    }

    /**
     * Returns the size of the accessor
     *
     * @returns the size of the accessor
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr dim<dimensionality> get_size() const
    {
        return size_;
    }

    /**
     * Returns a pointer to a stride array of size dimensionality - 1
     *
     * @returns returns a pointer to a stride array of size dimensionality - 1
     */
    GKO_ATTRIBUTES
    GKO_INLINE constexpr const std::array<const size_type, dimensionality - 1>
        &get_stride() const
    {
        return stride_;
    }

    /**
     * Returns the pointer to the storage data
     *
     * @returns the pointer to the storage data
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr storage_type *get_stored_data() const
    {
        return storage_;
    }

    /**
     * Returns a const pointer to the storage data
     *
     * @returns a const pointer to the storage data
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr const storage_type *get_const_storage()
        const
    {
        return storage_;
    }

protected:
    template <typename... Indices>
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type compute_index(
        Indices &&... indices) const
    {
        static_assert(sizeof...(Indices) == dimensionality,
                      "Number of indices must match dimensionality!");
        return helper::row_major_index<
            const size_type, dimensionality>::compute(size_, stride_,
                                                      std::forward<Indices>(
                                                          indices)...);
    }

    storage_type *storage_;
    const dim<dimensionality> size_;
    const std::array<const size_type, dimensionality - 1> stride_;
};


namespace detail {


// In case of a const type, do not provide a write function
template <int Dimensionality, typename Accessor, typename ScaleType,
          bool = std::is_const<ScaleType>::value>
struct enable_write_scalar {
    static_assert(std::is_const<ScaleType>::value,
                  "This class must have a constant ScaleType!");
    using scalar_type = ScaleType;
};

// In case of a non-const type, enable the write function
template <int Dimensionality, typename Accessor, typename ScaleType>
struct enable_write_scalar<Dimensionality, Accessor, ScaleType, false> {
    static_assert(!std::is_const<ScaleType>::value,
                  "This class must NOT have a constant ScaleType!");
    static_assert(Dimensionality >= 1,
                  "Dimensionality must be a positive number!");

    using scalar_type = ScaleType;

    /**
     * Writes the scalar value at the given indices.
     *
     * @param value  value to write
     * @param indices  indices where to write the value
     *
     * @returns the written value.
     */
    template <typename... Indices>
    GKO_ATTRIBUTES GKO_INLINE constexpr scalar_type write_scalar(
        scalar_type value, Indices &&... indices) const
    {
        static_assert(sizeof...(Indices) == Dimensionality,
                      "Number of indices must match dimensionality!");
        scalar_type *GKO_RESTRICT rest_scalar = self()->scalar_;
        return rest_scalar[self()->compute_scalar_index(
                   std::forward<Indices>(indices)...)] = value;
    }

private:
    GKO_ATTRIBUTES GKO_INLINE constexpr const Accessor *self() const
    {
        return static_cast<const Accessor *>(this);
    }
};


}  // namespace detail


/**
 * The scaled_reduced_row_major class allows a storage format that is different
 * from the arithmetic format (which is returned from the brace operator).
 * Additionally, a scalar is used when reading and writing data to allow for
 * a shift in range.
 * As storage, the StorageType is used.
 *
 * This accessor uses row-major access. For example, for three dimensions,
 * neighboring z coordinates are next to each other in memory, followed by y
 * coordinates and then x coordinates.
 *
 * @tparam Dimensionality  The number of dimensions managed by this accessor
 *
 * @tparam ArithmeticType  Value type used for arithmetic operations and
 *                         for in- and output
 *
 * @tparam StorageType  Value type used for storing the actual value to memory
 *
 * @tparam ScalarMask  Binary mask that marks which indices matter for the
 *                     scalar selection (set bit means the corresponding index
 *                     needs to be considered, 0 means it is not). The least
 *                     significand bit corresponds to the first index dimension,
 *                     the second least to the second index dimension, and so
 *                     on.
 *
 * @note  This class only manages the accesses and not the memory itself.
 */
template <int Dimensionality, typename ArithmeticType, typename StorageType,
          int32 ScalarMask>
class scaled_reduced_row_major
    : public detail::enable_write_scalar<
          Dimensionality,
          scaled_reduced_row_major<Dimensionality, ArithmeticType, StorageType,
                                   ScalarMask>,
          ArithmeticType> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;
    static constexpr size_type dimensionality{Dimensionality};
    static constexpr int32 scalar_mask{ScalarMask};
    static constexpr bool is_const{std::is_const<storage_type>::value};
    using scalar_type =
        std::conditional_t<is_const, const arithmetic_type, arithmetic_type>;

    using const_accessor =
        scaled_reduced_row_major<dimensionality, arithmetic_type,
                                 const storage_type, ScalarMask>;

    static_assert(Dimensionality >= 1,
                  "Dimensionality must be a positive number!");
    static_assert(dimensionality <= 32,
                  "Only Dimensionality <= 32 is currently supported");

    // Allow access to both `scalar_` and `compute_scalar_index()`
    friend class detail::enable_write_scalar<
        dimensionality, scaled_reduced_row_major, scalar_type>;
    friend class range<scaled_reduced_row_major>;

protected:
    using reference_type =
        reference_class::scaled_reduced_storage<arithmetic_type, StorageType>;

public:
    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param scalar  pointer to the block of memory containing the scalar
     *                values.
     * @param size  multidimensional size of the memory
     * @param stride  stride array used for memory accesses
     */
    GKO_ATTRIBUTES constexpr scaled_reduced_row_major(
        storage_type *storage, scalar_type *scalar, dim<dimensionality> size,
        const std::array<const size_type, dimensionality - 1> &stride)
        : storage_{storage}, scalar_{scalar}, size_{size}, stride_{stride}
    {}

    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param scalar  pointer to the block of memory containing the scalar
     *                values.
     * @param size  multidimensional size of the memory
     * @param stride  stride array used for memory accesses
     */
    GKO_ATTRIBUTES constexpr scaled_reduced_row_major(
        storage_type *storage, scalar_type *scalar, dim<dimensionality> size,
        std::array<const size_type, dimensionality - 1> &&stride)
        : storage_{storage}, scalar_{scalar}, size_{size}, stride_{stride}
    {}

    /**
     * Creates the accessor for an already allocated storage space.
     * It is assumed that all accesses are without padding.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param scalar  pointer to the block of memory containing the scalar
     *                values.
     * @param size  multidimensional size of the memory
     */
    GKO_ATTRIBUTES constexpr scaled_reduced_row_major(storage_type *storage,
                                                      scalar_type *scalar,
                                                      dim<dimensionality> size)
        : scaled_reduced_row_major{
              storage, scalar, size,
              helper::compute_stride_array<const size_type>(size)}
    {}

    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param scalar  pointer to the block of memory containing the scalar
     *                values.
     * @param size  multidimensional size of the memory
     * @param strides  stride array used for memory accesses
     */
    template <typename... Strides>
    GKO_ATTRIBUTES constexpr scaled_reduced_row_major(storage_type *storage,
                                                      scalar_type *scalar,
                                                      dim<dimensionality> size,
                                                      Strides &&... strides)
        : storage_{storage},
          scalar_{scalar},
          size_{size},
          stride_{std::forward<Strides>(strides)...}
    {
        static_assert(sizeof...(Strides) + 1 == dimensionality,
                      "Number of provided Strides must be dimensionality - 1!");
    }

    /**
     * Creates an empty accessor (pointing nowhere with an empty size)
     */
    GKO_ATTRIBUTES constexpr scaled_reduced_row_major()
        : scaled_reduced_row_major{nullptr, nullptr, {0, 0, 0}}
    {}

    /**
     * Creates a scaled_reduced_row_major range which contains a read-only
     * version of the current accessor.
     *
     * @returns  a scaled_reduced_row_major major range which is read-only.
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr range<const_accessor> to_const() const
    {
        return range<const_accessor>{storage_, scalar_, size_, stride_};
    }

    /**
     * Reads the scalar value at the given indices.
     *
     * @param indices  indices which data to access
     *
     * @returns the scalar value at the given indices.
     */
    template <typename... Indices>
    GKO_ATTRIBUTES GKO_INLINE constexpr scalar_type read_scalar(
        Indices &&... indices) const
    {
        const arithmetic_type *GKO_RESTRICT rest_scalar = scalar_;
        return rest_scalar[compute_scalar_index(
            std::forward<Indices>(indices)...)];
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @returns length in dimension `dimension`
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type length(
        size_type dimension) const
    {
        return dimension < dimensionality ? size_[dimension] : 1;
    }

    /**
     * Copies data from another accessor
     *
     * @warning Do not use this function since it is not optimized for a
     *          specific executor. It will always be performed sequentially.
     *          Please write an optimized version (adjusted to the architecture)
     *          by iterating through the values yourself.
     *
     * @tparam OtherAccessor  type of the other accessor
     *
     * @param other  other accessor
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        helper::multidim_for_each(size_, [&](auto... indices) {
            this->write_scalar(other.read_scalar(indices...), indices...);
            (*this)(indices...) = other(indices...);
        });
    }

    /**
     * Returns the stored value for the given indices. If the storage is const,
     * a value is returned, otherwise, a reference is returned.
     *
     * @param indices  indices which value is supposed to access
     *
     * @returns  the stored value if the accessor is const (if the storage type
     *           is const), or a reference if the accessor is non-const
     */
    template <typename... Indices>
    GKO_ATTRIBUTES GKO_INLINE constexpr std::enable_if_t<
        helper::are_all_integral<Indices...>::value,
        std::conditional_t<is_const, arithmetic_type, reference_type>>
    operator()(Indices... indices) const
    {
        return reference_type{
            this->storage_ + compute_index(std::forward<Indices>(indices)...),
            read_scalar(std::forward<Indices>(indices)...)};
    }

    /**
     * Returns a sub-range spinning the current range (x1_span, x2_span, ...)
     *
     * @param spans  span for the indices
     *
     * @returns a sub-range for the given spans.
     */
    template <typename... SpanTypes>
    GKO_ATTRIBUTES constexpr std::enable_if_t<
        helper::are_span_compatible<SpanTypes...>::value,
        range<scaled_reduced_row_major>>
    operator()(SpanTypes &&... spans) const
    {
        return helper::validate_spans(size_, spans...),
               range<scaled_reduced_row_major>{
                   storage_ + compute_index((span{spans}.begin)...),
                   scalar_ + compute_scalar_index(span{spans}.begin...),
                   dim<dimensionality>{
                       (span{spans}.end - span{spans}.begin)...},
                   stride_};
    }

    /**
     * Returns the size of the accessor
     *
     * @returns the size of the accessor
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr dim<dimensionality> get_size() const
    {
        return size_;
    }

    /**
     * Returns a pointer to a stride array of size dimensionality - 1
     *
     * @returns returns a pointer to a stride array of size dimensionality - 1
     */
    GKO_ATTRIBUTES
    GKO_INLINE constexpr const std::array<const size_type, dimensionality - 1>
        &get_stride() const
    {
        return stride_;
    }

    /**
     * Returns the pointer to the storage data
     *
     * @returns the pointer to the storage data
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr storage_type *get_stored_data() const
    {
        return storage_;
    }

    /**
     * Returns a const pointer to the storage data
     *
     * @returns a const pointer to the storage data
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr const storage_type *get_const_storage()
        const
    {
        return storage_;
    }

    /**
     * Returns the pointer to the scalar data
     *
     * @returns the pointer to the scalar data
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr scalar_type *get_scalar() const
    {
        return this->scalar_;
    }

    /**
     * Returns a const pointer to the scalar data
     *
     * @returns a const pointer to the scalar data
     */
    GKO_ATTRIBUTES GKO_INLINE constexpr const scalar_type *get_const_scalar()
        const
    {
        return this->scalar_;
    }

protected:
    template <typename... Indices>
    GKO_ATTRIBUTES GKO_INLINE constexpr size_type compute_index(
        Indices &&... indices) const
    {
        static_assert(sizeof...(Indices) == dimensionality,
                      "Number of indices must match dimensionality!");
        return helper::row_major_index<
            const size_type, dimensionality>::compute(size_, stride_,
                                                      std::forward<Indices>(
                                                          indices)...);
    }

    template <typename... Indices>
    GKO_ATTRIBUTES constexpr GKO_INLINE size_type
    compute_scalar_index(Indices &&... indices) const
    {
        static_assert(sizeof...(Indices) == dimensionality,
                      "Number of indices must match dimensionality!");
        return helper::row_major_mask_index<
            const size_type, dimensionality,
            scalar_mask>::compute(size_, stride_,
                                  std::forward_as_tuple(indices...));
    }

    storage_type *storage_;
    scalar_type *scalar_;
    const dim<dimensionality> size_;
    const std::array<const size_type, dimensionality - 1> stride_;
};


}  // namespace accessor
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_RANGE_ACCESSORS_HPP_
