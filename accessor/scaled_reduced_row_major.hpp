// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_SCALED_REDUCED_ROW_MAJOR_HPP_
#define GKO_ACCESSOR_SCALED_REDUCED_ROW_MAJOR_HPP_


#include <array>
#include <cinttypes>
#include <type_traits>
#include <utility>


#include "accessor_helper.hpp"
#include "index_span.hpp"
#include "range.hpp"
#include "scaled_reduced_row_major_reference.hpp"
#include "utils.hpp"


namespace gko {
/**
 * @brief The accessor namespace.
 *
 * @ingroup accessor
 */
namespace acc {
namespace detail {


// In case of a const type, do not provide a write function
template <int Dimensionality, typename Accessor, typename ScalarType,
          bool = std::is_const<ScalarType>::value>
struct enable_write_scalar {
    using scalar_type = ScalarType;
};

// In case of a non-const type, enable the write function
template <int Dimensionality, typename Accessor, typename ScalarType>
struct enable_write_scalar<Dimensionality, Accessor, ScalarType, false> {
    static_assert(Dimensionality >= 1,
                  "Dimensionality must be a positive number!");

    using scalar_type = ScalarType;

    /**
     * Writes the scalar value at the given indices.
     * The number of indices must be equal to the number of dimensions, even
     * if some of the indices are ignored (depending on the scalar mask).
     *
     * @param value  value to write
     * @param indices  indices where to write the value
     *
     * @returns the written value.
     */
    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES scalar_type
    write_scalar_masked(scalar_type value, Indices&&... indices) const
    {
        static_assert(sizeof...(Indices) == Dimensionality,
                      "Number of indices must match dimensionality!");
        scalar_type* GKO_ACC_RESTRICT rest_scalar = self()->scalar_;
        return rest_scalar[self()->compute_mask_scalar_index(
                   std::forward<Indices>(indices)...)] = value;
    }

    /**
     * Writes the scalar value at the given indices.
     * Only the actually used indices must be provided, meaning the number of
     * specified indices must be equal to the number of set bits in the
     * scalar mask.
     *
     * @param value  value to write
     * @param indices  indices where to write the value
     *
     * @returns the written value.
     */
    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES scalar_type
    write_scalar_direct(scalar_type value, Indices&&... indices) const
    {
        scalar_type* GKO_ACC_RESTRICT rest_scalar = self()->scalar_;
        return rest_scalar[self()->compute_direct_scalar_index(
                   std::forward<Indices>(indices)...)] = value;
    }


private:
    constexpr GKO_ACC_ATTRIBUTES const Accessor* self() const
    {
        return static_cast<const Accessor*>(this);
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
 *                     significand bit corresponds to the last index dimension,
 *                     the second least to the second last index dimension, and
 *                     so on.
 *                     For example, the mask = 0b011101 means that for the 5d
 *                     indices (x1, x2, x3, x4, x5), (x1, x2, x3, x5) are
 *                     considered for the scalar, making the scalar itself 4d.
 *
 * @note  This class only manages the accesses and not the memory itself.
 */
template <std::size_t Dimensionality, typename ArithmeticType,
          typename StorageType, std::uint64_t ScalarMask>
class scaled_reduced_row_major
    : public detail::enable_write_scalar<
          Dimensionality,
          scaled_reduced_row_major<Dimensionality, ArithmeticType, StorageType,
                                   ScalarMask>,
          ArithmeticType, std::is_const<StorageType>::value> {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;
    static constexpr auto dimensionality = Dimensionality;
    static constexpr auto scalar_mask = ScalarMask;
    static constexpr bool is_const{std::is_const<storage_type>::value};
    using scalar_type =
        std::conditional_t<is_const, const arithmetic_type, arithmetic_type>;

    using const_accessor =
        scaled_reduced_row_major<dimensionality, arithmetic_type,
                                 const storage_type, ScalarMask>;

    static_assert(!is_complex<ArithmeticType>::value &&
                      !is_complex<StorageType>::value,
                  "Both arithmetic and storage type must not be complex!");
    static_assert(Dimensionality >= 1,
                  "Dimensionality must be a positive number!");
    static_assert(dimensionality <= 32,
                  "Only Dimensionality <= 32 is currently supported");

    // Allow access to both `scalar_` and `compute_mask_scalar_index()`
    friend class detail::enable_write_scalar<
        dimensionality, scaled_reduced_row_major, scalar_type>;
    friend class range<scaled_reduced_row_major>;

protected:
    static constexpr std::size_t scalar_dim{
        helper::count_mask_dimensionality<scalar_mask, dimensionality>()};
    static constexpr std::size_t scalar_stride_dim{
        scalar_dim == 0 ? 0 : (scalar_dim - 1)};

    using dim_type = std::array<size_type, dimensionality>;
    using storage_stride_type = std::array<size_type, dimensionality - 1>;
    using scalar_stride_type = std::array<size_type, scalar_stride_dim>;
    using reference_type =
        reference_class::scaled_reduced_storage<arithmetic_type, StorageType>;

private:
    using index_type = std::int64_t;

protected:
    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param size  multidimensional size of the memory
     * @param storage  pointer to the block of memory containing the storage
     * @param storage_stride  stride array used for memory accesses to storage
     * @param scalar  pointer to the block of memory containing the scalar
     *                values.
     * @param scalar_stride  stride array used for memory accesses to scalar
     */
    constexpr GKO_ACC_ATTRIBUTES scaled_reduced_row_major(
        dim_type size, storage_type* storage,
        storage_stride_type storage_stride, scalar_type* scalar,
        scalar_stride_type scalar_stride)
        : size_(size),
          storage_{storage},
          storage_stride_(storage_stride),
          scalar_{scalar},
          scalar_stride_(scalar_stride)
    {}

    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param size  multidimensional size of the memory
     * @param storage  pointer to the block of memory containing the storage
     * @param stride  stride array used for memory accesses to storage
     * @param scalar  pointer to the block of memory containing the scalar
     *                values.
     */
    constexpr GKO_ACC_ATTRIBUTES scaled_reduced_row_major(
        dim_type size, storage_type* storage, storage_stride_type stride,
        scalar_type* scalar)
        : scaled_reduced_row_major{
              size, storage, stride, scalar,
              helper::compute_default_masked_row_major_stride_array<
                  scalar_mask, scalar_stride_dim, dimensionality>(size)}
    {}

    /**
     * Creates the accessor for an already allocated storage space.
     * It is assumed that all accesses are without padding.
     *
     * @param size  multidimensional size of the memory
     * @param storage  pointer to the block of memory containing the storage
     * @param scalar  pointer to the block of memory containing the scalar
     *                values.
     */
    constexpr GKO_ACC_ATTRIBUTES scaled_reduced_row_major(dim_type size,
                                                          storage_type* storage,
                                                          scalar_type* scalar)
        : scaled_reduced_row_major{
              size, storage,
              helper::compute_default_row_major_stride_array(size), scalar}
    {}

    /**
     * Creates an empty accessor (pointing nowhere with an empty size)
     */
    constexpr GKO_ACC_ATTRIBUTES scaled_reduced_row_major()
        : scaled_reduced_row_major{{0, 0, 0}, nullptr, nullptr}
    {}

public:
    /**
     * Creates a scaled_reduced_row_major range which contains a read-only
     * version of the current accessor.
     *
     * @returns  a scaled_reduced_row_major major range which is read-only.
     */
    constexpr GKO_ACC_ATTRIBUTES range<const_accessor> to_const() const
    {
        return range<const_accessor>{size_, storage_, storage_stride_, scalar_,
                                     scalar_stride_};
    }

    /**
     * Reads the scalar value at the given indices. Only indices where the
     * scalar mask bit is set are considered, the others are ignored.
     *
     * @param indices  indices which data to access
     *
     * @returns the scalar value at the given indices.
     */
    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES scalar_type
    read_scalar_masked(Indices&&... indices) const
    {
        const arithmetic_type* GKO_ACC_RESTRICT rest_scalar = scalar_;
        return rest_scalar[compute_mask_scalar_index(
            std::forward<Indices>(indices)...)];
    }

    /**
     * Reads the scalar value at the given indices. Only the actually used
     * indices must be provided, meaning the number of specified indices must
     * be equal to the number of set bits in the scalar mask.
     *
     * @param indices  indices which data to access
     *
     * @returns the scalar value at the given indices.
     */
    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES scalar_type
    read_scalar_direct(Indices&&... indices) const
    {
        const arithmetic_type* GKO_ACC_RESTRICT rest_scalar = scalar_;
        return rest_scalar[compute_direct_scalar_index(
            std::forward<Indices>(indices)...)];
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @returns length in dimension `dimension`
     */
    constexpr GKO_ACC_ATTRIBUTES size_type length(size_type dimension) const
    {
        return dimension < dimensionality ? size_[dimension] : 1;
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
    constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<
        are_all_integral<Indices...>::value,
        std::conditional_t<is_const, arithmetic_type, reference_type>>
    operator()(Indices... indices) const
    {
        return reference_type{storage_ + compute_index(indices...),
                              read_scalar_masked(indices...)};
    }

    /**
     * Returns a sub-range spinning the current range (x1_span, x2_span, ...)
     *
     * @param spans  span for the indices
     *
     * @returns a sub-range for the given spans.
     */
    template <typename... SpanTypes>
    constexpr GKO_ACC_ATTRIBUTES
        std::enable_if_t<helper::are_index_span_compatible<SpanTypes...>::value,
                         range<scaled_reduced_row_major>>
        operator()(SpanTypes... spans) const
    {
        return helper::validate_index_spans(size_, spans...),
               range<scaled_reduced_row_major>{
                   dim_type{
                       (index_span{spans}.end - index_span{spans}.begin)...},
                   storage_ + compute_index((index_span{spans}.begin)...),
                   storage_stride_,
                   scalar_ +
                       compute_mask_scalar_index(index_span{spans}.begin...),
                   scalar_stride_};
    }

    /**
     * Returns the size of the accessor
     *
     * @returns the size of the accessor
     */
    constexpr GKO_ACC_ATTRIBUTES dim_type get_size() const { return size_; }

    /**
     * Returns a const reference to the storage stride array of size
     * dimensionality - 1
     *
     * @returns a const reference to the storage stride array of size
     *          dimensionality - 1
     */
    constexpr GKO_ACC_ATTRIBUTES const storage_stride_type& get_storage_stride()
        const
    {
        return storage_stride_;
    }

    /**
     * Returns a const reference to the scalar stride array
     *
     * @returns a const reference to the scalar stride array
     */
    constexpr GKO_ACC_ATTRIBUTES const scalar_stride_type& get_scalar_stride()
        const
    {
        return scalar_stride_;
    }

    /**
     * Returns the pointer to the storage data
     *
     * @returns the pointer to the storage data
     */
    constexpr GKO_ACC_ATTRIBUTES storage_type* get_stored_data() const
    {
        return storage_;
    }

    /**
     * Returns a const pointer to the storage data
     *
     * @returns a const pointer to the storage data
     */
    constexpr GKO_ACC_ATTRIBUTES const storage_type* get_const_storage() const
    {
        return storage_;
    }

    /**
     * Returns the pointer to the scalar data
     *
     * @returns the pointer to the scalar data
     */
    constexpr GKO_ACC_ATTRIBUTES scalar_type* get_scalar() const
    {
        return scalar_;
    }

    /**
     * Returns a const pointer to the scalar data
     *
     * @returns a const pointer to the scalar data
     */
    constexpr GKO_ACC_ATTRIBUTES const scalar_type* get_const_scalar() const
    {
        return scalar_;
    }

protected:
    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES size_type
    compute_index(Indices&&... indices) const
    {
        static_assert(sizeof...(Indices) == dimensionality,
                      "Number of indices must match dimensionality!");
        return helper::compute_row_major_index<index_type, dimensionality>(
            size_, storage_stride_, std::forward<Indices>(indices)...);
    }

    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES size_type
    compute_mask_scalar_index(Indices&&... indices) const
    {
        static_assert(sizeof...(Indices) == dimensionality,
                      "Number of indices must match dimensionality!");
        return helper::compute_masked_index<index_type, scalar_mask,
                                            scalar_stride_dim>(
            size_, scalar_stride_, std::forward<Indices>(indices)...);
    }

    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES size_type
    compute_direct_scalar_index(Indices&&... indices) const
    {
        static_assert(
            sizeof...(Indices) == scalar_dim,
            "Number of indices must match number of set bits in scalar mask!");
        return helper::compute_masked_index_direct<index_type, scalar_mask,
                                                   scalar_stride_dim>(
            size_, scalar_stride_, std::forward<Indices>(indices)...);
    }


private:
    const dim_type size_;
    storage_type* const storage_;
    const storage_stride_type storage_stride_;
    scalar_type* const scalar_;
    const scalar_stride_type scalar_stride_;
};


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_SCALED_REDUCED_ROW_MAJOR_HPP_
