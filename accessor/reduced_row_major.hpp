// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_REDUCED_ROW_MAJOR_HPP_
#define GKO_ACCESSOR_REDUCED_ROW_MAJOR_HPP_


#include <array>
#include <cinttypes>
#include <memory>
#include <type_traits>
#include <utility>


#include "accessor_helper.hpp"
#include "index_span.hpp"
#include "range.hpp"
#include "reduced_row_major_reference.hpp"
#include "utils.hpp"


namespace gko {
/**
 * @brief The accessor namespace.
 *
 * @ingroup accessor
 */
namespace acc {


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
template <std::size_t Dimensionality, typename ArithmeticType,
          typename StorageType>
class reduced_row_major {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;
    static constexpr auto dimensionality = Dimensionality;
    static constexpr bool is_const{std::is_const<storage_type>::value};
    using const_accessor =
        reduced_row_major<dimensionality, arithmetic_type, const storage_type>;

    static_assert(Dimensionality >= 1,
                  "Dimensionality must be a positive number!");

    friend class range<reduced_row_major>;

protected:
    using dim_type = std::array<size_type, dimensionality>;
    using storage_stride_type = std::array<size_type, dimensionality - 1>;
    using reference_type =
        reference_class::reduced_storage<arithmetic_type, storage_type>;

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
     * @param stride  stride array used for memory accesses
     */
    constexpr GKO_ACC_ATTRIBUTES reduced_row_major(dim_type size,
                                                   storage_type* storage,
                                                   storage_stride_type stride)
        : size_(size), storage_{storage}, stride_(stride)
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
    constexpr GKO_ACC_ATTRIBUTES reduced_row_major(dim_type size,
                                                   storage_type* storage,
                                                   Strides&&... strides)
        : reduced_row_major{
              size, storage,
              storage_stride_type{{std::forward<Strides>(strides)...}}}
    {
        static_assert(sizeof...(Strides) + 1 == dimensionality,
                      "Number of provided Strides must be dimensionality - 1!");
    }

    /**
     * Creates the accessor for an already allocated storage space.
     * It is assumed that all accesses are without padding.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param size  multidimensional size of the memory
     */
    constexpr GKO_ACC_ATTRIBUTES reduced_row_major(dim_type size,
                                                   storage_type* storage)
        : reduced_row_major{
              size, storage,
              helper::compute_default_row_major_stride_array(size)}
    {}

    /**
     * Creates an empty accessor (pointing nowhere with an empty size)
     */
    constexpr GKO_ACC_ATTRIBUTES reduced_row_major()
        : reduced_row_major{{0, 0, 0}, nullptr}
    {}

public:
    /**
     * Creates a reduced_row_major range which contains a read-only version of
     * the current accessor.
     *
     * @returns  a reduced_row_major major range which is read-only.
     */
    constexpr GKO_ACC_ATTRIBUTES range<const_accessor> to_const() const
    {
        return range<const_accessor>{size_, storage_, stride_};
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @returns  length in dimension `dimension`
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
    operator()(Indices&&... indices) const
    {
        return reference_type{storage_ +
                              compute_index(std::forward<Indices>(indices)...)};
    }

    /**
     * Returns a sub-range spanning the current range (x1_span, x2_span, ...)
     *
     * @param spans  span for the indices
     *
     * @returns a sub-range for the given spans.
     */
    template <typename... SpanTypes>
    constexpr GKO_ACC_ATTRIBUTES
        std::enable_if_t<helper::are_index_span_compatible<SpanTypes...>::value,
                         range<reduced_row_major>>
        operator()(SpanTypes... spans) const
    {
        return helper::validate_index_spans(size_, spans...),
               range<reduced_row_major>{
                   dim_type{
                       (index_span{spans}.end - index_span{spans}.begin)...},
                   storage_ + compute_index((index_span{spans}.begin)...),
                   stride_};
    }


    /**
     * Returns the storage address for the given indices. If the storage is
     * const, a const address is returned, otherwise, an address is returned.
     *
     * @param indices  indices which value is supposed to access
     *
     * @returns  the const address if the accessor is const (if the storage type
     *           is const), or an address if the accessor is non-const
     */
    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<
        are_all_integral<Indices...>::value,
        std::conditional_t<is_const, const storage_type * GKO_ACC_RESTRICT,
                           storage_type * GKO_ACC_RESTRICT>>
    get_storage_address(Indices&&... indices) const
    {
        return storage_ + compute_index(std::forward<Indices>(indices)...);
    }

    /**
     * Returns the size of the accessor
     *
     * @returns the size of the accessor
     */
    constexpr GKO_ACC_ATTRIBUTES dim_type get_size() const { return size_; }

    /**
     * Returns a pointer to a stride array of size dimensionality - 1
     *
     * @returns returns a pointer to a stride array of size dimensionality - 1
     */
    GKO_ACC_ATTRIBUTES
    constexpr const storage_stride_type& get_stride() const { return stride_; }

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

protected:
    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES index_type
    compute_index(Indices&&... indices) const
    {
        static_assert(sizeof...(Indices) == dimensionality,
                      "Number of indices must match dimensionality!");
        return helper::compute_row_major_index<index_type, dimensionality>(
            size_, stride_, std::forward<Indices>(indices)...);
    }

private:
    const dim_type size_;
    storage_type* const storage_;
    const storage_stride_type stride_;
};


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_REDUCED_ROW_MAJOR_HPP_
