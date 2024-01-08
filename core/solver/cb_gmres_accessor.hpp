// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_CB_GMRES_ACCESSOR_HPP_
#define GKO_CORE_SOLVER_CB_GMRES_ACCESSOR_HPP_


#include <array>
#include <cinttypes>
#include <limits>
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


#include "accessor/range.hpp"
#include "accessor/reduced_row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"


namespace gko {
namespace cb_gmres {


namespace detail {


template <typename Accessor>
struct has_3d_scaled_accessor : public std::false_type {};

template <typename T1, typename T2, uint64 mask>
struct has_3d_scaled_accessor<
    acc::range<acc::scaled_reduced_row_major<3, T1, T2, mask>>>
    : public std::true_type {};

template <typename StorageType, bool = std::is_integral<StorageType>::value>
struct helper_require_scale {};

template <typename StorageType>
struct helper_require_scale<StorageType, false> : public std::false_type {};

template <typename StorageType>
struct helper_require_scale<StorageType, true> : public std::true_type {};


}  // namespace detail


template <typename ValueType, typename StorageType,
          bool = detail::helper_require_scale<StorageType>::value>
class Range3dHelper {};


template <typename ValueType, typename StorageType>
class Range3dHelper<ValueType, StorageType, true> {
public:
    using Accessor =
        acc::scaled_reduced_row_major<3, ValueType, StorageType, 0b101>;
    using Range = acc::range<Accessor>;

    Range3dHelper() = default;

    Range3dHelper(std::shared_ptr<const Executor> exec, dim<3> krylov_dim)
        : krylov_dim_{{static_cast<acc::size_type>(krylov_dim[0]),
                       static_cast<acc::size_type>(krylov_dim[1]),
                       static_cast<acc::size_type>(krylov_dim[2])}},
          bases_{exec, krylov_dim[0] * krylov_dim[1] * krylov_dim[2]},
          scale_{exec, krylov_dim[0] * krylov_dim[2]}
    {
        array<ValueType> h_scale{exec->get_master(),
                                 krylov_dim[0] * krylov_dim[2]};
        for (size_type i = 0; i < h_scale.get_size(); ++i) {
            h_scale.get_data()[i] = one<ValueType>();
        }
        scale_ = h_scale;
    }

    Range get_range()
    {
        return Range(krylov_dim_, bases_.get_data(), scale_.get_data());
    }

    gko::array<StorageType>& get_bases() { return bases_; }

private:
    std::array<acc::size_type, 3> krylov_dim_;
    array<StorageType> bases_;
    array<ValueType> scale_;
};


template <typename ValueType, typename StorageType>
class Range3dHelper<ValueType, StorageType, false> {
public:
    using Accessor = acc::reduced_row_major<3, ValueType, StorageType>;
    using Range = acc::range<Accessor>;

    Range3dHelper() = default;

    Range3dHelper(std::shared_ptr<const Executor> exec, dim<3> krylov_dim)
        : krylov_dim_{{static_cast<acc::size_type>(krylov_dim[0]),
                       static_cast<acc::size_type>(krylov_dim[1]),
                       static_cast<acc::size_type>(krylov_dim[2])}},
          bases_{std::move(exec), krylov_dim[0] * krylov_dim[1] * krylov_dim[2]}
    {}

    Range get_range() { return Range(krylov_dim_, bases_.get_data()); }

    gko::array<StorageType>& get_bases() { return bases_; }

private:
    std::array<acc::size_type, 3> krylov_dim_;
    array<StorageType> bases_;
};


template <typename Accessor3d,
          bool = detail::has_3d_scaled_accessor<Accessor3d>::value>
struct helper_functions_accessor {};

// Accessors having a scale
template <typename Accessor3d>
struct helper_functions_accessor<Accessor3d, true> {
    using arithmetic_type = typename Accessor3d::accessor::arithmetic_type;
    static constexpr auto dimensionality = Accessor3d::dimensionality;
    static_assert(detail::has_3d_scaled_accessor<Accessor3d>::value,
                  "Accessor must have a scalar here!");
    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scalar(Accessor3d krylov_bases,
                                                   IndexType vector_idx,
                                                   IndexType col_idx,
                                                   arithmetic_type value)
    {
        using storage_type = typename Accessor3d::accessor::storage_type;
        constexpr arithmetic_type correction =
            std::is_integral<storage_type>::value
                // Use 2 instead of 1 here to allow for a bit more room
                ? 2 / static_cast<arithmetic_type>(
                          std::numeric_limits<storage_type>::max())
                : 1;
        krylov_bases.get_accessor().write_scalar_direct(value * correction,
                                                        vector_idx, col_idx);
    }

    static constexpr GKO_ATTRIBUTES
        std::array<acc::size_type, dimensionality - 1>
        get_stride(Accessor3d krylov_bases)
    {
        return krylov_bases.get_accessor().get_storage_stride();
    }
};

// Accessors not having a scale
template <typename Accessor3d>
struct helper_functions_accessor<Accessor3d, false> {
    using arithmetic_type = typename Accessor3d::accessor::arithmetic_type;
    static constexpr size_type dimensionality = Accessor3d::dimensionality;
    static_assert(!detail::has_3d_scaled_accessor<Accessor3d>::value,
                  "Accessor must not have a scale here!");

    template <typename IndexType>
    static inline GKO_ATTRIBUTES void write_scalar(Accessor3d, IndexType,
                                                   IndexType, arithmetic_type)
    {
        // Since there is no scalar, there is nothing to write.
    }

    static constexpr GKO_ATTRIBUTES
        std::array<acc::size_type, dimensionality - 1>
        get_stride(Accessor3d krylov_bases)
    {
        return krylov_bases.get_accessor().get_stride();
    }
};


}  // namespace cb_gmres
}  // namespace gko


#endif  // GKO_CORE_SOLVER_CB_GMRES_ACCESSOR_HPP_
