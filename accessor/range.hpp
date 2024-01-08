// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_RANGE_HPP_
#define GKO_ACCESSOR_RANGE_HPP_


#include <utility>


#include "utils.hpp"


namespace gko {
namespace acc {


template <typename Accessor>
class range {
private:
    /**
     * the default check_if_same gives false.
     *
     * @tparam Ref  the reference type
     * @tparam Args  the input type
     */
    template <typename Ref, typename... Args>
    struct check_if_same : public std::false_type {};

    /**
     * check_if_same gives true if the decay type of input is the same type as
     * Ref.
     *
     * @tparam Ref  the reference type
     */
    template <typename Ref>
    struct check_if_same<Ref, Ref> : public std::true_type {};

public:
    /**
     * The type of the underlying accessor.
     */
    using accessor = Accessor;

    /**
     * The number of dimensions of the range.
     */
    static constexpr size_type dimensionality = accessor::dimensionality;

    /**
     * Use the default destructor.
     */
    ~range() = default;

    /**
     * Creates a new range.
     *
     * @tparam AccessorParam  types of parameters forwarded to the accessor
     *                        constructor.
     *
     * @param params  parameters forwarded to Accessor constructor.
     *
     * @note We use SFINAE to allow for a default copy and move constructor to
     *       be generated, so a `range` is trivially copyable if the `Accessor`
     *       is trivially copyable.
     */
    template <typename... AccessorParams,
              std::enable_if_t<
                  !check_if_same<range, std::decay_t<AccessorParams>...>::value,
                  int> = 0>
    GKO_ACC_ATTRIBUTES constexpr explicit range(AccessorParams&&... args)
        : accessor_{std::forward<AccessorParams>(args)...}
    {}

    /**
     * Returns a value (or a sub-range) with the specified indexes.
     *
     * @tparam DimensionTypes  The types of indexes. Supported types depend on
     *                         the underlying accessor, but are usually either
     *                         integer types or index_spans. If at least one
     *                         index is a span, the returned value will be a
     *                         sub-range (if that is supported by the accessor).
     *
     * @param dimensions  the indexes of the values or index_spans for the new
     *                    range.
     *
     * @return a value on position `(dimensions...)` or a sub-range with the
     *         given index_spans.
     */
    template <typename... DimensionTypes>
    GKO_ACC_ATTRIBUTES constexpr auto operator()(DimensionTypes&&... dimensions)
        const -> decltype(std::declval<accessor>()(
            std::forward<DimensionTypes>(dimensions)...))
    {
        static_assert(sizeof...(dimensions) <= dimensionality,
                      "Too many dimensions in range call");
        return accessor_(std::forward<DimensionTypes>(dimensions)...);
    }

    /**
     * Returns the length of the specified dimension of the range.
     *
     * @param dimension  the dimensions whose length is returned
     *
     * @return  the length of the `dimension`-th dimension of the range
     */
    GKO_ACC_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return accessor_.length(dimension);
    }

    /**
     * Returns a pointer to the accessor.
     *
     * Can be used to access data and functions of a specific accessor.
     *
     * @return pointer to the accessor
     */
    GKO_ACC_ATTRIBUTES constexpr const accessor* operator->() const noexcept
    {
        return &accessor_;
    }

    /**
     * `Returns a reference to the accessor.
     *
     * @return reference to the accessor
     */
    GKO_ACC_ATTRIBUTES constexpr const accessor& get_accessor() const noexcept
    {
        return accessor_;
    }

private:
    accessor accessor_;
};


}  // namespace acc
}  // namespace gko

#endif  // GKO_ACCESSOR_RANGE_HPP_
