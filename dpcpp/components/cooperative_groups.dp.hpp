// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_COOPERATIVE_GROUPS_DP_HPP_
#define GKO_DPCPP_COMPONENTS_COOPERATIVE_GROUPS_DP_HPP_


#include <type_traits>


#include <ginkgo/config.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dpct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


/**
 * Ginkgo uses cooperative groups to handle communication among the threads.
 *
 * However, DPCPP's implementation of cooperative groups is still quite limited
 * in functionality, and some parts are not supported on all hardware
 * interesting for Ginkgo. For this reason, Ginkgo exposes only a part of the
 * original functionality, and possibly extends it if it is required. Thus,
 * developers should include and use this header and the gko::group namespace
 * instead of the standard cooperative_groups.h header. The interface exposed
 * by Ginkgo's implementation is equivalent to the standard interface, with some
 * useful extensions.
 *
 * A cooperative group (both from standard DPCPP and from Ginkgo) is not a
 * specific type, but a concept. That is, any type satisfying the interface
 * imposed by the cooperative groups API is considered a cooperative
 * group (a.k.a. "duck typing"). To maximize the generality of components that
 * need cooperative groups, instead of creating the group manually, consider
 * requesting one as an input parameter. Make sure its type is a template
 * parameter to maximize the set of groups for which your algorithm can be
 * invoked. To maximize the amount of contexts in which your algorithm can be
 * called and avoid hidden requirements, do not depend on a specific setup of
 * kernel launch parameters (i.e. grid dimensions and block dimensions).
 * Instead, use the thread_rank() method of the group to distinguish between
 * distinct threads of a group.
 *
 * The original DPCPP implementation does not provide ways to verify if a
 * certain type represents a cooperative group. Ginkgo's implementation provides
 * metafunctions which do that. Additionally, not all cooperative groups have
 * equivalent functionality, so Ginkgo splits the cooperative group concept into
 * three sub-concepts which describe what functionality is available. Here is a
 * list of concepts and their interfaces:
 *
 * ```c++
 * concept Group {
 *   unsigned size() const;
 *   unsigned thread_rank() const;
 * };
 *
 * concept SynchronizableGroup : Group {
 *   void sync();
 * };
 *
 * concept CommunicatorGroup : SynchronizableGroup {
 *   template <typename T>
 *   T shfl(T var, int srcLane);
 *   T shfl_up(T var, unsigned delta);
 *   T shfl_down(T var, unsigned delta);
 *   T shfl_xor(T var, int laneMask);
 *   int all(int predicate);
 *   int any(int predicate);
 *   unsigned ballot(int predicate);
 * };
 * ```
 *
 * To check if a group T satisfies one of the concepts, one can use the
 * metafunctions is_group<T>::value, is_synchronizable_group<T>::value and
 * is_communicator_group<T>::value.
 *
 * @note Please note that the current implementation of cooperative groups
 *       contains only a subset of functionalities provided by those APIs. If
 *       you need more functionality, please add the appropriate implementations
 *       to existing cooperative groups, or create new groups if the existing
 *       groups do not cover your use-case. For an example, see the
 *       enable_extended_shuffle mixin, which adds extended shuffles support
 *       to built-in DPCPP cooperative groups.
 */
namespace group {


// metafunctions
namespace detail {


template <typename T>
struct is_group_impl : std::false_type {};


template <typename T>
struct is_synchronizable_group_impl : std::false_type {};


template <typename T>
struct is_communicator_group_impl : std::true_type {};


}  // namespace detail


/**
 * Check if T is a Group.
 */
template <typename T>
using is_group = detail::is_group_impl<std::decay_t<T>>;


/**
 * Check if T is a SynchronizableGroup.
 */
template <typename T>
using is_synchronizable_group =
    detail::is_synchronizable_group_impl<std::decay_t<T>>;


/**
 * Check if T is a CommunicatorGroup.
 */
template <typename T>
using is_communicator_group =
    detail::is_communicator_group_impl<std::decay_t<T>>;


// types
namespace detail {


/**
 * This is a limited implementation of the DPCPP thread_block_tile.
 */
template <unsigned Size>
class thread_block_tile : public sycl::sub_group {
    using sub_group = sycl::sub_group;
    using id_type = sub_group::id_type;
    using mask_type = config::lane_mask_type;

public:
    // note: intel calls nd_item.get_sub_group(), but it still call
    // sycl::sub_group() to create the sub_group.
    template <typename Group>
    explicit thread_block_tile(const Group& parent_group)
        : data_{Size, 0}, sub_group()
    {
#ifndef NDEBUG
        assert(this->get_local_range().get(0) == Size);
#endif
        data_.rank = this->get_local_id();
    }


    __dpct_inline__ unsigned thread_rank() const noexcept { return data_.rank; }

    __dpct_inline__ unsigned size() const noexcept { return Size; }

    __dpct_inline__ void sync() const noexcept { this->barrier(); }

#define GKO_BIND_SHFL(ShflOpName, ShflOp)                                      \
    template <typename ValueType, typename SelectorType>                       \
    __dpct_inline__ ValueType ShflOpName(ValueType var, SelectorType selector) \
        const noexcept                                                         \
    {                                                                          \
        return this->ShflOp(var, selector);                                    \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

    GKO_BIND_SHFL(shfl, shuffle);
    GKO_BIND_SHFL(shfl_xor, shuffle_xor);

    // the shfl_up of out-of-range value gives undefined behavior, we
    // manually set it as the original value such that give the same result as
    // cuda/hip.
    template <typename ValueType, typename SelectorType>
    __dpct_inline__ ValueType shfl_up(ValueType var,
                                      SelectorType selector) const noexcept
    {
        const auto result = this->shuffle_up(var, selector);
        return (data_.rank < selector) ? var : result;
    }

    // the shfl_down of out-of-range value gives undefined behavior, we
    // manually set it as the original value such that give the same result as
    // cuda/hip.
    template <typename ValueType, typename SelectorType>
    __dpct_inline__ ValueType shfl_down(ValueType var,
                                        SelectorType selector) const noexcept
    {
        const auto result = this->shuffle_down(var, selector);
        return (data_.rank + selector >= Size) ? var : result;
    }

    /**
     * Returns a bitmask containing the value of the given predicate
     * for all threads in the group.
     * This means that the ith bit is equal to the predicate of the
     * thread with thread_rank() == i in the group.
     * Note that the whole group needs to execute the same operation.
     */
    __dpct_inline__ mask_type ballot(int predicate) const noexcept
    {
        // todo: change it when OneAPI update the mask related api
        return sycl::reduce_over_group(
            static_cast<sycl::sub_group>(*this),
            (predicate != 0) ? mask_type(1) << data_.rank : mask_type(0),
            sycl::plus<mask_type>());
    }

    /**
     * Returns true iff the predicate is true for at least one threads in the
     * group. Note that the whole group needs to execute the same operation.
     */
    __dpct_inline__ bool any(int predicate) const noexcept
    {
        return sycl::any_of_group(*this, (predicate != 0));
    }

    /**
     * Returns true iff the predicate is true for all threads in the group.
     * Note that the whole group needs to execute the same operation.
     */
    __dpct_inline__ bool all(int predicate) const noexcept
    {
        return sycl::all_of_group(*this, (predicate != 0));
    }


private:
    struct alignas(8) {
        unsigned size;
        unsigned rank;
    } data_;
};


// specialization for 1
template <>
class thread_block_tile<1> {
    using mask_type = config::lane_mask_type;
    static constexpr unsigned Size = 1;

public:
    template <typename Group>
    explicit thread_block_tile(const Group& parent_group) : data_{Size, 0}
    {}


    __dpct_inline__ unsigned thread_rank() const noexcept { return data_.rank; }

    __dpct_inline__ unsigned size() const noexcept { return Size; }

    __dpct_inline__ void sync() const noexcept {}


#define GKO_DISABLE_SHFL(ShflOpName)                                           \
    template <typename ValueType, typename SelectorType>                       \
    __dpct_inline__ ValueType ShflOpName(ValueType var, SelectorType selector) \
        const noexcept                                                         \
    {                                                                          \
        return var;                                                            \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

    GKO_DISABLE_SHFL(shfl);
    GKO_DISABLE_SHFL(shfl_up);
    GKO_DISABLE_SHFL(shfl_down);
    GKO_DISABLE_SHFL(shfl_xor);

    /**
     * Returns a bitmask containing the value of the given predicate
     * for all threads in the group.
     * This means that the ith bit is equal to the predicate of the
     * thread with thread_rank() == i in the group.
     * Note that the whole group needs to execute the same operation.
     */
    __dpct_inline__ mask_type ballot(int predicate) const noexcept
    {
        return (predicate != 0) ? mask_type(1) : mask_type(0);
    }

    /**
     * Returns true iff the predicate is true for at least one threads in the
     * group. Note that the whole group needs to execute the same operation.
     */
    __dpct_inline__ bool any(int predicate) const noexcept
    {
        return (predicate != 0);
    }

    /**
     * Returns true iff the predicate is true for all threads in the group.
     * Note that the whole group needs to execute the same operation.
     */
    __dpct_inline__ bool all(int predicate) const noexcept
    {
        return (predicate != 0);
    }


private:
    struct alignas(8) {
        unsigned size;
        unsigned rank;
    } data_;
};


}  // namespace detail


using detail::thread_block_tile;


// Only support tile_partition with 2, 4, 8, 16, 32, 64.
template <unsigned Size, typename Group>
__dpct_inline__
    std::enable_if_t<(Size > 1) && Size <= 64 && !(Size & (Size - 1)),
                     detail::thread_block_tile<Size>>
    tiled_partition(const Group& group)
{
    return detail::thread_block_tile<Size>(group);
}


template <unsigned Size, typename Group>
__dpct_inline__ std::enable_if_t<Size == 1, detail::thread_block_tile<Size>>
tiled_partition(const Group& group)
{
    return detail::thread_block_tile<Size>(group);
}


namespace detail {


template <unsigned Size>
struct is_group_impl<thread_block_tile<Size>> : std::true_type {};


template <unsigned Size>
struct is_synchronizable_group_impl<thread_block_tile<Size>> : std::true_type {
};


template <unsigned Size>
struct is_communicator_group_impl<thread_block_tile<Size>> : std::true_type {};


}  // namespace detail


class thread_block {
    friend __dpct_inline__ thread_block this_thread_block(sycl::nd_item<3>&);

public:
    __dpct_inline__ unsigned thread_rank() const noexcept { return data_.rank; }

    __dpct_inline__ unsigned size() const noexcept { return data_.size; }

    __dpct_inline__ void sync() const noexcept { group_.barrier(); }

private:
    __dpct_inline__ thread_block(sycl::nd_item<3>& group)
        : group_{group},
          data_{static_cast<unsigned>(group.get_local_range().size()),
                static_cast<unsigned>(group.get_local_linear_id())}
    {}
    struct alignas(8) {
        unsigned size;
        unsigned rank;
    } data_;

    sycl::nd_item<3>& group_;
};


__dpct_inline__ thread_block this_thread_block(sycl::nd_item<3>& group)
{
    return thread_block(group);
}


namespace detail {


template <>
struct is_group_impl<thread_block> : std::true_type {};


template <>
struct is_synchronizable_group_impl<thread_block> : std::true_type {};


}  // namespace detail


/**
 * This is a limited implementation of the DPCPP grid_group that works even on
 * devices that do not support device-wide synchronization and without special
 * kernel launch syntax.
 *
 * Note that this implementation does not support large grids, since it uses 32
 * bits to represent sizes and ranks, while at least 73 bits (63 bit grid + 10
 * bit block) would have to be used to represent the full space of thread ranks.
 */
class grid_group {
    friend __dpct_inline__ grid_group this_grid(sycl::nd_item<3>&);

public:
    __dpct_inline__ unsigned size() const noexcept { return data_.size; }

    __dpct_inline__ unsigned thread_rank() const noexcept { return data_.rank; }

private:
    __dpct_inline__ grid_group(sycl::nd_item<3>& group)
        : data_{static_cast<unsigned>(group.get_global_range().size()),
                static_cast<unsigned>(group.get_global_linear_id())}
    {}

    struct alignas(8) {
        unsigned size;
        unsigned rank;
    } data_;
};

// Not using this, as grid_group is not universally supported.
// grid_group this_grid()
// using cooperative_groups::this_grid;
// Instead, use our limited implementation:
__dpct_inline__ grid_group this_grid(sycl::nd_item<3>& group)
{
    return grid_group(group);
}


}  // namespace group
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


// Enable group can directly use group function
#if GINKGO_DPCPP_MAJOR_VERSION < 6
inline namespace cl {
#endif
namespace sycl {
namespace detail {


template <unsigned Size>
struct is_sub_group<
    ::gko::kernels::dpcpp::group::detail::thread_block_tile<Size>>
    : std::true_type {};


namespace spirv {


template <typename Group>
struct group_scope;

template <unsigned Size>
struct group_scope<
    ::gko::kernels::dpcpp::group::detail::thread_block_tile<Size>> {
    static constexpr __spv::Scope::Flag value = __spv::Scope::Flag::Subgroup;
};


}  // namespace spirv
}  // namespace detail
}  // namespace sycl
#if GINKGO_DPCPP_MAJOR_VERSION < 6
}  // namespace cl
#endif


#endif  // GKO_DPCPP_COMPONENTS_COOPERATIVE_GROUPS_DP_HPP_
