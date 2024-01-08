// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename IndexType>
struct overflowing_sum {
    constexpr static IndexType max = std::numeric_limits<IndexType>::max();
    constexpr static IndexType sentinel = -1;

    constexpr IndexType operator()(IndexType i, IndexType j) const
    {
        if (did_overflow(i) || did_overflow(j) || max - i < j) {
            return sentinel;
        }
        return i + j;
    }

    constexpr static bool did_overflow(IndexType i) { return i < 0; }
};


template <>
struct overflowing_sum<size_type> {
    constexpr static size_type max = std::numeric_limits<size_type>::max();
    constexpr static size_type sentinel = max;

    constexpr size_type operator()(size_type i, size_type j) const
    {
        if (did_overflow(i) || did_overflow(j) || max - i < j) {
            return sentinel;
        }
        return i + j;
    }

    constexpr static bool did_overflow(size_type i) { return i == sentinel; }
};


template <typename IndexType>
void prefix_sum_nonnegative(std::shared_ptr<const DefaultExecutor> exec,
                            IndexType* counts, size_type num_entries)
{
    constexpr auto max = std::numeric_limits<IndexType>::max();
    thrust::exclusive_scan(thrust_policy(exec), counts, counts + num_entries,
                           counts, IndexType{}, overflowing_sum<IndexType>{});
    if (num_entries > 0 &&
        overflowing_sum<IndexType>::did_overflow(
            exec->copy_val_to_host(counts + num_entries - 1))) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(IndexType)));
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PREFIX_SUM_NONNEGATIVE_KERNEL);

// instantiate for size_type as well, as this is used in the Sellp format
template void prefix_sum_nonnegative<size_type>(
    std::shared_ptr<const DefaultExecutor>, size_type*, size_type);
