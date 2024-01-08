// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_INDEX_SET_HPP_
#define GKO_PUBLIC_CORE_BASE_INDEX_SET_HPP_


#include <algorithm>
#include <initializer_list>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


/**
 * An index set class represents an ordered set of intervals. The index set
 * contains subsets which store the starting and end points of a range,
 * [a,b), storing the first index and one past the last index. As the
 * index set only stores the end-points of ranges, it can be quite efficient in
 * terms of storage.
 *
 * This class is particularly useful in storing continuous ranges. For example,
 * consider the index set (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 18, 19, 20, 21,
 * 42). Instead of storing the entire array of indices, one can store intervals
 * ([1,9), [10,13), [18,22), [42,43)), thereby only using half the storage.
 *
 * We store three arrays, one (subsets_begin) with the starting indices of the
 * subsets in the index set, another (subsets_end) storing one
 * index beyond the end indices of the subsets and the last
 * (superset_cumulative_indices) storing the cumulative number of indices in the
 * subsequent subsets with an initial zero which speeds up the
 * querying. Additionally, the arrays conataining the range boundaries
 * (subsets_begin, subsets_end) are stored in a sorted fashion.
 *
 * Therefore the storage would look as follows
 *
 * > index_set = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 18, 19, 20, 21, 42)
 * > subsets_begin = {1, 10, 18, 42}
 * > subsets_end = {9, 13, 22, 43}
 * > superset_cumulative_indices = {0, 8, 11, 15, 16}
 *
 * @tparam index_type  type of the indices being stored in the index set.
 *
 * @ingroup index_set
 */
template <typename IndexType = int32>
class index_set {
public:
    /**
     * The type of elements stored in the index set.
     */
    using index_type = IndexType;

    /**
     * Creates an empty index_set tied to the specified Executor.
     *
     * @param exec  the Executor where the index_set data is allocated
     */
    explicit index_set(std::shared_ptr<const Executor> exec) noexcept
        : exec_(std::move(exec)),
          index_space_size_{0},
          num_stored_indices_{0},
          subsets_begin_{array<index_type>(exec_)},
          subsets_end_{array<index_type>(exec_)},
          superset_cumulative_indices_{array<index_type>(exec_)}
    {}

    /**
     * Creates an index set on the specified executor from the initializer list.
     *
     * @param exec  the Executor where the index set data will be allocated
     * @param init_list  the indices that the index set should hold in an
     *                   initializer_list.
     * @param is_sorted  a parameter that specifies if the indices array is
     *                   sorted or not. `true` if sorted.
     */
    explicit index_set(std::shared_ptr<const gko::Executor> exec,
                       std::initializer_list<IndexType> init_list,
                       const bool is_sorted = false)
        : exec_(std::move(exec)),
          index_space_size_(init_list.size() > 0
                                ? *(std::max_element(std::begin(init_list),
                                                     std::end(init_list))) +
                                      1
                                : 0),
          num_stored_indices_{static_cast<IndexType>(init_list.size())}
    {
        GKO_ASSERT(index_space_size_ > 0);
        this->populate_subsets(
            array<IndexType>(this->get_executor(), init_list), is_sorted);
    }

    /**
     * Creates an index set on the specified executor and the given size
     *
     * @param exec  the Executor where the index set data will be allocated
     * @param size  the maximum index the index set it allowed to hold. This
     *              is the size of the index space.
     * @param indices  the indices that the index set should hold.
     * @param is_sorted  a parameter that specifies if the indices array is
     *                   sorted or not. `true` if sorted.
     */
    explicit index_set(std::shared_ptr<const gko::Executor> exec,
                       const index_type size,
                       const gko::array<index_type>& indices,
                       const bool is_sorted = false)
        : exec_(std::move(exec)), index_space_size_(size)
    {
        GKO_ASSERT(index_space_size_ >= indices.get_size());
        this->populate_subsets(indices, is_sorted);
    }

    /**
     * Creates a copy of the input index_set on a different executor.
     *
     * @param exec  the executor where the new index_set will be created
     * @param other  the index_set to copy from
     */
    index_set(std::shared_ptr<const Executor> exec, const index_set& other)
        : index_set(exec)
    {
        *this = other;
    }

    /**
     * Creates a copy of the input index_set.
     *
     * @param other the index_set to copy from
     */
    index_set(const index_set& other) : index_set(other.get_executor(), other)
    {}

    /**
     * Moves the input index_set to a different executor.
     *
     * @param exec  the executor where the new index_set will be moved to
     * @param other the index_set to move from
     */
    index_set(std::shared_ptr<const Executor> exec, index_set&& other)
        : index_set(exec)
    {
        *this = std::move(other);
    }

    /**
     * Moves the input index_set.
     *
     * @param other the index_set to move from
     */
    index_set(index_set&& other)
        : index_set(other.get_executor(), std::move(other))
    {}

    /**
     * Copies data from another index_set
     *
     * The executor of this is preserved. In case this does not have an assigned
     * executor, it will inherit the executor of other.
     *
     * @param other  the index_set to copy from
     *
     * @return this
     */
    index_set& operator=(const index_set& other)
    {
        if (&other == this) {
            return *this;
        }
        this->index_space_size_ = other.index_space_size_;
        this->num_stored_indices_ = other.num_stored_indices_;
        this->subsets_begin_ = other.subsets_begin_;
        this->subsets_end_ = other.subsets_end_;
        this->superset_cumulative_indices_ = other.superset_cumulative_indices_;

        return *this;
    }

    /**
     * Moves data from another index_set
     *
     * The executor of this is preserved. In case this does not have an assigned
     * executor, it will inherit the executor of other.
     *
     * @param other  the index_set to move from
     *
     * @return this
     */
    index_set& operator=(index_set&& other)
    {
        if (&other == this) {
            return *this;
        }
        this->index_space_size_ = std::exchange(other.index_space_size_, 0);
        this->num_stored_indices_ = std::exchange(other.num_stored_indices_, 0);
        this->subsets_begin_ = std::move(other.subsets_begin_);
        this->subsets_end_ = std::move(other.subsets_end_);
        this->superset_cumulative_indices_ =
            std::move(other.superset_cumulative_indices_);

        return *this;
    }

    /**
     * Deallocates all data used by the index_set.
     *
     * The index_set is left in a valid, but empty state, so the same index_set
     * can be used to allocate new memory. Calls to
     * index_set::get_subsets_begin() will return a `nullptr`.
     */
    void clear() noexcept
    {
        this->index_space_size_ = 0;
        this->num_stored_indices_ = 0;
        this->subsets_begin_.clear();
        this->subsets_end_.clear();
        this->superset_cumulative_indices_.clear();
    }

    /**
     * Returns the executor of the index_set
     *
     * @return  the executor.
     */
    std::shared_ptr<const Executor> get_executor() const { return this->exec_; }

    /**
     * Returns the size of the index set space.
     *
     * @return  the size of the index set space.
     */
    index_type get_size() const { return this->index_space_size_; }

    /**
     * Returns if the index set is contiguous
     *
     * @return  if the index set is contiguous.
     */
    bool is_contiguous() const { return (this->get_num_subsets() <= 1); }

    /**
     * Return the actual number of indices stored in the index set
     *
     * @return  number of indices stored in the index set
     */
    index_type get_num_elems() const { return this->num_stored_indices_; };

    /**
     * Return the global index given a local index.
     *
     * Consider the set idx_set = (0, 1, 2, 4, 6, 7, 8, 9). This function
     * returns the element at the global index k stored in the index set. For
     * example, `idx_set.get_global_index(0) == 0` `idx_set.get_global_index(3)
     * == 4` and `idx_set.get_global_index(7) == 9`
     *
     * @note This function returns a scalar value and needs a scalar value.
     *       For repeated queries, it is more efficient to use the array
     *       functions that take and return arrays which allow for more
     *       throughput.
     *
     * @param local_index  the local index.
     * @return  the global index from the index set.
     *
     * @warning This single entry query can have significant kernel launch
     *          overheads and should be avoided if possible.
     */
    index_type get_global_index(index_type local_index) const;

    /**
     * Return the local index given a global index.
     *
     * Consider the set idx_set = (0, 1, 2, 4, 6, 7, 8, 9). This function
     * returns the local index in the index set of the provided index set. For
     * example, `idx_set.get_local_index(0) == 0` `idx_set.get_local_index(4)
     * == 3` and `idx_set.get_local_index(6) == 4`.
     *
     * @note This function returns a scalar value and needs a scalar value.
     *       For repeated queries, it is more efficient to use the array
     *       functions that take and return arrays which allow for more
     *       throughput.
     *
     * @param global_index  the global index.
     *
     * @return  the local index of the element in the index set.
     *
     * @warning This single entry query can have significant kernel launch
     *          overheads and should be avoided if possible.
     */
    index_type get_local_index(index_type global_index) const;

    /**
     * This is an array version of the scalar function above.
     *
     * @param local_indices  the local index array.
     * @param is_sorted  a parameter that specifies if the query array is sorted
     *                   or not. `true` if sorted .
     *
     * @return  the global index array from the index set.
     *
     * @note Whenever possible, passing a sorted array is preferred as the
     *       queries can be significantly faster.
     * @note Passing local indices from [0, size) is equivalent to using the
     *       @to_global_indices function.
     */
    array<index_type> map_local_to_global(
        const array<index_type>& local_indices,
        const bool is_sorted = false) const;

    /**
     * This is an array version of the scalar function above.
     *
     * @param global_indices  the global index array.
     * @param is_sorted  a parameter that specifies if the query array is sorted
     *                   or not. `true` if sorted.
     *
     * @return  the local index array from the index set.
     *
     * @note Whenever possible, passing a sorted array is preferred as the
     *       queries can be significantly faster.
     */
    array<index_type> map_global_to_local(
        const array<index_type>& global_indices,
        const bool is_sorted = false) const;

    /**
     * This function allows the user obtain a decompressed global_indices array
     * from the indices stored in the index set
     *
     * @return  the decompressed set of indices.
     */
    array<index_type> to_global_indices() const;

    /**
     * Checks if the individual global indeices exist in the index set.
     *
     * @param global_indices  the indices to check.
     * @param is_sorted  a parameter that specifies if the query array is sorted
     *                   or not. `true` if sorted.
     *
     * @return  the array that contains element wise whether the corresponding
     *          global index in the index set or not.
     */
    array<bool> contains(const array<index_type>& global_indices,
                         const bool is_sorted = false) const;

    /**
     * Checks if the global index exists in the index set.
     *
     * @param global_index  the index to check.
     *
     * @return  whether the element exists in the index set.
     *
     * @warning This single entry query can have significant kernel launch
     *          overheads and should be avoided if possible.
     */
    bool contains(const index_type global_index) const;

    /**
     * Returns the number of subsets stored in the index set.
     *
     * @return  the number of stored subsets.
     */
    index_type get_num_subsets() const
    {
        return this->subsets_begin_.get_size();
    }

    /**
     * Returns a pointer to the beginning indices of the subsets.
     *
     * @return  a pointer to the beginning indices of the subsets.
     */
    const index_type* get_subsets_begin() const
    {
        return this->subsets_begin_.get_const_data();
    }

    /**
     * Returns a pointer to the end indices of the subsets.
     *
     * @return  a pointer to the end indices of the subsets.
     */
    const index_type* get_subsets_end() const
    {
        return this->subsets_end_.get_const_data();
    }

    /**
     * Returns a pointer to the cumulative indices of the superset of
     * the subsets.
     *
     * @return  a pointer to the cumulative indices of the superset of the
     *          subsets.
     */
    const index_type* get_superset_indices() const
    {
        return this->superset_cumulative_indices_.get_const_data();
    }

private:
    void populate_subsets(const gko::array<index_type>& indices,
                          const bool is_sorted);

    std::shared_ptr<const Executor> exec_;
    index_type index_space_size_;
    index_type num_stored_indices_;
    gko::array<index_type> subsets_begin_;
    gko::array<index_type> subsets_end_;
    gko::array<index_type> superset_cumulative_indices_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_INDEX_SET_HPP_
