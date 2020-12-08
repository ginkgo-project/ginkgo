/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

// --------------------------------------------------------------------------
//
// Copyright (C) 2009 - 2020 by the deal.II authors
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// --------------------------------------------------------------------------
//
// This index set class borrows the ideas and the core class implementation
// from deal.ii, but has been modified to adapt to Ginkgo's needs.
//
// --------------------------------------------------------------------------


#ifndef GKO_CORE_BASE_INDEX_SET_HPP_
#define GKO_CORE_BASE_INDEX_SET_HPP_


#include <algorithm>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/allocator.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/subset.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


/**
 * An index set class represents an ordered set object in a mathematical sense.
 * The index set contains subsets which store the starting and end points of a
 * range, [a,b), storing the first index and one past the last index. As the
 * index set only stores the end-points of ranges, it can be quite efficient in
 * terms of storage.
 *
 * This class automatically merges index ranges that are continuous when new
 * indices are added. To access the subsets, two iterators are provided, an
 * Element iterator which can iterate through the elements in the subsets stored
 * in the index set. An interval iterator is also provided to iterate between
 * the the different subsets themselves.
 *
 * The bulk of this class has been taken from the deal.ii finite element library
 * and it has been modified to adapt to Ginkgo's needs.
 *
 * @tparam index_type  type of the indices being stored in the index set.
 *
 * @ingroup IndexSet
 */
template <typename IndexType = int32>
class IndexSet {
public:
    class ElementIterator;
    class IntervalIterator;

    using index_type = IndexType;

    /**
     * Default constructor.
     */
    IndexSet() noexcept
        : is_merged_(true),
          largest_subset_(invalid_index_type<index_type>()),
          index_space_size_(0),
          exec_(nullptr),
          subsets_(exec_)
    {}

    /**
     * Constructor that also sets the overall size of the index range.
     */
    explicit IndexSet(std::shared_ptr<const gko::Executor> executor,
                      const index_type size)
        : is_merged_(true),
          largest_subset_(invalid_index_type<index_type>()),
          index_space_size_(size),
          exec_(executor),
          subsets_(exec_)
    {}

    /**
     * Copy constructor.
     */
    IndexSet(const IndexSet &other)
        : is_merged_(),
          largest_subset_(),
          index_space_size_(),
          merge_mutex_(),
          subsets_(other.exec_)
    {
        is_merged_ = other.is_merged_;
        subsets_ = other.subsets_;
        largest_subset_ = other.largest_subset_;
        index_space_size_ = other.index_space_size_;
        exec_ = other.exec_;
    }

    /**
     * Copy assignment operator.
     */
    IndexSet &operator=(const IndexSet &other)
    {
        is_merged_ = other.is_merged_;
        subsets_ = other.subsets_;
        largest_subset_ = other.largest_subset_;
        index_space_size_ = other.index_space_size_;
        exec_ = other.exec_;

        return *this;
    }

    /**
     * Move constructor. Create a new IndexSet by transferring the internal data
     * of the input set.
     */
    IndexSet(IndexSet &&other) noexcept;

    /**
     * Move assignment operator. Transfer the internal data of the input set
     * into the current one.
     */
    IndexSet &operator=(IndexSet &&other) noexcept
    {
        is_merged_ = other.is_merged_;
        subsets_ = std::move(other.subsets_);
        largest_subset_ = other.largest_subset_;
        index_space_size_ = other.index_space_size_;
        exec_ = other.exec_;

        other.subsets_.clear();
        other.is_merged_ = true;
        other.index_space_size_ = 0;
        other.largest_subset_ = invalid_index_type<index_type>();
        other.exec_ = nullptr;
        merge();

        return *this;
    }


    /**
     * Returns the Executor associated with the index set.
     *
     * @return the Executor associated with the index set
     */
    std::shared_ptr<const gko::Executor> get_executor() const noexcept
    {
        return exec_;
    }


    /**
     * Return the size of the index space of which this index set is a subset
     * of.
     *
     * Note that the result is not equal to the number of indices within this
     * set. The latter information is returned by get_num_elems().
     */
    index_type get_size() const { return index_space_size_; }

    /**
     * Set the maximal size of the indices upon which this object operates.
     *
     * This function can only be called if the index set does not yet contain
     * any elements.  This can be achieved by calling clear(), for example.
     */
    void set_size(const index_type input_size)
    {
        GKO_ASSERT_CONDITION(subsets_.empty());
        index_space_size_ = input_size;
        is_merged_ = true;
    }

    /**
     * Merge the internal representation by merging individual elements with
     * contiguous subsets, etc. This function does not have any external effect.
     */
    void merge() const;

    /**
     * Add a single dens row of a certain stride to the set of
     * indices represented by this class.
     * @param[in] row The row of the matrix to be added.
     * @param[in] stride The stride of the row.
     *
     * @note Rows start from 0.
     */
    void add_dense_row(const index_type row, const index_type stride);

    /**
     * Add the dense rows [begin,end] of a certain stride to the set of
     * indices represented by this class.
     * @param[in] begin The first row of the matrix to be added.
     * @param[in] end The last row of the matrix to be added.
     * @param[in] stride The stride of the rows.
     *
     * @note Rows start from 0.
     */
    void add_dense_rows(const index_type begin, const index_type end,
                        const index_type stride);

    /**
     * Add a single sparse row to the set of
     * indices represented by this class.
     * @param[in] row The row of the matrix to be added.
     * @param[in] nnz_in_row The number of nonzeros in the rows.
     *
     * @note Rows start from 0.
     */
    void add_sparse_row(const index_type row, const index_type nnz_in_row);

    /**
     * Add the sparse rows of a matrix to the set of
     * indices represented by this class.
     * @param[in] begin The first row of to be added.
     * @param[in] end The last row of the subset to be added.
     * @param[in] nnz_per_row The Array containing the number of nnz per row for
     * all the rows from [begin,end]
     *
     * @note Rows start from 0.
     */
    void add_sparse_rows(const index_type begin, const index_type end,
                         const gko::vector<index_type> &nnz_per_row);

    /**
     * Add the half-open subset $[\text{begin},\text{end})$ to the set of
     * indices represented by this class.
     * @param[in] begin The first element of the subset to be added.
     * @param[in] end The past-the-end element of the subset to be added.
     */
    void add_subset(const index_type begin, const index_type end);

    /**
     * Add the half-open subset $[\text{begin},\text{end})$ to the set of
     * indices represented by this class.
     * @param[in] subset The subset to be added.
     */
    void add_subset(Subset<index_type> &subset);

    /**
     * Add an individual index to the set of indices.
     */
    void add_index(const index_type index);

    /**
     * Add the given IndexSet @p other to the current one, constructing the
     * union of *this and @p other.
     *
     * If the @p offset argument is nonzero, then every index in @p other is
     * shifted by @p offset before being added to the current index set. This
     * allows to construct, for example, one index set from several others that
     * are supposed to represent index sets corresponding to different subsets_
     * (e.g., when constructing the set of nonzero entries of a block vector
     * from the sets of nonzero elements of the individual blocks of a vector).
     *
     * This function will generate an exception if any of the (possibly shifted)
     * indices of the @p other index set lie outside the subset
     * <code>[0,size())</code> represented by the current object.
     */
    void add_indices(const IndexSet &other, const index_type offset = 0);

    /**
     * Add a whole set of indices described by dereferencing every element of
     * the iterator subset <code>[begin,end)</code>.
     *
     * @param[in] begin Iterator to the first element of subset of indices to be
     * added
     * @param[in] end The past-the-end iterator for the subset of elements to be
     * added. @pre The condition <code>begin@<=end</code> needs to be satisfied.
     */
    template <typename ForwardIterator>
    void add_indices(const ForwardIterator &begin, const ForwardIterator &end)
    {
        if (begin == end) return;

        // identify subsets_ in the given iterator subset by checking whether
        // some indices happen to be consecutive. to avoid quadratic complexity
        // when calling add_subset many times (as add_subset() going into the
        // middle of an already existing subset must shift entries around), we
        // first collect a vector of subsets_.
        gko::vector<std::pair<index_type, index_type>> tmp_subsets(exec_);
        bool subsets_are_sorted = true;
        for (ForwardIterator p = begin; p != end;) {
            const index_type begin_index = *p;
            index_type end_index = begin_index + 1;
            ForwardIterator q = p;
            ++q;
            while ((q != end) && (*q == end_index)) {
                ++end_index;
                ++q;
            }

            tmp_subsets.emplace_back(begin_index, end_index);
            p = q;

            // if the starting index of the next go-around of the for loop is
            // less than the end index of the one just identified, then we will
            // have at least one pair of subsets_ that are not sorted, and
            // consequently the whole collection of subsets_ is not sorted.
            if (p != end && *p < end_index) subsets_are_sorted = false;
        }

        if (!subsets_are_sorted)
            std::sort(tmp_subsets.begin(), tmp_subsets.end());

        // if we have many subsets_, we first construct a temporary index set
        // (where we add subsets_ in a consecutive way, so fast), otherwise, we
        // work with add_subset(). the number 9 is chosen heuristically given
        // the fact that there are typically up to 8 independent subsets_ when
        // adding the degrees of freedom on a 3D cell or 9 when adding degrees
        // of freedom of faces. if doing cell-by-cell additions, we want to
        // avoid repeated calls to IndexSet::merge() which gets called upon
        // merging two index sets, so we want to be in the other branch then.
        if (tmp_subsets.size() > 9) {
            IndexSet<index_type> tmp_set(exec_, get_size());
            tmp_set.subsets_.reserve(tmp_subsets.size());
            for (const auto &i : tmp_subsets)
                tmp_set.add_subset(i.first, i.second);
            this->add_indices(tmp_set);
        } else
            for (const auto &i : tmp_subsets) add_subset(i.first, i.second);
    }

    /**
     * Return whether the specified index is an element of the index set.
     */
    bool is_element(const index_type index) const;

    /**
     * Return whether the index set stored by this object defines a contiguous
     * subset. This is true also if no indices are stored at all.
     */
    bool is_contiguous() const;

    /**
     * Return whether the index set stored by this object contains no elements.
     * This is similar, but faster than checking <code>get_num_elems() ==
     * 0</code>.
     */
    bool is_empty() const { return subsets_.empty(); }


    /**
     * Return the number of elements stored in this index set.
     */
    index_type get_num_elems() const;

    /**
     * Return the global index of the local index with number @p local_index
     * stored in this index set. @p local_index obviously needs to be less than
     * get_num_elems().
     */
    index_type get_global_index(const index_type local_index) const;

    /**
     * Return the how-manyth element of this set (counted in ascending order) @p
     * global_index is. @p global_index needs to be less than the size. This
     * function returns invalid_index if the index @p global_index
     * is not actually a member of this index set, i.e. if
     * is_element(global_index) is false.
     */
    index_type get_local_index(const index_type global_index) const;

    /**
     * Each index set can be represented as the union of a number of contiguous
     * intervals of indices, where if necessary intervals may only consist of
     * individual elements to represent isolated members of the index set.
     *
     * This function returns the minimal number of such intervals that are
     * needed to represent the index set under consideration.
     */
    index_type get_num_subsets() const;

    /**
     * This function returns the local index of the beginning of the largest
     * subset.
     *
     * In other words, the return value is superset_index(x), where x is the
     * first index of the largest contiguous subset of indices in the
     * IndexSet. The return value is therefore equal to the number of elements
     * in the set that come before the largest subset.
     *
     * This call assumes that the IndexSet is nonempty.
     */
    index_type get_largest_subset_starting_index() const;

    /**
     * This function returns the largest element in the subset
     *
     * This call assumes that the IndexSet is nonempty.
     */
    index_type get_largest_element_in_set() const;

    /**
     * Comparison for equality of index sets. This operation is only allowed if
     * the size of the two sets is the same (though of course they do not have
     * to have the same number of indices).
     */
    bool operator==(const IndexSet &other) const;

    /**
     * Comparison for inequality of index sets. This operation is only allowed
     * if the size of the two sets is the same (though of course they do not
     * have to have the same number of indices).
     */
    bool operator!=(const IndexSet &other) const;

    /**
     * Return the intersection of the current index set and the argument given,
     * i.e. a set of indices that are elements of both index sets. The two index
     * sets must have the same size (though of course they do not have to have
     * the same number of indices).
     */
    IndexSet operator&(const IndexSet &other) const;


    /**
     * Remove all elements contained in @p other from this set. In other words,
     * if $x$ is the current object and $o$ the argument, then we compute $x
     * \leftarrow x \backslash o$.
     */
    void subtract_set(const IndexSet &other);


    /**
     * Remove and return the last element of the last subset.
     * This function throws an exception if the IndexSet is empty.
     */
    index_type pop_back();

    /**
     * Remove and return the first element of the first subset.
     * This function throws an exception if the IndexSet is empty.
     */
    index_type pop_front();

    /**
     * Remove all indices from this index set. The index set retains its size,
     * however.
     */
    void clear()
    {
        // reset so that there are no indices in the set any more; however,
        // as documented, the index set retains its size
        subsets_.clear();
        is_merged_ = true;
        largest_subset_ = invalid_index_type<index_type>();
    }


    /**
     * Dereferencing an IntervalIterator will return a reference to an object of
     * this type. It allows access to a contiguous interval $[a,b[$ (also called
     * a subset) of the IndexSet being iterated over.
     * This class is a modified version of the class from the deal.ii library.
     */
    class IntervalAccessor {
    public:
        /**
         * Construct a valid accessor given an IndexSet and the index @p
         * subset_idx of the subset to point to.
         */
        IntervalAccessor(const IndexSet *idxset, const index_type subset_idx)
            : index_set_(idxset), subset_idx_(subset_idx)
        {}

        /**
         * Construct an invalid accessor for the IndexSet.
         */
        explicit IntervalAccessor(const IndexSet *idxset)
            : index_set_(idxset), subset_idx_(invalid_index_type<index_type>())
        {}

        /**
         * Number of elements in this interval.
         */
        index_type get_num_elems() const
        {
            return index_set_->subsets_[subset_idx_].end_ -
                   index_set_->subsets_[subset_idx_].begin_;
        }

        /**
         * If true, we are pointing at a valid interval in the IndexSet.
         */
        bool is_valid() const
        {
            return index_set_ != nullptr &&
                   subset_idx_ < index_set_->get_num_elems();
        }

        /**
         * Return an iterator pointing at the first index in this interval.
         */
        ElementIterator begin() const
        {
            GKO_ASSERT_CONDITION(is_valid());
            return {index_set_, subset_idx_,
                    index_set_->subsets_[subset_idx_].begin_};
        }

        /**
         * Return an iterator pointing directly after the last index in this
         * interval.
         */
        ElementIterator end() const
        {
            GKO_ASSERT_CONDITION(is_valid());
            if (subset_idx_ < index_set_->subsets_.size() - 1)
                return {index_set_, subset_idx_ + 1,
                        index_set_->subsets_[subset_idx_ + 1].begin_};
            else
                return index_set_->end();
        }

        /**
         * Return the index of the last index in this interval.
         */
        index_type last() const
        {
            GKO_ASSERT_CONDITION(is_valid());
            return index_set_->subsets_[subset_idx_].end_ - 1;
        }

    private:
        /**
         * Private copy constructor.
         */
        IntervalAccessor(const IntervalAccessor &other)
            : index_set_(other.index_set_), subset_idx_(other.subset_idx_)
        {}

        /**
         * Private copy operator.
         */
        IntervalAccessor &operator=(const IntervalAccessor &other)
        {
            index_set_ = other.index_set_;
            subset_idx_ = other.subset_idx_;
            GKO_ASSERT_CONDITION(
                subset_idx_ == invalid_index_type<index_type>() || is_valid());
            return *this;
        }

        /**
         * Test for equality, used by IntervalIterator.
         */
        bool operator==(const IntervalAccessor &other) const
        {
            GKO_ASSERT_CONDITION(index_set_ == other.index_set_);
            return subset_idx_ == other.subset_idx_;
        }

        /**
         * Smaller-than operator, used by IntervalIterator.
         */
        bool operator<(const IntervalAccessor &other) const
        {
            GKO_ASSERT_CONDITION(index_set_ == other.index_set_);
            return subset_idx_ < other.subset_idx_;
        }

        /**
         * Advance this accessor to point to the next interval in the @p
         * index_set.
         */
        void advance()
        {
            GKO_ASSERT_CONDITION(is_valid());
            ++subset_idx_;

            // set ourselves to invalid if we walk off the end
            if (subset_idx_ >= index_set_->subsets_.size()) {
                subset_idx_ = invalid_index_type<index_type>();
            }
        }

        /**
         * Reference to the IndexSet.
         */
        const IndexSet *index_set_;

        /**
         * Index into index_set.subsets[]. Set to numbers::invalid_dof_index if
         * invalid or the end iterator.
         */
        index_type subset_idx_;

        friend class IntervalIterator;
    };

    /**
     * Class that represents an iterator pointing to a contiguous interval
     * $[a,b[$ as returned by IndexSet::begininterval().
     * This class a modified version of the class from the deal.ii library.
     */
    class IntervalIterator {
    public:
        /**
         * Construct a valid iterator pointing to the interval with index @p
         * subset_idx.
         */
        IntervalIterator(const IndexSet *idxset, const index_type subset_idx)
            : accessor_(idxset, subset_idx)
        {}

        /**
         * Construct an invalid iterator (used as end()).
         */
        explicit IntervalIterator(const IndexSet *idxset) : accessor_(idxset) {}

        /**
         * Construct an empty iterator.
         */
        IntervalIterator() : accessor_(nullptr) {}

        /**
         * Copy constructor from @p other iterator.
         */
        IntervalIterator(const IntervalIterator &other) = default;

        /**
         * Assignment of another iterator.
         */
        IntervalIterator &operator=(const IntervalIterator &other) = default;

        /**
         * Prefix increment.
         */
        IntervalIterator &operator++()
        {
            accessor_.advance();
            return *this;
        }

        /**
         * Postfix increment.
         */
        IntervalIterator operator++(int)
        {
            const IndexSet::IntervalIterator iter = *this;
            accessor_.advance();
            return iter;
        }

        /**
         * Dereferencing operator, returns an IntervalAccessor.
         */
        const IntervalAccessor &operator*() const { return accessor_; }

        /**
         * Dereferencing operator, returns a pointer to an IntervalAccessor.
         */
        const IntervalAccessor *operator->() const { return &accessor_; }

        /**
         * Comparison.
         */
        bool operator==(const IntervalIterator &other) const
        {
            return accessor_ == other.accessor_;
        }

        /**
         * Inverse of <tt>==</tt>.
         */
        bool operator!=(const IntervalIterator &other) const
        {
            return !(*this == other);
        }

        /**
         * Comparison operator.
         */
        bool operator<(const IntervalIterator &other) const
        {
            return accessor_ < other.accessor_;
        }

        /**
         * Return the distance between the current iterator and the argument.
         * The distance is given by how many times one has to apply operator++
         * to the current iterator to get the argument (for a positive return
         * value), or operator-- (for a negative return value).
         */
        int operator-(const IntervalIterator &other) const
        {
            GKO_ASSERT_CONDITION(accessor_.index_set_ ==
                                 other.accessor_.index_set_);

            const index_type lhs =
                (accessor_.subset_idx_ == invalid_index_type<index_type>())
                    ? accessor_.index_set_->subsets_.size()
                    : accessor_.subset_idx_;
            const index_type rhs = (other.accessor_.subset_idx_ ==
                                    invalid_index_type<index_type>())
                                       ? accessor_.index_set_->subsets_.size()
                                       : other.accessor_.subset_idx_;

            if (lhs > rhs)
                return static_cast<index_type>(lhs - rhs);
            else
                return -static_cast<index_type>(rhs - lhs);
        }

        /**
         * Mark the class as forward iterator and declare some alias which are
         * standard for iterators and are used by algorithms to enquire about
         * the specifics of the iterators they work on.
         */
        using iterator_category = std::forward_iterator_tag;
        using value_type = IntervalAccessor;
        using difference_type = std::ptrdiff_t;
        using pointer = IntervalAccessor *;
        using reference = IntervalAccessor &;

    private:
        /**
         * Accessor that contains what IndexSet and interval we are pointing at.
         */
        IntervalAccessor accessor_;
    };

    /**
     * Class that represents an iterator pointing to a single element in the
     * IndexSet as returned by IndexSet::begin().
     */
    class ElementIterator {
    public:
        /**
         * Construct an iterator pointing to the global index @p index in the
         * interval @p subset_index
         */
        ElementIterator(const IndexSet *indexset, const index_type subset_index,
                        const index_type index)
            : index_set_(indexset), subset_index_(subset_index), index_(index)
        {
            GKO_ASSERT_CONDITION(subset_index_ < index_set_->subsets_.size());
            GKO_ASSERT_CONDITION(
                index_ >= index_set_->subsets_[subset_index_].begin_ &&
                index_ < index_set_->subsets_[subset_index_].end_);
        }

        /**
         * Construct an iterator pointing to the end of the IndexSet.
         */
        explicit ElementIterator(const IndexSet *index_set)
            : index_set_(index_set),
              subset_index_(invalid_index_type<index_type>()),
              index_(invalid_index_type<index_type>())
        {}

        /**
         * Does this iterator point to an existing element?
         */
        bool is_valid() const
        {
            GKO_ASSERT_CONDITION(
                (subset_index_ == invalid_index_type<index_type>() &&
                 index_ == invalid_index_type<index_type>()) ||
                (subset_index_ < index_set_->subsets_.size() &&
                 index_ < index_set_->subsets_[subset_index_].end_));

            return (subset_index_ < index_set_->subsets_.size() &&
                    index_ < index_set_->subsets_[subset_index_].end_);
        }

        /**
         * Dereferencing operator. The returned value is the index of the
         * element inside the IndexSet.
         */
        index_type operator*() const
        {
            GKO_ASSERT_CONDITION(is_valid());
            return index_;
        }

        /**
         * Prefix increment.
         */
        ElementIterator &operator++()
        {
            advance();
            return *this;
        }

        /**
         * Postfix increment.
         */
        ElementIterator operator++(int)
        {
            const IndexSet::ElementIterator it = *this;
            advance();
            return it;
        }

        /**
         * Comparison.
         */
        bool operator==(const ElementIterator &other) const
        {
            GKO_ASSERT_CONDITION(index_set_ == other.index_set_);
            return subset_index_ == other.subset_index_ &&
                   index_ == other.index_;
        }

        /**
         * Inverse of <tt>==</tt>.
         */
        bool operator!=(const ElementIterator &other) const
        {
            return !(*this == other);
        }

        /**
         * Comparison operator.
         */
        bool operator<(const ElementIterator &other) const
        {
            GKO_ASSERT_CONDITION(index_set_ == other.index_set_);
            return subset_index_ < other.subset_index_ ||
                   (subset_index_ == other.subset_index_ &&
                    index_ < other.index_);
        }

        /**
         * Return the distance between the current iterator and the argument. In
         * the expression <code>it_left-it_right</code> the distance is given by
         * how many times one has to apply operator++ to the right operand @p
         * it_right to get the left operand @p it_left (for a positive return
         * value), or to @p it_left to get the @p it_right (for a negative
         * return value).
         */
        std::ptrdiff_t operator-(const ElementIterator &other) const
        {
            GKO_ASSERT_CONDITION(index_set_ == other.index_set_);
            if (*this == other) return 0;
            if (!(*this < other)) return -(other - *this);

            // only other can be equal to end() because of the checks above.
            GKO_ASSERT_CONDITION(is_valid());

            // Note: we now compute how far advance *this in "*this < other" to
            // get other, so we need to return -c at the end.

            // first finish the current subset:
            std::ptrdiff_t c =
                index_set_->subsets_[subset_index_].end_ - index_;

            // now walk in steps of subsets_ (need to start one behind our
            // current one):
            for (index_type subset = subset_index_ + 1;
                 subset < index_set_->subsets_.size() &&
                 subset <= other.subset_index_;
                 ++subset)
                c += index_set_->subsets_[subset].end_ -
                     index_set_->subsets_[subset].begin_;

            GKO_ASSERT_CONDITION(
                other.subset_index_ < index_set_->subsets_.size() ||
                other.subset_index_ == invalid_index_type<index_type>());

            // We might have walked too far because we went until the end of
            // other.subset_index, so walk backwards to other.index:
            if (other.subset_index_ != invalid_index_type<index_type>())
                c -= index_set_->subsets_[other.subset_index_].end_ -
                     other.index_;

            return -c;
        }

        /**
         * Mark the class as forward iterator and declare some alias which are
         * standard for iterators and are used by algorithms to enquire about
         * the specifics of the iterators they work on.
         */
        using iterator_category = std::forward_iterator_tag;
        using value_type = index_type;
        using difference_type = std::ptrdiff_t;
        using pointer = index_type *;
        using reference = index_type &;

    private:
        /**
         * Advance iterator by one.
         */
        void advance()
        {
            GKO_ASSERT_CONDITION(is_valid());
            if (index_ < index_set_->subsets_[subset_index_].end_) ++index_;
            // end of this subset?
            if (index_ == index_set_->subsets_[subset_index_].end_) {
                // point to first element in next interval if possible
                if (subset_index_ < index_set_->subsets_.size() - 1) {
                    ++subset_index_;
                    index_ = index_set_->subsets_[subset_index_].begin_;
                } else {
                    // we just fell off the end, set to invalid:
                    subset_index_ = invalid_index_type<index_type>();
                    index_ = invalid_index_type<index_type>();
                }
            }
        }

        /**
         * The parent IndexSet.
         */
        const IndexSet *index_set_;

        /**
         * Index into set
         */
        index_type subset_index_;

        /**
         * The global index this iterator is pointing at.
         */
        index_type index_;
    };

    /**
     * Return an iterator that points at the first index that is contained in
     * this IndexSet.
     */
    ElementIterator begin() const;

    /**
     * Return an element iterator pointing to the element with global index
     * @p global_index or the next larger element if the index is not in the
     * set. This is equivalent to
     * @code
     * auto p = begin();
     * while (*p<global_index)
     *   ++p;
     * return p;
     * @endcode
     *
     * If there is no element in this IndexSet at or behind @p global_index,
     * this method will return end().
     */
    ElementIterator at(const index_type global_index) const;

    /**
     * Return an iterator that points one after the last index that is contained
     * in this IndexSet.
     */
    ElementIterator end() const;

    /**
     * Return an Iterator that points at the first interval of this IndexSet.
     */
    IntervalIterator get_first_interval() const;

    /**
     * Return an Iterator that points one after the last interval of this
     * IndexSet.
     */
    IntervalIterator get_last_interval() const;

private:
    void merge_impl() const;

    std::shared_ptr<const gko::Executor> exec_;

    mutable bool is_merged_;
    mutable index_type largest_subset_;
    mutable std::mutex merge_mutex_;
    index_type index_space_size_;
    mutable gko::vector<Subset<index_type>> subsets_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_INDEX_SET_HPP_
