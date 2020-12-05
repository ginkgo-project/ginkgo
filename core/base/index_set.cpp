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

#include <ginkgo/core/base/index_set.hpp>


// ---------------------------------------------------------------------
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


#include <algorithm>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {


template <typename IndexType>
IndexSet<IndexType>::IndexSet(IndexSet<IndexType> &&other) noexcept
    : subsets_(std::move(other.subsets_)),
      is_merged_(other.is_merged_),
      index_space_size_(other.index_space_size_),
      largest_subset_(other.largest_subset_),
      exec_(other.exec_)
{
    other.subsets_.clear();
    other.is_merged_ = true;
    other.index_space_size_ = 0;
    other.largest_subset_ = invalid_index_type<unsigned int>();
    other.exec_ = nullptr;

    merge();
}


template <typename IndexType>
void IndexSet<IndexType>::merge() const
{
    if (is_merged_ == true) return;
    merge_impl();
}


template <typename IndexType>
void IndexSet<IndexType>::add_dense_row(const IndexType row,
                                        const IndexType stride)
{
    auto exec = this->get_executor();
    const auto begin = row * stride;
    const auto end = (row + 1) * stride;
    GKO_ASSERT_CONDITION(
        (begin < index_space_size_) ||
        ((begin == index_space_size_) && (end == index_space_size_)));
    GKO_ASSERT_CONDITION(end <= index_space_size_);
    // Should change to something similar to AssertIndexsubset
    GKO_ASSERT_CONDITION(begin < end + 1);

    const Subset<IndexType> new_subset(exec, begin, end);

    // the new index might be larger than the last index present in the
    // subsets_. Then we can skip the binary search
    if (subsets_.size() == 0 || begin > subsets_.back().end_) {
        subsets_.push_back(new_subset);
    } else {
        subsets_.insert(
            std::lower_bound(subsets_.begin(), subsets_.end(), new_subset),
            new_subset);
    }
    is_merged_ = false;
}


template <typename IndexType>
void IndexSet<IndexType>::add_dense_rows(const IndexType begin,
                                         const IndexType end,
                                         const IndexType stride)
{
    auto exec = this->get_executor();
    GKO_ASSERT_CONDITION(
        (begin < index_space_size_) ||
        ((begin == index_space_size_) && (end == index_space_size_)));
    GKO_ASSERT_CONDITION(end <= index_space_size_);
    // Should change to something similar to AssertIndexsubset
    GKO_ASSERT_CONDITION(begin < end + 1);

    for (auto i = begin; i <= end; ++i) {
        const Subset<IndexType> new_subset(exec, i * stride, (i + 1) * stride);

        // the new index might be larger than the last index present in the
        // subsets_. Then we can skip the binary search
        if (subsets_.size() == 0 || (i * stride) > subsets_.back().end_) {
            subsets_.push_back(new_subset);
        } else {
            subsets_.insert(
                std::lower_bound(subsets_.begin(), subsets_.end(), new_subset),
                new_subset);
        }
    }
    is_merged_ = false;
}


template <typename IndexType>
void IndexSet<IndexType>::add_sparse_row(const IndexType row,
                                         const IndexType nnz_in_row)
{
    add_dense_row(row, nnz_in_row);
}


template <typename IndexType>
void IndexSet<IndexType>::add_sparse_rows(
    const IndexType begin, const IndexType end,
    const std::vector<IndexType> &nnz_per_row)
{
    GKO_ASSERT_CONDITION(
        (begin < index_space_size_) ||
        ((begin == index_space_size_) && (end == index_space_size_)));
    GKO_ASSERT_CONDITION(end <= index_space_size_);
    // Should change to something similar to AssertIndexsubset
    GKO_ASSERT_CONDITION(begin < end + 1);
    auto exec = this->get_executor();

    for (auto i = begin; i <= end; ++i) {
        auto stride = nnz_per_row[i];
        const Subset<IndexType> new_subset(exec, i * stride, (i + 1) * stride);

        // the new index might be larger than the last index present in the
        // subsets_. Then we can skip the binary search
        if (subsets_.size() == 0 || (i * stride) > subsets_.back().end_) {
            subsets_.push_back(new_subset);
        } else {
            subsets_.insert(
                std::lower_bound(subsets_.begin(), subsets_.end(), new_subset),
                new_subset);
        }
    }
    is_merged_ = false;
}


template <typename IndexType>
void IndexSet<IndexType>::add_subset(const IndexType begin, const IndexType end)
{
    GKO_ASSERT_CONDITION(
        (begin < index_space_size_) ||
        ((begin == index_space_size_) && (end == index_space_size_)));
    GKO_ASSERT_CONDITION(end <= index_space_size_);
    // Should change to something similar to AssertIndexsubset
    GKO_ASSERT_CONDITION(begin < end + 1);

    auto exec = this->get_executor();
    if (begin != end) {
        const Subset<IndexType> new_subset(exec, begin, end);

        // the new index might be larger than the last index present in the
        // subsets_. Then we can skip the binary search
        if (subsets_.size() == 0 || begin > subsets_.back().end_)
            subsets_.push_back(new_subset);
        else
            subsets_.insert(
                std::lower_bound(subsets_.begin(), subsets_.end(), new_subset),
                new_subset);
        is_merged_ = false;
    }
}


template <typename IndexType>
void IndexSet<IndexType>::add_index(const IndexType index)
{
    // Update to check AssertIndexsubset to check if index is within subset
    GKO_ASSERT_CONDITION(index < index_space_size_);

    auto exec = this->get_executor();
    const Subset<IndexType> new_subset(exec, index, index + 1);
    if (subsets_.size() == 0 || index > subsets_.back().end_)
        subsets_.push_back(new_subset);
    else if (index == subsets_.back().end_)
        subsets_.back().end_++;
    else
        subsets_.insert(
            std::lower_bound(subsets_.begin(), subsets_.end(), new_subset),
            new_subset);
    is_merged_ = false;
}


template <typename IndexType>
void IndexSet<IndexType>::add_indices(const IndexSet<IndexType> &other,
                                      const IndexType offset)
{
    if ((this == &other) && (offset == 0)) return;

    if (other.subsets_.size() != 0) {
        // AssertIndexsubset(other.subsets_.back().end_ - 1, index_space_size);
        GKO_ASSERT_CONDITION(other.subsets_.back().end_ - 1 <
                             index_space_size_);
    }

    auto exec = this->get_executor();
    merge();
    other.merge();
    typename std::vector<Subset<IndexType>>::const_iterator
        r1 = subsets_.begin(),
        r2 = other.subsets_.begin();

    std::vector<Subset<IndexType>> new_subsets;
    // just get the start and end of the subsets_ right in this method,
    // everything else will be done in merge()
    while (r1 != subsets_.end() || r2 != other.subsets_.end()) {
        // the two subsets_ do not overlap or we are at the end of one of the
        // subsets_
        if (r2 == other.subsets_.end() ||
            (r1 != subsets_.end() && r1->end_ < (r2->begin_ + offset))) {
            new_subsets.push_back(*r1);
            ++r1;
        } else if (r1 == subsets_.end() || (r2->end_ + offset) < r1->begin_) {
            new_subsets.emplace_back(exec, r2->begin_ + offset,
                                     r2->end_ + offset);
            ++r2;
        } else {
            // ok, we do overlap, so just take the combination of the current
            // subset (do not bother to merge with subsequent subsets_)
            Subset<IndexType> next(exec,
                                   std::min(r1->begin_, r2->begin_ + offset),
                                   std::max(r1->end_, r2->end_ + offset));
            new_subsets.push_back(next);
            ++r1;
            ++r2;
        }
    }
    subsets_.swap(new_subsets);

    is_merged_ = false;
    merge();
}


template <typename IndexType>
bool IndexSet<IndexType>::is_element(const IndexType index) const
{
    if (subsets_.empty() == false) {
        merge();

        // fast check whether the index is in the largest subset
        GKO_ASSERT_CONDITION(largest_subset_ < subsets_.size());
        if (index >= subsets_[largest_subset_].begin_ &&
            index < subsets_[largest_subset_].end_)
            return true;

        auto exec = this->get_executor();
        // get the element after which we would have to insert a subset that
        // consists of all elements from this element to the end of the
        // index subset plus one. after this call we know that if p!=end()
        // then p->begin<=index unless there is no such subset at all
        //
        // if the searched for element is an element of this subset, then
        // we're done. otherwise, the element can't be in one of the
        // following subsets_ because otherwise p would be a different
        // iterator
        //
        // since we already know the position relative to the largest subset
        // (we called merge!), we can perform the binary search on
        // subsets_ with lower/higher number compared to the largest subset
        typename std::vector<Subset<IndexType>>::const_iterator p =
            std::upper_bound(
                subsets_.begin() + (index < subsets_[largest_subset_].begin_
                                        ? 0
                                        : largest_subset_ + 1),
                index < subsets_[largest_subset_].begin_
                    ? subsets_.begin() + largest_subset_
                    : subsets_.end(),
                Subset<IndexType>(exec, index, get_size() + 1));

        if (p == subsets_.begin())
            return ((index >= p->begin_) && (index < p->end_));

        GKO_ASSERT_CONDITION((p == subsets_.end()) || (p->begin_ > index));

        // now move to that previous subset
        --p;
        GKO_ASSERT_CONDITION(p->begin_ <= index);

        return (p->end_ > index);
    }

    // didn't find this index, so it's not in the set
    return false;
}


template <typename IndexType>
bool IndexSet<IndexType>::is_contiguous() const
{
    merge();
    return (subsets_.size() <= 1);
}


template <typename IndexType>
IndexType IndexSet<IndexType>::get_num_elems() const
{
    // make sure we have non-overlapping subsets_
    merge();

    IndexType v = 0;
    if (!subsets_.empty()) {
        Subset<IndexType> &r = subsets_.back();
        v = r.superset_index_ + r.end_ - r.begin_;
    }

#ifdef DEBUG
    IndexType s = 0;
    for (const auto &subset : subsets_) s += (subset.end_ - subset.begin_);
    GKO_ASSERT_CONDITION(s == v);
#endif

    return v;
}


template <typename IndexType>
IndexType IndexSet<IndexType>::get_global_index(
    const IndexType local_index) const
{
    // AssertIndexsubset(n, get_num_elems());
    GKO_ASSERT_CONDITION(local_index < this->get_num_elems());

    merge();

    auto exec = this->get_executor();
    // first check whether the index is in the largest subset
    GKO_ASSERT_CONDITION(largest_subset_ < subsets_.size());
    typename std::vector<Subset<IndexType>>::const_iterator main_subset =
        subsets_.begin() + largest_subset_;
    if (local_index >= main_subset->superset_index_ &&
        local_index < main_subset->superset_index_ +
                          (main_subset->end_ - main_subset->begin_))
        return main_subset->begin_ +
               (local_index - main_subset->superset_index_);

    // find out which chunk the local index local_index belongs to by using a
    // binary search. the comparator is based on the end of the subsets_. Use
    // the position relative to main_subset to subdivide the subsets_
    Subset<IndexType> r(exec, local_index, local_index + 1);
    r.superset_index_ = local_index;
    typename std::vector<Subset<IndexType>>::const_iterator subset_begin,
        subset_end;
    if (local_index < main_subset->superset_index_) {
        subset_begin = subsets_.begin();
        subset_end = main_subset;
    } else {
        subset_begin = main_subset + 1;
        subset_end = subsets_.end();
    }

    const typename std::vector<Subset<IndexType>>::const_iterator p =
        std::lower_bound(subset_begin, subset_end, r,
                         Subset<IndexType>::superset_index_compare);

    GKO_ASSERT_CONDITION(p != subsets_.end());
    return p->begin_ + (local_index - p->superset_index_);
}


template <typename IndexType>
IndexType IndexSet<IndexType>::get_local_index(
    const IndexType global_index) const
{
    // to make this call thread-safe, merge() must not be called through
    // this function
    GKO_ASSERT_CONDITION(is_merged_ == true);
    // AssertIndexsubset(n, size());
    GKO_ASSERT_CONDITION(global_index < get_size());

    // return immediately if the index set is empty
    if (is_empty()) return invalid_index_type<IndexType>();

    auto exec = this->get_executor();
    // check whether the index is in the largest subset. use the result to
    // perform a one-sided binary search afterward
    GKO_ASSERT_CONDITION(largest_subset_ < subsets_.size());
    typename std::vector<Subset<IndexType>>::const_iterator main_subset =
        subsets_.begin() + largest_subset_;
    if (global_index >= main_subset->begin_ && global_index < main_subset->end_)
        return (global_index - main_subset->begin_) +
               main_subset->superset_index_;

    Subset<IndexType> r(exec, global_index, global_index);
    typename std::vector<Subset<IndexType>>::const_iterator subset_begin,
        subset_end;
    if (global_index < main_subset->begin_) {
        subset_begin = subsets_.begin();
        subset_end = main_subset;
    } else {
        subset_begin = main_subset + 1;
        subset_end = subsets_.end();
    }

    typename std::vector<Subset<IndexType>>::const_iterator p =
        std::lower_bound(subset_begin, subset_end, r,
                         Subset<IndexType>::compare_end);

    // if global_index is not in this set
    if (p == subset_end || p->end_ == global_index || p->begin_ > global_index)
        return invalid_index_type<IndexType>();

    GKO_ASSERT_CONDITION(p != subsets_.end());
    GKO_ASSERT_CONDITION(p->begin_ <= global_index);
    GKO_ASSERT_CONDITION(global_index < p->end_);
    return (global_index - p->begin_) + p->superset_index_;
}


template <typename IndexType>
IndexType IndexSet<IndexType>::get_num_subsets() const
{
    merge();
    return subsets_.size();
}


template <typename IndexType>
IndexType IndexSet<IndexType>::get_largest_subset_starting_index() const
{
    GKO_ASSERT_CONDITION(subsets_.empty() == false);

    merge();
    const typename std::vector<Subset<IndexType>>::const_iterator main_subset =
        subsets_.begin() + largest_subset_;

    return main_subset->superset_index_;
}


template <typename IndexType>
IndexType IndexSet<IndexType>::get_largest_element_in_set() const
{
    GKO_ASSERT_CONDITION(subsets_.empty() == false);

    merge();
    return (subsets_.back()).end_ - 1;
}


template <typename IndexType>
bool IndexSet<IndexType>::operator==(const IndexSet<IndexType> &other) const
{
    GKO_ASSERT_CONDITION(get_size() == other.get_size());
    // Are these merges expensive ? Can maybe affect performance if this index
    // set equality is called a lot.
    merge();
    other.merge();

    return subsets_ == other.subsets_;
}


template <typename IndexType>
bool IndexSet<IndexType>::operator!=(const IndexSet<IndexType> &other) const
{
    GKO_ASSERT_CONDITION(get_size() == other.get_size());

    merge();
    other.merge();

    return subsets_ != other.subsets_;
}


template <typename IndexType>
IndexSet<IndexType> IndexSet<IndexType>::operator&(
    const IndexSet<IndexType> &other) const
{
    GKO_ASSERT_CONDITION(get_size() == other.get_size());

    merge();
    other.merge();

    auto exec = this->get_executor();
    typename std::vector<Subset<IndexType>>::const_iterator
        r1 = subsets_.begin(),
        r2 = other.subsets_.begin();
    IndexSet<IndexType> result(exec, get_size());

    while ((r1 != subsets_.end()) && (r2 != other.subsets_.end())) {
        // if r1 and r2 do not overlap at all, then move the pointer that
        // sits to the left of the other up by one
        if (r1->end_ <= r2->begin_)
            ++r1;
        else if (r2->end_ <= r1->begin_)
            ++r2;
        else {
            // the subsets_ must overlap somehow
            GKO_ASSERT_CONDITION(
                ((r1->begin_ <= r2->begin_) && (r1->end_ > r2->begin_)) ||
                ((r2->begin_ <= r1->begin_) && (r2->end_ > r1->begin_)));

            // add the overlapping subset to the result
            result.add_subset(std::max(r1->begin_, r2->begin_),
                              std::min(r1->end_, r2->end_));

            // now move that iterator that ends earlier one up. note that it
            // has to be this one because a subsequent subset may still have
            // a chance of overlapping with the subset that ends later
            if (r1->end_ <= r2->end_)
                ++r1;
            else
                ++r2;
        }
    }

    result.merge();
    return result;
}


template <typename IndexType>
void IndexSet<IndexType>::subtract_set(const IndexSet<IndexType> &other)
{
    merge();
    other.merge();
    is_merged_ = false;


    auto exec = this->get_executor();
    // we save new subsets_ to be added to our IndexSet in an temporary vector
    // and add all of them in one go at the end.
    std::vector<Subset<IndexType>> new_subset;

    typename std::vector<Subset<IndexType>>::iterator own_it = subsets_.begin();
    typename std::vector<Subset<IndexType>>::iterator other_it =
        other.subsets_.begin();

    while (own_it != subsets_.end() && other_it != other.subsets_.end()) {
        // advance own iterator until we get an overlap
        if (own_it->end_ <= other_it->begin_) {
            ++own_it;
            continue;
        }
        // we are done with other_it, so advance
        if (own_it->begin_ >= other_it->end_) {
            ++other_it;
            continue;
        }

        // Now own_it and other_it overlap.  First save the part of own_it that
        // is before other_it (if not empty).
        if (own_it->begin_ < other_it->begin_) {
            Subset<IndexType> r(exec, own_it->begin_, other_it->begin_);
            r.superset_index_ = 0;  // fix warning of unused variable
            new_subset.push_back(r);
        }
        // change own_it to the sub subset behind other_it. Do not delete own_it
        // in any case. As removal would invalidate iterators, we just shrink
        // the subset to an empty one.
        own_it->begin_ = other_it->end_;
        if (own_it->begin_ > own_it->end_) {
            own_it->begin_ = own_it->end_;
            ++own_it;
        }

        // continue without advancing iterators, the right one will be advanced
        // next.
    }

    // Now delete all empty subsets_ we might
    // have created.
    for (typename std::vector<Subset<IndexType>>::iterator it =
             subsets_.begin();
         it != subsets_.end();) {
        if (it->begin_ >= it->end_)
            it = subsets_.erase(it);
        else
            ++it;
    }

    // done, now add the temporary subsets_
    const typename std::vector<Subset<IndexType>>::iterator end =
        new_subset.end();
    for (typename std::vector<Subset<IndexType>>::iterator it =
             new_subset.begin();
         it != end; ++it)
        add_subset(it->begin_, it->end_);

    merge();
}


template <typename IndexType>
IndexType IndexSet<IndexType>::pop_back()
{
    GKO_ASSERT_CONDITION(is_empty() == false);

    const IndexType index = subsets_.back().end_ - 1;
    --subsets_.back().end_;

    if (subsets_.back().begin_ == subsets_.back().end_) subsets_.pop_back();

    return index;
}


template <typename IndexType>
IndexType IndexSet<IndexType>::pop_front()
{
    GKO_ASSERT_CONDITION(is_empty() == false);

    const IndexType index = subsets_.front().begin_;
    ++subsets_.front().begin_;

    if (subsets_.front().begin_ == subsets_.front().end_)
        subsets_.erase(subsets_.begin());

    // We have to set this in any case, because superset_index_ is no longer
    // up to date for all but the first subset
    is_merged_ = false;

    return index;
}


template <typename IndexType>
typename IndexSet<IndexType>::ElementIterator IndexSet<IndexType>::begin() const
{
    merge();
    if (subsets_.size() > 0)
        return {this, 0, subsets_[0].begin_};
    else
        return end();
}


template <typename IndexType>
typename IndexSet<IndexType>::ElementIterator IndexSet<IndexType>::at(
    const IndexType global_index) const
{
    merge();
    GKO_ASSERT_CONDITION(global_index < this->get_size());

    if (subsets_.empty()) return end();

    auto exec = this->get_executor();
    typename std::vector<Subset<IndexType>>::const_iterator main_subset =
        subsets_.begin() + largest_subset_;

    Subset<IndexType> s(exec, global_index, global_index + 1);
    // This optimization makes the bounds for lower_bound smaller by
    // checking the largest subset first.
    typename std::vector<Subset<IndexType>>::const_iterator subset_begin,
        subset_end;
    if (global_index < main_subset->begin_) {
        subset_begin = subsets_.begin();
        subset_end = main_subset;
    } else {
        subset_begin = main_subset;
        subset_end = subsets_.end();
    }

    // This will give us the first subset p=[a,b[ with b>=global_index using
    // a binary search
    const typename std::vector<Subset<IndexType>>::const_iterator p =
        std::lower_bound(subset_begin, subset_end, s,
                         Subset<IndexType>::compare_end);

    // We couldn't find a subset, which means we have no subset that
    // contains global_index and also no subset behind it, meaning we need
    // to return end().
    if (p == subsets_.end()) return end();

    // Finally, we can have two cases: Either global_index is not in [a,b[,
    // which means we need to return an iterator to a because global_index,
    // ..., a-1 is not in the IndexSet<IndexType> (if branch). Alternatively,
    // global_index is in [a,b[ and we will return an iterator pointing
    // directly at global_index (else branch).
    if (global_index < p->begin_)
        return {this, static_cast<IndexType>(p - subsets_.begin()), p->begin_};
    else
        return {this, static_cast<IndexType>(p - subsets_.begin()),
                global_index};
}

template <typename IndexType>
typename IndexSet<IndexType>::ElementIterator IndexSet<IndexType>::end() const
{
    merge();
    return IndexSet<IndexType>::ElementIterator(this);
}


template <typename IndexType>
typename IndexSet<IndexType>::IntervalIterator
IndexSet<IndexType>::get_first_interval() const
{
    merge();
    if (subsets_.size() > 0)
        return IntervalIterator(this, 0);
    else
        return get_last_interval();
}


template <typename IndexType>
typename IndexSet<IndexType>::IntervalIterator
IndexSet<IndexType>::get_last_interval() const
{
    merge();
    return IntervalIterator(this);
}


template <typename IndexType>
void IndexSet<IndexType>::merge_impl() const
{
    // we will, in the following, modify mutable variables. this can only
    // work in multithreaded applications if we lock the data structures
    // via a mutex, so that users can call 'const' functions from threads
    // in parallel (and these 'const' functions can then call merge()
    // which itself calls the current function)
    // std::lock_guard<std::mutex> lock(merge_mutex_);
    std::lock_guard<std::mutex> lock(merge_mutex_);

    auto exec = this->get_executor();
    // see if any of the contiguous subsets_ can be merged. do not use
    // std::vector::erase in-place as it is quadratic in the number of
    // subsets_. since the subsets_ are sorted by their first index,
    // determining overlap isn't all that hard
    typename std::vector<Subset<IndexType>>::iterator store = subsets_.begin();
    for (typename std::vector<Subset<IndexType>>::iterator i = subsets_.begin();
         i != subsets_.end();) {
        typename std::vector<Subset<IndexType>>::iterator next = i;
        ++next;

        IndexType first_index = i->begin_;
        IndexType last_index = i->end_;

        // see if we can merge any of the following subsets_
        while (next != subsets_.end() && (next->begin_ <= last_index)) {
            last_index = std::max(last_index, next->end_);
            ++next;
        }
        i = next;

        // store the new subset in the slot we last occupied
        *store = Subset<IndexType>(exec, first_index, last_index);
        ++store;
    }
    // use a compact array with exactly the right amount of storage
    if (store != subsets_.end()) {
        std::vector<Subset<IndexType>> new_subset(subsets_.begin(), store);
        subsets_.swap(new_subset);
    }

    // now compute indices within set and the subset with most elements
    IndexType next_index = 0, largest_subset_size = 0;
    for (typename std::vector<Subset<IndexType>>::iterator i = subsets_.begin();
         i != subsets_.end(); ++i) {
        GKO_ASSERT_CONDITION(i->begin_ < i->end_);

        i->superset_index_ = next_index;
        next_index += (i->end_ - i->begin_);
        if (i->end_ - i->begin_ > largest_subset_size) {
            largest_subset_size = i->end_ - i->begin_;
            largest_subset_ = i - subsets_.begin();
        }
    }
    is_merged_ = true;

    // check that next_index is correct. needs to be after the previous
    // statement because we otherwise will get into an endless loop
    GKO_ASSERT_CONDITION(next_index == get_num_elems());
}


#define GKO_DECLARE_INDEX_SET(_type) class IndexSet<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_AND_SIZE_TYPE(GKO_DECLARE_INDEX_SET);


}  // namespace gko
