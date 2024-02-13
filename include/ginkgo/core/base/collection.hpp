// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <numeric>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>


namespace gko {
/**
 * \brief Same as span, but templated
 * \tparam IntegerType an integer type to store the span
 */
template <typename IntegerType>
struct typed_span {
    GKO_ATTRIBUTES constexpr typed_span(IntegerType point) noexcept
        : typed_span{point, point + 1}
    {}

    GKO_ATTRIBUTES constexpr typed_span(IntegerType begin,
                                        IntegerType end) noexcept
        : begin(begin), end(end)
    {}

    GKO_ATTRIBUTES constexpr bool is_valid() const { return begin <= end; }

    GKO_ATTRIBUTES constexpr IntegerType length() const { return end - begin; }

    IntegerType begin;

    IntegerType end;
};


namespace collection {


template <typename T>
using span = std::vector<typed_span<T>>;


template <typename IndexType>
IndexType get_min(const span<IndexType>& spans)
{
    if (spans.empty()) {
        return std::numeric_limits<IndexType>::max();
    }
    return std::min_element(
               spans.begin(), spans.end(),
               [](const auto& a, const auto& b) { return a.begin < b.begin; })
        ->begin;
}


template <typename IndexType>
IndexType get_max(const span<IndexType>& spans)
{
    if (spans.empty()) {
        return std::numeric_limits<IndexType>::min();
    }
    return std::max_element(
               spans.begin(), spans.end(),
               [](const auto& a, const auto& b) { return a.end < b.end; })
        ->end;
}


template <typename IndexType>
size_type get_size(const span<IndexType>& spans)
{
    if (spans.empty()) {
        return 0;
    }
    return get_max(spans) - get_min(spans);
}


template <typename T>
struct array {
    using value_type = T;
    using iterator = gko::array<T>*;
    using const_iterator = const gko::array<T>*;
    using reference = gko::array<T>&;
    using const_reference = const gko::array<T>&;

    array() = default;

    array(std::shared_ptr<const Executor> exec) : buffer_(exec) {}

    // needs special care to make sure elems_ stay views
    array(const array& other) { *this = other; }

    array& operator=(const array& other)
    {
        if (this != &other) {
            buffer_ = other.buffer_;
            offsets_ = other.offsets_;
            elems_.resize(other.elems_.size());
            for (size_type i = 0; i < elems_.size(); ++i) {
                elems_[i] = make_array_view(
                    buffer_.get_executor(), other[i].get_size(),
                    buffer_.get_data() +
                        std::distance(other.buffer_.get_const_data(),
                                      other[i].get_const_data()));
            }
        }
        return *this;
    }

    array(array&&) noexcept = default;
    array& operator=(array&&) noexcept = default;

    template <typename SizeType = std::size_t>
    array(std::shared_ptr<const Executor> exec,
          const std::vector<SizeType>& sizes)
        : array(gko::array<T>(exec,
                              std::accumulate(sizes.begin(), sizes.end(), 0)),
                sizes)
    {}

    template <typename SizeType = std::size_t>
    array(gko::array<T> buffer, const std::vector<SizeType>& sizes)
        : buffer_(std::move(buffer))
    {
        auto exec = buffer_.get_executor();
        offsets_ = gko::array<size_type>(exec->get_master(), sizes.size() + 1);
        offsets_.fill(0);
        std::partial_sum(sizes.begin(), sizes.end(), offsets_.get_data() + 1);
        buffer_.resize_and_reset(offsets_.get_data()[sizes.size()]);
        // can't use emplace/push_back because then the view will get copied
        // into owning arrays
        elems_.resize(sizes.size());
        for (size_type i = 0; i < sizes.size(); ++i) {
            elems_[i] = make_array_view(
                exec, sizes[i], buffer_.get_data() + offsets_.get_data()[i]);
        }
        offsets_.set_executor(exec);
    }

    reference operator[](size_type i) { return elems_[i]; }
    const_reference operator[](size_type i) const { return elems_[i]; }

    [[nodiscard]] size_type size() const { return elems_.size(); }

    iterator begin() { return elems_.data(); }
    const_iterator begin() const { return elems_.data(); }

    iterator end() { return begin() + elems_.size(); }
    const_iterator end() const { return begin() + elems_.size(); }

    [[nodiscard]] bool empty() const { return elems_.empty(); }

    reference get_flat() { return buffer_; }
    const_reference get_flat() const { return buffer_; }

    const gko::array<size_type>& get_offsets() const { return offsets_; }

    std::shared_ptr<const Executor> get_executor() const
    {
        return buffer_.get_executor();
    }

private:
    gko::array<T> buffer_;
    gko::array<size_type> offsets_;
    std::vector<gko::array<T>> elems_;
};


template <typename IndexType>
IndexType get_min(const array<IndexType>& arrs);


template <typename IndexType>
IndexType get_max(const array<IndexType>& arrs);


template <typename IndexType>
size_type get_size(const array<IndexType>& arrs)
{
    return arrs.get_flat().get_size();
}


}  // namespace collection


namespace detail {
template <typename T>
struct temporary_clone_helper<collection::array<T>> {
    static std::unique_ptr<collection::array<T>> create(
        std::shared_ptr<const Executor> exec, collection::array<T>* ptr,
        bool copy_data)
    {
        std::vector<size_type> sizes(ptr->size());
        std::transform(ptr->begin(), ptr->end(), sizes.begin(),
                       [](const auto& a) { return a.get_size(); });
        if (copy_data) {
            return std::make_unique<collection::array<T>>(
                array<T>{exec, ptr->get_flat()}, sizes);
        } else {
            return std::make_unique<collection::array<T>>(std::move(exec),
                                                          sizes);
        }
    }
};

template <typename T>
struct temporary_clone_helper<const collection::array<T>> {
    static std::unique_ptr<const collection::array<T>> create(
        std::shared_ptr<const Executor> exec, const collection::array<T>* ptr,
        bool)
    {
        std::vector<size_type> sizes(ptr->size());
        std::transform(ptr->begin(), ptr->end(), sizes.begin(),
                       [](const auto& a) { return a.get_size(); });
        return std::make_unique<collection::array<T>>(
            array<T>{exec, ptr->get_flat()}, sizes);
    }
};


// specialization for non-constant arrays, copying back via assignment
template <typename T>
class copy_back_deleter<collection::array<T>> {
public:
    using pointer = collection::array<T>*;

    /**
     * Creates a new deleter object.
     *
     * @param original  the origin object where the data will be copied before
     *                  deletion
     */
    copy_back_deleter(pointer original) : original_{original} {}

    /**
     * Copies back the pointed-to object to the original and deletes it.
     *
     * @param ptr  pointer to the object to be copied back and deleted
     */
    void operator()(pointer ptr) const
    {
        *original_ = *ptr;
        delete ptr;
    }

private:
    pointer original_;
};
}  // namespace detail
}  // namespace gko
