// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/multivector.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/device_views.hpp>


namespace gko {
namespace matrix::view {


template <typename ValueType>
struct vector_collection {
    size_type size;
    dim<2> element_size;
    size_type stride;
    ValueType* values;

    constexpr auto at(size_type index) const -> dense<ValueType>
    {
        return {element_size, stride,
                values + index * stride * element_size[0]};
    }

    constexpr auto operator()(size_type index, size_type row,
                              size_type col) const -> ValueType&
    {
        auto elem = at(index);
        return elem(row, col);
    }
};


}  // namespace matrix::view


class VectorCollection : public PolymorphicObject {
public:
    template <typename ValueType>
    using device_view = matrix::view::vector_collection<ValueType>;


    static auto create_with_type_of(
        ptr_param<const AbstractMultiVector> blueprint,
        std::shared_ptr<const Executor> exec, size_type collection_size,
        dim<2> size, dim<2> local_size) -> std::unique_ptr<VectorCollection>
    {
        return std::unique_ptr<VectorCollection>(
            new VectorCollection(blueprint.get(), std::move(exec),
                                 collection_size, size, local_size));
    }

    static auto create(ptr_param<const AbstractMultiVector> vectors)
        -> std::unique_ptr<VectorCollection>
    {
        std::vector<std::shared_ptr<const AbstractMultiVector>> vector_ptrs;
        for (size_type i = 0; i < vectors->get_size()[1]; ++i) {
            vector_ptrs.push_back(vectors->create_subview(
                local_span(0, vectors->get_local_size()[0]), local_span(i)));
        }
        return create(vector_ptrs.begin(), vector_ptrs.end());
    }

    template <typename VectorIterator>
    static auto create(VectorIterator begin, VectorIterator end)
        -> std::unique_ptr<VectorCollection>
    {
        return std::unique_ptr<VectorCollection>(
            new VectorCollection(begin, end));
    }

    auto at(size_type index) -> std::unique_ptr<AbstractMultiVector>
    {
        return compact_vectors_->create_subview(
            local_span(index * element_local_size_[0],
                       (index + 1) * element_local_size_[0]),
            local_span(0, element_local_size_[1]));
    }

    auto at(size_type index) const -> std::unique_ptr<const AbstractMultiVector>
    {
        return compact_vectors_->create_subview(
            local_span(index * element_local_size_[0],
                       (index + 1) * element_local_size_[0]),
            local_span(0, element_local_size_[1]));
    }

    auto get_size() const -> size_type { return size_; }

    auto get_element_size() const -> dim<2> { return element_local_size_; }

    template <typename ValueType>
    device_view<ValueType> get_device_view()
    {
        auto compact =
            compact_vectors_->template get_local_device_view<ValueType>();
        return {size_, compact.size, compact.stride, compact.values};
    }

    template <typename ValueType>
    device_view<const ValueType> get_device_view() const
    {
        auto compact =
            compact_vectors_->template get_local_device_view<ValueType>();
        return {size_, compact.size, compact.stride, compact.values};
    }

private:
    VectorCollection(ptr_param<const AbstractMultiVector> blueprint,
                     std::shared_ptr<const Executor> exec,
                     size_type collection_size, dim<2> size, dim<2> local_size)
        : PolymorphicObject(exec),
          size_{collection_size},
          element_size_{size},
          element_local_size_{local_size}
    {
        compact_vectors_ = AbstractMultiVector::create_with_type_of(
            blueprint, exec, {size[0] * collection_size, size[1]},
            {local_size[0] * collection_size, local_size[1]});
    }

    template <typename Iterator>
    static auto safe_access(Iterator it)
    {
        if (*it == nullptr) {
            GKO_INVALID_STATE(
                "VectorCollection cannot be created from empty "
                "range");
        }
        return *it;
    }

    template <typename Iterator>
    VectorCollection(Iterator begin, Iterator end)
        : VectorCollection(
              safe_access(begin), safe_access(begin)->get_executor(),
              std::distance(begin, end), safe_access(begin)->get_size(),
              safe_access(begin)->get_local_size())
    {
        for (auto it = begin; it != end; ++it) {
            if (typeid(*it) != typeid(safe_access(begin)) ||
                (*it)->get_size() != element_size_ ||
                (*it)->get_local_size() != element_local_size_) {
                GKO_INVALID_STATE(
                    "VectorCollection cannot be created from vectors with "
                    "different types or sizes");
            }
            this->at(std::distance(begin, it))->copy_from(*it);
        }
    }

    size_type size_;
    dim<2> element_size_;
    dim<2> element_local_size_;
    std::unique_ptr<AbstractMultiVector> compact_vectors_;
};


}  // namespace gko
