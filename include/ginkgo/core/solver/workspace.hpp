// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_WORKSPACE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_WORKSPACE_HPP_


#include <typeinfo>

#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace solver {
namespace detail {


/**
 * Type-erased object storing any kind of gko::array
 */
class any_array {
public:
    template <typename ValueType>
    array<ValueType>& init(std::shared_ptr<const Executor> exec, size_type size)
    {
        auto container = std::make_unique<concrete_container<ValueType>>(
            std::move(exec), size);
        auto& arr = container->arr;
        data_ = std::move(container);
        return arr;
    }

    bool empty() const { return data_.get() == nullptr; }

    template <typename ValueType>
    bool contains() const
    {
        return dynamic_cast<const concrete_container<ValueType>*>(data_.get());
    }

    template <typename ValueType>
    array<ValueType>& get()
    {
        GKO_ASSERT(this->template contains<ValueType>());
        return dynamic_cast<concrete_container<ValueType>*>(data_.get())->arr;
    }

    template <typename ValueType>
    const array<ValueType>& get() const
    {
        GKO_ASSERT(this->template contains<ValueType>());
        return dynamic_cast<const concrete_container<ValueType>*>(data_.get())
            ->arr;
    }

    void clear() { data_.reset(); }

private:
    struct generic_container {
        virtual ~generic_container() = default;
    };

    template <typename ValueType>
    struct concrete_container : generic_container {
        template <typename... Args>
        concrete_container(Args&&... args) : arr{std::forward<Args>(args)...}
        {}

        array<ValueType> arr;
    };

    std::unique_ptr<generic_container> data_;
};


class workspace {
public:
    workspace(std::shared_ptr<const Executor> exec) : exec_{std::move(exec)} {}

    workspace(const workspace& other) : workspace{other.get_executor()} {}

    workspace(workspace&& other) : workspace{other.get_executor()}
    {
        other.clear();
    }

    workspace& operator=(const workspace& other) { return *this; }

    workspace& operator=(workspace&& other)
    {
        other.clear();
        return *this;
    }

    template <typename CreateOperation>
    MultiVector* create_or_get_vector(int op_id, CreateOperation create,
                                      const std::type_info& expected_type,
                                      dim<2> size)
    {
        GKO_ASSERT(op_id >= 0 && op_id < vectors_.size());
        // does the existing object have the wrong type?
        // vector types may vary e.g. if users derive from Dense
        auto stored_op = vectors_[op_id].get();
        MultiVector* op{};
        if (!stored_op || typeid(*stored_op) != expected_type) {
            auto new_op = create();
            op = new_op.get();
            vectors_[op_id] = std::move(new_op);
            return op;
        }
        // does the existing object have the wrong dimensions?
        if (stored_op->get_size() != size) {
            auto new_op = create();
            stored_op = new_op.get();
            vectors_[op_id] = std::move(new_op);
        }
        return stored_op;
    }

    const MultiVector* get_vector(int op_id) const
    {
        GKO_ASSERT(op_id >= 0 && op_id < vectors_.size());
        return vectors_[op_id].get();
    }

    template <typename ValueType>
    array<ValueType>& init_or_get_array(int array_id)
    {
        GKO_ASSERT(array_id >= 0 && array_id < arrays_.size());
        auto& array = arrays_[array_id];
        if (array.empty()) {
            auto& result =
                array.template init<ValueType>(this->get_executor(), 0);
            return result;
        }
        // array types should not change!
        GKO_ASSERT(array.template contains<ValueType>());
        return array.template get<ValueType>();
    }

    template <typename ValueType>
    array<ValueType>& create_or_get_array(int array_id, size_type size)
    {
        auto& result = init_or_get_array<ValueType>(array_id);
        if (result.get_size() != size) {
            result.resize_and_reset(size);
        }
        return result;
    }

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    void set_size(int num_operators, int num_arrays)
    {
        vectors_.resize(num_operators);
        arrays_.resize(num_arrays);
    }

    void clear()
    {
        for (auto& op : vectors_) {
            op.reset();
        }
        for (auto& array : arrays_) {
            array.clear();
        }
    }

private:
    std::shared_ptr<const Executor> exec_;
    std::vector<std::unique_ptr<MultiVector>> vectors_;
    std::vector<any_array> arrays_;
};


}  // namespace detail
}  // namespace solver
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_SOLVER_WORKSPACE_HPP_
