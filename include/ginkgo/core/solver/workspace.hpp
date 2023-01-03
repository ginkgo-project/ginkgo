/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

    template <typename LinOpType, typename CreateOperation>
    LinOpType* create_or_get_op(int op_id, CreateOperation create,
                                const std::type_info& expected_type,
                                dim<2> size, size_type stride)
    {
        GKO_ASSERT(op_id >= 0 && op_id < operators_.size());
        // does the existing object have the wrong type?
        // vector types may vary e.g. if users derive from Dense
        auto stored_op = operators_[op_id].get();
        LinOpType* op{};
        if (!stored_op || typeid(*stored_op) != expected_type) {
            auto new_op = create();
            op = new_op.get();
            operators_[op_id] = std::move(new_op);
            return op;
        }
        // does the existing object have the wrong dimensions?
        op = dynamic_cast<LinOpType*>(operators_[op_id].get());
        GKO_ASSERT(op);
        if (op->get_size() != size || op->get_stride() != stride) {
            auto new_op = create();
            op = new_op.get();
            operators_[op_id] = std::move(new_op);
        }
        return op;
    }

    const LinOp* get_op(int op_id) const
    {
        GKO_ASSERT(op_id >= 0 && op_id < operators_.size());
        return operators_[op_id].get();
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
        if (result.get_num_elems() != size) {
            result.resize_and_reset(size);
        }
        return result;
    }

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    void set_size(int num_operators, int num_arrays)
    {
        operators_.resize(num_operators);
        arrays_.resize(num_arrays);
    }

    void clear()
    {
        for (auto& op : operators_) {
            op.reset();
        }
        for (auto& array : arrays_) {
            array.clear();
        }
    }

private:
    std::shared_ptr<const Executor> exec_;
    std::vector<std::unique_ptr<LinOp>> operators_;
    std::vector<any_array> arrays_;
};


}  // namespace detail
}  // namespace solver
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_SOLVER_WORKSPACE_HPP_
