/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
 * Type-erased object storing any kind of gko::Array
 */
class any_array {
public:
    template <typename ValueType>
    Array<ValueType> init(std::shared_ptr<const Executor> exec, size_type size)
    {
        auto container = std::make_unique<concrete_container<ValueType>>(
            std::move(exec), size);
        auto& array = container->array;
        data_ = std::move(container);
        return array;
    }

    bool empty() const { return data_.get() == nullptr; }

    template <typename ValueType>
    bool contains() const
    {
        return dynamic_cast<const concrete_container<ValueType>*>(data_.get());
    }

    template <typename ValueType>
    Array<ValueType>& get()
    {
        GKO_ASSERT(this->template contains<ValueType>());
        return dynamic_cast<concrete_container<ValueType>*>(data_.get())->array;
    }

    template <typename ValueType>
    const Array<ValueType>& get() const
    {
        GKO_ASSERT(this->template contains<ValueType>());
        return dynamic_cast<const concrete_container<ValueType>*>(data_.get())
            ->array;
    }

private:
    struct generic_container {
        virtual ~generic_container() = default;
    };

    template <typename ValueType>
    struct concrete_container : generic_container {
        template <typename... Args>
        concrete_container(Args&&... args) : array{std::forward<Args>(args)...}
        {}

        Array<ValueType> array;
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

    template <typename VectorType, typename CreateOperation>
    VectorType* create_or_get_vector(int vector_id, CreateOperation op,
                                     const std::type_info& expected_type,
                                     dim<2> size, size_type stride)
    {
        if (vector_id >= vectors_.size()) {
            vectors_.resize(vector_id + 1);
        }
        // does the existing object have the wrong type?
        // vector types may vary e.g. if users derive from Dense
        auto stored_vec = vectors_[vector_id].get();
        VectorType* vec{};
        if (!stored_vec || typeid(*stored_vec) != expected_type) {
            auto new_vec = op();
            vec = new_vec.get();
            vectors_[vector_id] = std::move(new_vec);
            return vec;
        }
        // does the existing object have the wrong dimensions?
        vec = dynamic_cast<VectorType*>(vectors_[vector_id].get());
        GKO_ASSERT(vec);
        if (vec->get_size() != size || vec->get_stride() != stride) {
            auto new_vec = op();
            vec = new_vec.get();
            vectors_[vector_id] = std::move(new_vec);
        }
        return vec;
    }

    const LinOp* get_vector(int vector_id) const
    {
        GKO_ASSERT(vector_id < vectors_.size());
        return vectors_[vector_id].get();
    }

    template <typename ValueType>
    Array<ValueType>& create_or_get_array(int array_id, size_type size)
    {
        if (array_id >= arrays_.size()) {
            arrays_.resize(array_id + 1);
        }
        auto& array = arrays_[array_id];
        if (array.empty()) {
            array.template init<ValueType>(this->get_executor(), size);
        }
        // array types should not change!
        GKO_ASSERT(array.template contains<ValueType>());
        auto& result = array.template get<ValueType>();
        if (result.get_num_elems() != size) {
            result.resize_and_reset(size);
        }
        return result;
    }

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    void clear()
    {
        vectors_.clear();
        arrays_.clear();
    }

private:
    std::shared_ptr<const Executor> exec_;
    std::vector<std::unique_ptr<LinOp>> vectors_;
    std::vector<any_array> arrays_;
};


}  // namespace detail
}  // namespace solver
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_SOLVER_WORKSPACE_HPP_
