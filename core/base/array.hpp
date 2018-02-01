/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_ARRAY_H_
#define GKO_CORE_BASE_ARRAY_H_


#include <memory>
#include <utility>


#include "core/base/executor.hpp"
#include "core/base/types.hpp"


namespace gko {


/**
 * An Array is a container which encapsulates fixed-sized arrays, stored on the
 * Executor tied to the Array.
 * The array stores and transfers its data as __raw__ memory, which means that
 * the constructors of its elements are not called when constructing, copying or
 * moving the Array. Thus, the Array class is most suitable for storing POD
 * types.
 *
 * @tparam ValueType  the type of elements stored in the array
 */
template <typename ValueType>
class Array {
public:
    /**
     * The type of elements stored in the array.
     */
    using value_type = ValueType;

    /**
     * Creates an empty Array tied to the specified Executor.
     *
     * @param exec  the Executor where the array data is allocated
     */
    Array(std::shared_ptr<const Executor> exec) noexcept
        : num_elems_(0), data_(nullptr), exec_(std::move(exec))
    {}

    /**
     * Creates an array on the specified Executor.
     *
     * @param exec  the Executor where the array data is be allocated
     * @param num_elems  the amount of memory (expressed as the number of
     *                   `value_type` elements) allocated on the Executor
     */
    Array(std::shared_ptr<const Executor> exec, size_type num_elems)
        : num_elems_(num_elems), data_(nullptr), exec_(std::move(exec))
    {
        if (num_elems > 0) {
            data_ = exec_->alloc<value_type>(num_elems);
        }
    }

    /**
     * Creates an array on the specified Executor and initializes it with
     * values.
     *
     * @tparam T  type of values used to initialize the array (T has to be
     *            implicitly convertible to value_type)
     *
     * @param exec  the Executor where the array data is be allocated
     * @param init_list  list of values used to initialize the Array
     */
    template <typename T>
    Array(std::shared_ptr<const Executor> exec,
          std::initializer_list<T> init_list)
        : Array(exec)
    {
        Array tmp(exec->get_master(), init_list.size());
        int i = 0;
        for (const auto &elem : init_list) {
            tmp.data_[i++] = elem;
        }
        *this = std::move(tmp);
    }

    /**
     * Creates a copy of another array.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param other  the Array to copy from
     */
    Array(const Array &other) : Array(other.get_executor())
    {
        this->copy_from(other);
    }

    /**
     * Moves another array.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param other  the Array to move
     */
    Array(Array &&other) : Array(other.get_executor())
    {
        this->move_from(std::move(other));
    }

    /**
     * Copies data from another array.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param other  the Array to copy from
     *
     * @return this
     */
    Array &operator=(const Array &other)
    {
        this->copy_from(other);
        return *this;
    }

    /**
     * Moves data from another array.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param other  the Array to move data from
     *
     * @return this
     */
    Array &operator=(Array &&other)
    {
        this->move_from(std::move(other));
        return *this;
    }

    ~Array() noexcept { this->clear(); }

    /**
     * Deallocates all data used by the Array.
     *
     * The array is left in a valid, but empty state, so the same array can be
     * used to allocate new memory. Calls to Array::get_data() will return
     * a `nullptr`.
     */
    void clear() noexcept
    {
        if (data_ != nullptr) {
            exec_->free(data_);
            data_ = nullptr;
            num_elems_ = 0;
        }
    }

    /**
     * Resizes the array so it is able to hold the specified number of elements.
     *
     * All data stored in the array will be lost.
     *
     * @param num_elems  the amount of memory (expressed as the number of
     *                   `value_type` elements) allocated on the Executor
     */
    void resize_and_reset(size_type num_elems)
    {
        if (num_elems == num_elems_) {
            return;
        }
        this->clear();
        if (num_elems > 0) {
            num_elems_ = num_elems;
            data_ = exec_->alloc<value_type>(num_elems);
        }
    }

    /**
     * Gets the number of elements in the Array.
     */
    size_type get_num_elems() const noexcept { return num_elems_; }

    /**
     * Gets a pointer to the block of memory used to store the elements of the
     * Array.
     */
    value_type *get_data() noexcept { return data_; }

    /**
     * Gets a constant pointer to the block of memory used to store the elements
     * of the Array.
     */
    const value_type *get_const_data() const noexcept { return data_; }

    /**
     * Gets the Executor associated with the array.
     */
    std::shared_ptr<const Executor> get_executor() const noexcept
    {
        return exec_;
    }

    /**
     * Takes control of the data allocated elsewhere in the program (but in the
     * same executor).
     *
     * The behavior of the array will be as if the data was allocated by the
     * library, i.e. the array will deallocate and change the block of data
     * as needed. Thus the passed block of memory must not be freed elsewhere.
     * To regain control of the memory block, call Array::release().
     *
     * @param num_elems  size (in the number of `value_type` elements) of data
     * @param data  pointer to the allocated block of memory
     */
    void manage(size_type num_elems, value_type *data) noexcept
    {
        this->clear();
        data_ = data;
        num_elems_ = num_elems;
    }

    /**
     * Releases control of a block of memory previously gained by a call to
     * Array::manage().
     */
    void release() noexcept
    {
        data_ = nullptr;
        num_elems_ = 0;
    }

    /**
     * Changes the Executor of the Array, moving the allocated data to the new
     * Executor.
     *
     * @param exec  the Executor where the data will be moved to
     */
    void set_executor(std::shared_ptr<const Executor> exec)
    {
        if (exec == exec_) {
            // moving to the same executor, no-op
            return;
        }
        Array tmp(exec);
        tmp = *this;
        this->clear();
        exec_ = exec;
        *this = std::move(tmp);
    }

private:
    void copy_from(const Array &other)
    {
        this->resize_and_reset(other.get_num_elems());
        exec_->copy_from(other.get_executor().get(), num_elems_,
                         other.get_const_data(), data_);
    }

    void move_from(Array &&other)
    {
        using std::swap;
        if (exec_ == other.get_executor()) {
            // same device, only move the pointer
            swap(data_, other.data_);
            swap(num_elems_, other.num_elems_);
        } else {
            // different device, copy the data
            this->copy_from(other);
            other.clear();
        }
    }

    size_type num_elems_;
    value_type *data_;
    std::shared_ptr<const Executor> exec_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_ARRAY_H_
