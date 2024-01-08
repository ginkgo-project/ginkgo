// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_ARRAY_HPP_
#define GKO_PUBLIC_CORE_BASE_ARRAY_HPP_


#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


template <typename ValueType>
class array;


namespace detail {


/**
 * @internal
 *
 * Converts `size` elements of type `SourceType` stored at `src` on `exec`
 * to `TargetType` stored at `dst`.
 */
template <typename SourceType, typename TargetType>
void convert_data(std::shared_ptr<const Executor> exec, size_type size,
                  const SourceType* src, TargetType* dst);


/**
 * @internal
 *
 * array-like non-owning wrapper for const data, to be used in conjunction with
 * `array_const_cast` and `create_const` to create matrix type wrappers from
 * constant data.
 *
 * @tparam ValueType  the type of elements stored in the array view.
 */
template <typename ValueType>
class const_array_view {
public:
    /**
     * The type of elements stored in the array view.
     */
    using value_type = ValueType;

    /**
     * Constructs an array view from existing data.
     *
     * @param exec  the executor in whose memory space the data resides.
     * @param size  the number of elements in this array view.
     * @param data  a pointer to the first element of this array view.
     */
    const_array_view(std::shared_ptr<const Executor> exec, size_type size,
                     const ValueType* data)
        : exec_{std::move(exec)}, size_{size}, data_{data}
    {}

    /*
     * To avoid any collisions with the value semantics of normal arrays,
     * disable assignment and copy-construction altogether.
     */
    const_array_view& operator=(const const_array_view&) = delete;
    const_array_view& operator=(const_array_view&&) = delete;
    const_array_view(const const_array_view&) = delete;
    /*
     * TODO C++17: delete this overload as well, it is no longer necessary due
     * to guaranteed RVO.
     */
    const_array_view(const_array_view&& other)
        : const_array_view{other.exec_, other.size_, other.data_}
    {
        other.size_ = 0;
        other.data_ = nullptr;
    }

    /**
     * Returns the number of elements in the array view.
     *
     * @return the number of elements in the array view
     */
    size_type get_size() const noexcept { return size_; }

    /**
     * Returns the number of elements in the array view.
     *
     * @return the number of elements in the array view
     */
    GKO_DEPRECATED("use get_size() instead")
    size_type get_num_elems() const noexcept { return get_size(); }

    /**
     * Returns a constant pointer to the first element of this array view.
     *
     * @return a constant pointer to the first element of this array view.
     */
    const value_type* get_const_data() const noexcept { return data_; }

    /**
     * Returns the Executor associated with the array view.
     *
     * @return the Executor associated with the array view
     */
    std::shared_ptr<const Executor> get_executor() const noexcept
    {
        return exec_;
    }

    /**
     * Returns false, to be consistent with the array interface.
     */
    bool is_owning() const noexcept { return false; }

    /**
     * Creates a array copied from const_array_view.
     *
     * @return an array constructed from const_array_view data
     */
    array<ValueType> copy_to_array() const;

private:
    std::shared_ptr<const Executor> exec_;
    size_type size_;
    const ValueType* data_;
};


template <typename ValueType>
using ConstArrayView GKO_DEPRECATED("please use const_array_view") =
    const_array_view<ValueType>;


template <typename ValueType>
array<ValueType> array_const_cast(const_array_view<ValueType> view);


}  // namespace detail


/**
 * An array is a container which encapsulates fixed-sized arrays, stored on the
 * Executor tied to the array.
 * The array stores and transfers its data as __raw__ memory, which means that
 * the constructors of its elements are not called when constructing, copying or
 * moving the array. Thus, the array class is most suitable for storing POD
 * types.
 *
 * @tparam ValueType  the type of elements stored in the array
 *
 * @ingroup array
 */
template <typename ValueType>
class array {
public:
    /**
     * The type of elements stored in the array.
     */
    using value_type = ValueType;

    /**
     * The default deleter type used by array.
     */
    using default_deleter = executor_deleter<value_type[]>;

    /**
     * The deleter type used for views.
     */
    using view_deleter = null_deleter<value_type[]>;

    /**
     * Creates an empty array not tied to any executor.
     *
     * An array without an assigned executor can only be empty. Attempts to
     * change its size (e.g. via the resize_and_reset method) will result in an
     * exception. If such an array is used as the right hand side of an
     * assignment or move assignment expression, the data of the target array
     * will be cleared, but its executor will not be modified.
     *
     * The executor can later be set by using the set_executor method. If an
     * array with no assigned executor is assigned or moved to, it will inherit
     * the executor of the source array.
     */
    array() noexcept
        : size_(0), data_(nullptr, default_deleter{nullptr}), exec_(nullptr)
    {}

    /**
     * Creates an empty array tied to the specified Executor.
     *
     * @param exec  the Executor where the array data is allocated
     */
    explicit array(std::shared_ptr<const Executor> exec) noexcept
        : size_(0),
          data_(nullptr, default_deleter{exec}),
          exec_(std::move(exec))
    {}

    /**
     * Creates an array on the specified Executor.
     *
     * @param exec  the Executor where the array data will be allocated
     * @param size  the amount of memory (expressed as the number of
     *              `value_type` elements) allocated on the Executor
     */
    array(std::shared_ptr<const Executor> exec, size_type size)
        : size_(size),
          data_(nullptr, default_deleter{exec}),
          exec_(std::move(exec))
    {
        if (size > 0) {
            data_.reset(exec_->alloc<value_type>(size));
        }
    }

    /**
     * Creates an array from existing memory.
     *
     * The memory will be managed by the array, and deallocated using the
     * specified deleter (e.g. use std::default_delete for data allocated with
     * new).
     *
     * @tparam DeleterType  type of the deleter
     *
     * @param exec  executor where `data` is located
     * @param size  number of elements in `data`
     * @param data  chunk of memory used to create the array
     * @param deleter  the deleter used to free the memory
     *
     * @see array::view() to create an array that does not deallocate memory
     * @see array(std::shared_ptr<cont Executor>, size_type, value_type*) to
     *      deallocate the memory using Executor::free() method
     */
    template <typename DeleterType>
    array(std::shared_ptr<const Executor> exec, size_type size,
          value_type* data, DeleterType deleter)
        : size_{size}, data_(data, deleter), exec_{exec}
    {}

    /**
     * Creates an array from existing memory.
     *
     * The memory will be managed by the array, and deallocated using the
     * Executor::free method.
     *
     * @param exec  executor where `data` is located
     * @param size  number of elements in `data`
     * @param data  chunk of memory used to create the array
     */
    array(std::shared_ptr<const Executor> exec, size_type size,
          value_type* data)
        : array(exec, size, data, default_deleter{exec})
    {}

    /**
     * Creates an array on the specified Executor and initializes it with
     * values.
     *
     * @tparam RandomAccessIterator  type of the iterators
     *
     * @param exec  the Executor where the array data will be allocated
     * @param begin  start of range of values
     * @param end  end of range of values
     */
    template <typename RandomAccessIterator>
    array(std::shared_ptr<const Executor> exec, RandomAccessIterator begin,
          RandomAccessIterator end)
        : array(exec)
    {
        array tmp(exec->get_master(), std::distance(begin, end));
        std::copy(begin, end, tmp.data_.get());
        *this = std::move(tmp);
    }

    /**
     * Creates an array on the specified Executor and initializes it with
     * values.
     *
     * @tparam T  type of values used to initialize the array (T has to be
     *            implicitly convertible to value_type)
     *
     * @param exec  the Executor where the array data will be allocated
     * @param init_list  list of values used to initialize the array
     */
    template <typename T>
    array(std::shared_ptr<const Executor> exec,
          std::initializer_list<T> init_list)
        : array(exec, begin(init_list), end(init_list))
    {}

    /**
     * Creates a copy of another array on a different executor.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param exec  the executor where the new array will be created
     * @param other  the array to copy from
     */
    array(std::shared_ptr<const Executor> exec, const array& other)
        : array(exec)
    {
        *this = other;
    }

    /**
     * Creates a copy of another array.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param other  the array to copy from
     */
    array(const array& other) : array(other.get_executor(), other) {}

    /**
     * Moves another array to a different executor.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param exec  the executor where the new array will be moved
     * @param other  the array to move
     */
    array(std::shared_ptr<const Executor> exec, array&& other) : array(exec)
    {
        *this = std::move(other);
    }

    /**
     * Moves another array.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * @param other  the array to move
     */
    array(array&& other) : array(other.get_executor(), std::move(other)) {}

    /**
     * Creates an array from existing memory.
     *
     * The array does not take ownership of the memory, and will not deallocate
     * it once it goes out of scope. This array type cannot use the function
     * `resize_and_reset` since it does not own the data it should resize.
     *
     * @param exec  executor where `data` is located
     * @param size  number of elements in `data`
     * @param data  chunk of memory used to create the array
     *
     * @return an array constructed from `data`
     */
    static array view(std::shared_ptr<const Executor> exec, size_type size,
                      value_type* data)
    {
        return array{exec, size, data, view_deleter{}};
    }

    /**
     * Creates a constant (immutable) array from existing memory.
     *
     * The array does not take ownership of the memory, and will not deallocate
     * it once it goes out of scope. This array type cannot use the function
     * `resize_and_reset` since it does not own the data it should resize.
     *
     * @param exec  executor where `data` is located
     * @param size  number of elements in `data`
     * @param data  chunk of memory used to create the array
     *
     * @return an array constructed from `data`
     */
    static detail::const_array_view<ValueType> const_view(
        std::shared_ptr<const Executor> exec, size_type size,
        const value_type* data)
    {
        return {exec, size, data};
    }

    /**
     * Returns a non-owning view of the memory owned by this array.
     * It can only be used until this array gets deleted, cleared or resized.
     */
    array<ValueType> as_view()
    {
        return view(this->get_executor(), this->get_size(), this->get_data());
    }

    /**
     * Returns a non-owning constant view of the memory owned by this array.
     * It can only be used until this array gets deleted, cleared or resized.
     */
    detail::const_array_view<ValueType> as_const_view() const
    {
        return const_view(this->get_executor(), this->get_size(),
                          this->get_const_data());
    }

    /**
     * Copies data from another array or view. In the case of an array target,
     * the array is resized to match the source's size. In the case of a view
     * target, if the dimensions are not compatible a gko::OutOfBoundsError is
     * thrown.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * The executor of this is preserved. In case this does not have an assigned
     * executor, it will inherit the executor of other.
     *
     * @param other  the array to copy from
     *
     * @return this
     */
    array& operator=(const array& other)
    {
        if (&other == this) {
            return *this;
        }
        if (exec_ == nullptr) {
            exec_ = other.get_executor();
            data_ = data_manager{nullptr, other.data_.get_deleter()};
        }
        if (other.get_executor() == nullptr) {
            this->clear();
            return *this;
        }

        if (this->is_owning()) {
            this->resize_and_reset(other.get_size());
        } else {
            GKO_ENSURE_COMPATIBLE_BOUNDS(other.get_size(), this->get_size());
        }
        exec_->copy_from(other.get_executor(), other.get_size(),
                         other.get_const_data(), this->get_data());
        return *this;
    }

    /**
     * Moves data from another array or view. Only the pointer and deleter type
     * change, a copy only happens when targeting another executor's data. This
     * means that in the following situation:
     * ```cpp
     *   gko::array<int> a; // an existing array or view
     *   gko::array<int> b; // an existing array or view
     *   b = std::move(a);
     * ```
     * Depending on whether `a` and `b` are array or view, this happens:
     * + `a` and `b` are views, `b` becomes the only valid view of `a`;
     * + `a` and `b` are arrays, `b` becomes the only valid array of `a`;
     * + `a` is a view and `b` is an array, `b` frees its data and becomes the
     *    only valid view of `a` ();
     * + `a` is an array and `b` is a view, `b` becomes the only valid array
     *    of `a`.
     *
     * In all the previous cases, `a` becomes empty (e.g., a `nullptr`).
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * The executor of this is preserved. In case this does not have an assigned
     * executor, it will inherit the executor of other.
     *
     * @param other  the array to move data from
     *
     * @return this
     */
    array& operator=(array&& other)
    {
        if (&other == this) {
            return *this;
        }
        if (exec_ == nullptr) {
            exec_ = other.get_executor();
            data_ = data_manager{nullptr, default_deleter{exec_}};
        }
        if (other.get_executor() == nullptr) {
            this->clear();
            return *this;
        }
        if (exec_ == other.get_executor()) {
            // same device, only move the pointer
            data_ = std::exchange(
                other.data_, data_manager{nullptr, default_deleter{exec_}});
            size_ = std::exchange(other.size_, 0);
        } else {
            // different device, copy the data
            *this = other;
            other.clear();
        }
        return *this;
    }

    /**
     * Copies and converts data from another array with another data type.
     * In the case of an array target, the array is resized to match the
     * source's size. In the case of a view target, if the dimensions are not
     * compatible a gko::OutOfBoundsError is thrown.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * The executor of this is preserved. In case this does not have an assigned
     * executor, it will inherit the executor of other.
     *
     * @param other  the array to copy from
     * @tparam OtherValueType  the value type of `other`
     *
     * @return this
     */
    template <typename OtherValueType>
    std::enable_if_t<!std::is_same<ValueType, OtherValueType>::value, array>&
    operator=(const array<OtherValueType>& other)
    {
        if (this->exec_ == nullptr) {
            this->exec_ = other.get_executor();
            this->data_ = data_manager{nullptr, default_deleter{this->exec_}};
        }
        if (other.get_executor() == nullptr) {
            this->clear();
            return *this;
        }

        if (this->is_owning()) {
            this->resize_and_reset(other.get_size());
        } else {
            GKO_ENSURE_COMPATIBLE_BOUNDS(other.get_size(), this->get_size());
        }
        array<OtherValueType> tmp{this->exec_};
        const OtherValueType* source = other.get_const_data();
        // if we are on different executors: copy, then convert
        if (this->exec_ != other.get_executor()) {
            tmp = other;
            source = tmp.get_const_data();
        }
        detail::convert_data(this->exec_, other.get_size(), source,
                             this->get_data());
        return *this;
    }

    /**
     * Copies data from a const_array_view.
     *
     * In the case of an array target, the array is resized to match the
     * source's size. In the case of a view target, if the dimensions are not
     * compatible a gko::OutOfBoundsError is thrown.
     *
     * This does not invoke the constructors of the elements, instead they are
     * copied as POD types.
     *
     * The executor of this is preserved. In case this does not have an assigned
     * executor, it will inherit the executor of other.
     *
     * @param other  the const_array_view to copy from
     *
     * @return this
     */
    array& operator=(const detail::const_array_view<ValueType>& other)
    {
        if (this->exec_ == nullptr) {
            this->exec_ = other.get_executor();
            this->data_ = data_manager{nullptr, default_deleter{this->exec_}};
        }
        if (other.get_executor() == nullptr) {
            this->clear();
            return *this;
        }

        if (this->is_owning()) {
            this->resize_and_reset(other.get_size());
        } else {
            GKO_ENSURE_COMPATIBLE_BOUNDS(other.get_size(), this->get_size());
        }
        array tmp{this->exec_};
        const ValueType* source = other.get_const_data();
        // if we are on different executors: copy
        if (this->exec_ != other.get_executor()) {
            tmp = other.copy_to_array();
            source = tmp.get_const_data();
        }
        exec_->copy_from(other.get_executor(), other.get_size(), source,
                         this->get_data());
        return *this;
    }

    /**
     * Deallocates all data used by the array.
     *
     * The array is left in a valid, but empty state, so the same array can be
     * used to allocate new memory. Calls to array::get_data() will return
     * a `nullptr`.
     */
    void clear() noexcept
    {
        size_ = 0;
        data_.reset(nullptr);
    }

    /**
     * Resizes the array so it is able to hold the specified number of elements.
     * For a view and other non-owning array types, this throws an exception
     * since these types cannot be resized.
     *
     * All data stored in the array will be lost.
     *
     * If the array is not assigned an executor, an exception will be thrown.
     *
     * @param size  the amount of memory (expressed as the number of
     *              `value_type` elements) allocated on the Executor
     */
    void resize_and_reset(size_type size)
    {
        if (size == size_) {
            return;
        }
        if (exec_ == nullptr) {
            throw gko::NotSupported(__FILE__, __LINE__, __func__,
                                    "gko::Executor (nullptr)");
        }
        if (!this->is_owning()) {
            throw gko::NotSupported(__FILE__, __LINE__, __func__,
                                    "Non owning gko::array cannot be resized.");
        }

        if (size > 0 && this->is_owning()) {
            size_ = size;
            data_.reset(exec_->alloc<value_type>(size));
        } else {
            this->clear();
        }
    }

    /**
     * Fill the array with the given value.
     *
     * @param value the value to be filled
     */
    void fill(const value_type value);

    /**
     * Returns the number of elements in the array.
     *
     * @return the number of elements in the array
     */
    size_type get_size() const noexcept { return size_; }

    /**
     * Returns the number of elements in the array.
     *
     * @return the number of elements in the array
     */
    GKO_DEPRECATED("use get_size() instead")
    size_type get_num_elems() const noexcept { return get_size(); }

    /**
     * Returns a pointer to the block of memory used to store the elements of
     * the array.
     *
     * @return a pointer to the block of memory used to store the elements of
     * the array
     */
    value_type* get_data() noexcept { return data_.get(); }

    /**
     * Returns a constant pointer to the block of memory used to store the
     * elements of the array.
     *
     * @return a constant pointer to the block of memory used to store the
     * elements of the array
     */
    const value_type* get_const_data() const noexcept { return data_.get(); }

    /**
     * Returns the Executor associated with the array.
     *
     * @return the Executor associated with the array
     */
    std::shared_ptr<const Executor> get_executor() const noexcept
    {
        return exec_;
    }

    /**
     * Changes the Executor of the array, moving the allocated data to the new
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
        array tmp(std::move(exec));
        tmp = *this;
        exec_ = std::move(tmp.exec_);
        data_ = std::move(tmp.data_);
    }

    /**
     * Tells whether this array owns its data or not.
     *
     * Views do not own their data and this has multiple implications. They
     * cannot be resized since the data is not owned by the array which stores a
     * view. It is also unclear whether custom deleter types are owning types as
     * they could be a user-created view-type, therefore only proper array which
     * use the `default_deleter` are considered owning types.
     *
     * @return whether this array can be resized or not.
     */
    bool is_owning()
    {
        return data_.get_deleter().target_type() == typeid(default_deleter);
    }


private:
    // Allow other array types to access private members
    template <typename OtherValueType>
    friend class array;

    using data_manager =
        std::unique_ptr<value_type[], std::function<void(value_type[])>>;

    size_type size_;
    data_manager data_;
    std::shared_ptr<const Executor> exec_;
};


template <typename ValueType>
using Array GKO_DEPRECATED("please use array") = array<ValueType>;


/**
 * Reduce (sum) the values in the array
 *
 * @tparam The type of the input data
 *
 * @param [in] input_arr the input array to be reduced
 * @param [in] init_val the initial value
 *
 * @return the reduced value
 */
template <typename ValueType>
ValueType reduce_add(const array<ValueType>& input_arr,
                     const ValueType init_val = 0);

/**
 * Reduce (sum) the values in the array
 *
 * @tparam The type of the input data
 *
 * @param [in] input_arr the input array to be reduced
 * @param [out,in] result the reduced value. The result is written into the
 *                 first entry and the value in the first entry is used as the
 *                 initial value for the reduce.
 */
template <typename ValueType>
void reduce_add(const array<ValueType>& input_arr, array<ValueType>& result);


/**
 * Helper function to create an array view deducing the value type.
 *
 * @param exec  the executor on which the array resides
 * @param size  the number of elements for the array
 * @param data  the pointer to the array we create a view on.
 *
 * @tparam ValueType  the type of the array elements
 *
 * @return `array<ValueType>::view(exec, size, data)`
 */
template <typename ValueType>
array<ValueType> make_array_view(std::shared_ptr<const Executor> exec,
                                 size_type size, ValueType* data)
{
    return array<ValueType>::view(exec, size, data);
}


/**
 * Helper function to create a const array view deducing the value type.
 *
 * @param exec  the executor on which the array resides
 * @param size  the number of elements for the array
 * @param data  the pointer to the array we create a view on.
 *
 * @tparam ValueType  the type of the array elements
 *
 * @return `array<ValueType>::const_view(exec, size, data)`
 */
template <typename ValueType>
detail::const_array_view<ValueType> make_const_array_view(
    std::shared_ptr<const Executor> exec, size_type size, const ValueType* data)
{
    return array<ValueType>::const_view(exec, size, data);
}


namespace detail {


template <typename T>
struct temporary_clone_helper<array<T>> {
    static std::unique_ptr<array<T>> create(
        std::shared_ptr<const Executor> exec, array<T>* ptr, bool copy_data)
    {
        if (copy_data) {
            return std::make_unique<array<T>>(std::move(exec), *ptr);
        } else {
            return std::make_unique<array<T>>(std::move(exec), ptr->get_size());
        }
    }
};

template <typename T>
struct temporary_clone_helper<const array<T>> {
    static std::unique_ptr<const array<T>> create(
        std::shared_ptr<const Executor> exec, const array<T>* ptr, bool)
    {
        return std::make_unique<const array<T>>(std::move(exec), *ptr);
    }
};


// specialization for non-constant arrays, copying back via assignment
template <typename T>
class copy_back_deleter<array<T>> {
public:
    using pointer = array<T>*;

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


/**
 * @internal
 *
 * Casts away const-ness from a array view to get a array representing the same
 * data. This needs to be used carefully, as the class this array gets passed to
 * must not modify its data. That is usually achieved by creating a `const`
 * instance of the class.
 *
 * @param view  the array view to be cast.
 * @returns a non-const non-owning array wrapping the same pointer on the same
 *          executor as `view`.
 */
template <typename ValueType>
array<ValueType> array_const_cast(const_array_view<ValueType> view)
{
    return array<ValueType>::view(
        view.get_executor(), view.get_size(),
        const_cast<ValueType*>(view.get_const_data()));
}


template <typename ValueType>
array<ValueType> const_array_view<ValueType>::copy_to_array() const
{
    array<ValueType> result(this->get_executor(), this->get_size());
    result.get_executor()->copy_from(this->get_executor(), this->get_size(),
                                     this->get_const_data(), result.get_data());
    return result;
}


}  // namespace detail
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_ARRAY_HPP_
