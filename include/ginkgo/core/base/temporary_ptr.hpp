// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <functional>
#include <memory>

namespace gko {


template <typename T>
class TemporaryPtr {
    std::unique_ptr<T, std::function<void(T*)>> ptr_;
    T* orig_ptr_;

public:
    TemporaryPtr(std::unique_ptr<T, std::function<void(T*)>> ptr, T* orig_ptr)
        : ptr_(std::move(ptr)), orig_ptr_(orig_ptr)
    {}

    void copy_back()
    {
        if (orig_ptr_ != ptr_.get()) {
            orig_ptr_->copy_from(ptr_.get());
        }
    }

    T* operator->() const { return ptr_.get(); }

    T* get() const { return ptr_.get(); }

    T& operator*() const { return *ptr_; }
};


}  // namespace gko
