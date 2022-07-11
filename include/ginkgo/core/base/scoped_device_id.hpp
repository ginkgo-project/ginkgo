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


#ifndef GKO_PUBLIC_CORE_BASE_SCOPED_DEVICE_ID_HPP_
#define GKO_PUBLIC_CORE_BASE_SCOPED_DEVICE_ID_HPP_


#include <memory>


namespace gko {


class OmpExecutor;
class CudaExecutor;
class HipExecutor;
class DpcppExecutor;


namespace detail {


class generic_scoped_device_id {
public:
    generic_scoped_device_id() = default;
    virtual ~generic_scoped_device_id() noexcept(
        false){};  // TODO: this should be a purely virtual funtion, but somehow
                   // that leads to linker errors

    generic_scoped_device_id(generic_scoped_device_id& other) = delete;

    generic_scoped_device_id& operator=(const generic_scoped_device_id& other) =
        delete;
};


class noop_scoped_device_id : public generic_scoped_device_id {
public:
    noop_scoped_device_id() = default;
    ~noop_scoped_device_id() noexcept(false) override {}
};


class cuda_scoped_device_id : public generic_scoped_device_id {
public:
    explicit cuda_scoped_device_id(int device_id);

    ~cuda_scoped_device_id() noexcept(false) override;

    cuda_scoped_device_id(cuda_scoped_device_id&& other) noexcept;

    cuda_scoped_device_id& operator=(cuda_scoped_device_id&& other) noexcept;

private:
    int original_device_id_;
    bool need_reset_;
};


class hip_scoped_device_id : public generic_scoped_device_id {
public:
    explicit hip_scoped_device_id(int device_id);

    ~hip_scoped_device_id() noexcept(false) override;

    hip_scoped_device_id(hip_scoped_device_id&& other) noexcept;

    hip_scoped_device_id& operator=(hip_scoped_device_id&& other) noexcept;

private:
    int original_device_id_;
    bool need_reset_;
};

}  // namespace detail


class scoped_device_id {
public:
    scoped_device_id(const OmpExecutor* exec, int device_id)
        : scope_(std::make_unique<detail::noop_scoped_device_id>())
    {}

    scoped_device_id(const CudaExecutor* exec, int device_id)
        : scope_(std::make_unique<detail::cuda_scoped_device_id>(device_id))
    {}

    scoped_device_id(const HipExecutor* exec, int device_id)
        : scope_(std::make_unique<detail::hip_scoped_device_id>(device_id))
    {}

    scoped_device_id(const DpcppExecutor* exec, int device_id)
        : scope_(std::make_unique<detail::noop_scoped_device_id>())
    {}

    scoped_device_id() = default;

    scoped_device_id(const scoped_device_id&) = delete;

    scoped_device_id(scoped_device_id&&) = default;

    scoped_device_id& operator=(const scoped_device_id&) = delete;

    scoped_device_id& operator=(scoped_device_id&&) = default;

    ~scoped_device_id() = default;

private:
    std::unique_ptr<detail::generic_scoped_device_id> scope_;
};


}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_SCOPED_DEVICE_ID_HPP_
