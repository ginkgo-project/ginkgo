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


/**
 * A RAII, move-only base class for the scoped device id used for different
 * executors.
 */
class generic_scoped_device_id {
public:
    generic_scoped_device_id() = default;

    // TODO: this should be a purely virtual funtion, but somehow that leads to
    // linker errors
    // This is explicitly not noexcept, since setting the device id may throw in
    // the derived classes for CUDA and HIP.
    virtual ~generic_scoped_device_id() noexcept(false){};

    // Prohibit copy construction
    generic_scoped_device_id(generic_scoped_device_id& other) = delete;

    // Prohibit copy assignment
    generic_scoped_device_id& operator=(const generic_scoped_device_id& other) =
        delete;
};


/**
 * An implementation of generic_scoped_device_id that does nothing.
 *
 * This is used for OmpExecutor and DpcppExecutor, since they don't require
 * setting a device id.
 */
class noop_scoped_device_id : public generic_scoped_device_id {
public:
    noop_scoped_device_id() = default;
    ~noop_scoped_device_id() noexcept(false) override {}
};


/**
 * A scoped device id for CUDA.
 */
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


/**
 * A scoped device id for HIP.
 */
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


/**
 * This move-only class uses RAII to set the device id within a scoped block, if
 * necessary.
 *
 * The class behaves similar to std::scoped_lock. The scoped guard will make
 * sure that the device code is run on the correct device within one scoped
 * block, when run with multiple devices. Depending on the executor it will
 * record the current device id and set the device id to the one being passed
 * in. After the scope has been exited, the destructor sets the device_id back
 * to the one before entering the scope. The OmpExecutor and DpcppExecutor don't
 * require setting an device id, so in those cases, the class is a no-op.
 *
 * The device id scope has to be constructed from a executor with concrete type
 * (not plain Executor) and a device id. Only the type of the executor object is
 * relevant, so the pointer will not be accessed, and may even be a nullptr.
 * From the executor type the correct derived class of
 * detail::generic_scoped_device_id is picked. The following illustrates the
 * usage of this class:
 * ```
 * {
 *   scoped_device_id g{static_cast<CudaExecutor>(nullptr), 1};
 *   // now the device id is set to 1
 * }
 * // now the device id is reverted again
 * ```
 */
class scoped_device_id {
public:
    /**
     * Create a scoped device id from an OmpExecutor.
     *
     * This will pick the noop_scoped_device_id.
     *
     * @param exec  Not used.
     * @param device_id  Not used.
     */
    scoped_device_id(const OmpExecutor* exec, int device_id)
        : scope_(std::make_unique<detail::noop_scoped_device_id>())
    {}

    /**
     * Create a scoped device id from an CudaExecutor.
     *
     * This will pick the cuda_scoped_device_id.
     *
     * @param exec  Not used.
     * @param device_id  The device id to use within the scope.
     */
    scoped_device_id(const CudaExecutor* exec, int device_id)
        : scope_(std::make_unique<detail::cuda_scoped_device_id>(device_id))
    {}

    /**
     * Create a scoped device id from an HipExecutor.
     *
     * This will pick the hip_scoped_device_id.
     *
     * @param exec  Not used.
     * @param device_id  The device id to use within the scope.
     */
    scoped_device_id(const HipExecutor* exec, int device_id)
        : scope_(std::make_unique<detail::hip_scoped_device_id>(device_id))
    {}

    /**
     * Create a scoped device id from an OmpExecutor.
     *
     * This will pick the noop_scoped_device_id.
     *
     * @param exec  Not used.
     * @param device_id  Not used.
     */
    scoped_device_id(const DpcppExecutor* exec, int device_id)
        : scope_(std::make_unique<detail::noop_scoped_device_id>())
    {}

    scoped_device_id() = default;

    // Prohibit copy construction.
    scoped_device_id(const scoped_device_id&) = delete;

    // Allow move construction.
    scoped_device_id(scoped_device_id&&) = default;

    // Prohibit copy assignment.
    scoped_device_id& operator=(const scoped_device_id&) = delete;

    // Allow move construction.
    scoped_device_id& operator=(scoped_device_id&&) = default;

    ~scoped_device_id() = default;

private:
    std::unique_ptr<detail::generic_scoped_device_id> scope_;
};


}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_SCOPED_DEVICE_ID_HPP_
