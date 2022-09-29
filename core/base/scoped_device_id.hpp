#ifndef GINKGO_CORE_BASE_SCOPED_DEVICE_ID_HPP
#define GINKGO_CORE_BASE_SCOPED_DEVICE_ID_HPP


#include <ginkgo/core/base/scoped_device_id.hpp>


namespace gko {
namespace detail {


/**
 * An implementation of generic_scoped_device_id that does nothing.
 *
 * This is used for OmpExecutor and DpcppExecutor, since they don't require
 * setting a device id.
 */
class noop_scoped_device_id : public generic_scoped_device_id {};


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
}  // namespace gko


#endif  // GINKGO_CORE_BASE_SCOPED_DEVICE_ID_HPP
