// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_SCOPED_DEVICE_ID_GUARD_HPP_
#define GKO_PUBLIC_CORE_BASE_SCOPED_DEVICE_ID_GUARD_HPP_


#include <memory>


namespace gko {


class OmpExecutor;
class ReferenceExecutor;
class CudaExecutor;
class HipExecutor;
class DpcppExecutor;


namespace detail {


/**
 * A RAII, move-only base class for the scoped device id used for different
 * executors.
 */
class generic_scoped_device_id_guard {
public:
    generic_scoped_device_id_guard() = default;

    // TODO: this should be a purely virtual function, but somehow that leads to
    // linker errors
    virtual ~generic_scoped_device_id_guard() = default;

    // Prohibit copy construction
    generic_scoped_device_id_guard(
        const generic_scoped_device_id_guard& other) = delete;

    // Prohibit copy assignment
    generic_scoped_device_id_guard& operator=(
        const generic_scoped_device_id_guard& other) = delete;
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
 * detail::generic_scoped_device_id_guard is picked. The following illustrates
 * the usage of this class:
 * ```
 * {
 *   scoped_device_id_guard g{static_cast<CudaExecutor>(nullptr), 1};
 *   // now the device id is set to 1
 * }
 * // now the device id is reverted again
 * ```
 */
class scoped_device_id_guard {
public:
    /**
     * Create a scoped device id from an Reference.
     *
     * The resulting object will be a noop.
     *
     * @param exec  Not used.
     * @param device_id  Not used.
     */
    scoped_device_id_guard(const ReferenceExecutor* exec, int device_id);

    /**
     * Create a scoped device id from an OmpExecutor.
     *
     * The resulting object will be a noop.
     *
     * @param exec  Not used.
     * @param device_id  Not used.
     */
    scoped_device_id_guard(const OmpExecutor* exec, int device_id);

    /**
     * Create a scoped device id from an CudaExecutor.
     *
     * The resulting object will set the cuda device id accordingly.
     *
     * @param exec  Not used.
     * @param device_id  The device id to use within the scope.
     */
    scoped_device_id_guard(const CudaExecutor* exec, int device_id);

    /**
     * Create a scoped device id from an HipExecutor.
     *
     * The resulting object will set the hip device id accordingly.
     *
     * @param exec  Not used.
     * @param device_id  The device id to use within the scope.
     */
    scoped_device_id_guard(const HipExecutor* exec, int device_id);

    /**
     * Create a scoped device id from an DpcppExecutor.
     *
     * The resulting object will be a noop.
     *
     * @param exec  Not used.
     * @param device_id  Not used.
     */
    scoped_device_id_guard(const DpcppExecutor* exec, int device_id);

    scoped_device_id_guard() = default;

    // Prohibit copy construction.
    scoped_device_id_guard(const scoped_device_id_guard&) = delete;

    // Allow move construction.
    // These are needed, since C++14 does not guarantee copy elision.
    scoped_device_id_guard(scoped_device_id_guard&&) = default;

    // Prohibit copy assignment.
    scoped_device_id_guard& operator=(const scoped_device_id_guard&) = delete;

    // Allow move construction.
    // These are needed, since C++14 does not guarantee copy elision.
    scoped_device_id_guard& operator=(scoped_device_id_guard&&) = default;

    ~scoped_device_id_guard() = default;

private:
    std::unique_ptr<detail::generic_scoped_device_id_guard> scope_;
};


}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_SCOPED_DEVICE_ID_GUARD_HPP_
