// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_EXECUTOR_HPP_
#define GKO_PUBLIC_CORE_BASE_EXECUTOR_HPP_


#include <array>
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>


#include <ginkgo/core/base/device.hpp>
#include <ginkgo/core/base/fwd_decls.hpp>
#include <ginkgo/core/base/machine_topology.hpp>
#include <ginkgo/core/base/memory.hpp>
#include <ginkgo/core/base/scoped_device_id_guard.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


namespace gko {


/** How Logger events are propagated to their Executor. */
enum class log_propagation_mode {
    /**
     * Events only get reported at loggers attached to the triggering object.
     * (Except for allocation/free, copy and Operations, since they happen at
     * the Executor).
     */
    never,
    /**
     * Events get reported to loggers attached to the triggering object and
     * propagating loggers (Logger::needs_propagation() return true) attached to
     * its Executor.
     */
    automatic
};


/**
 * Specify the mode of allocation for CUDA/HIP GPUs.
 *
 * `device` allocates memory on the device and Unified Memory model is not used.
 *
 * `unified_global` allocates memory on the device, but is accessible by the
 * host through the Unified memory model.
 *
 * `unified_host` allocates memory on the
 * host and it is not available on devices which do not have concurrent accesses
 * switched on, but this access can be explicitly switched on, when necessary.
 */
enum class allocation_mode { device, unified_global, unified_host };


#ifdef NDEBUG

// When in release, prefer device allocations
constexpr allocation_mode default_cuda_alloc_mode = allocation_mode::device;

constexpr allocation_mode default_hip_alloc_mode = allocation_mode::device;

#else

// When in debug, always UM allocations.
constexpr allocation_mode default_cuda_alloc_mode =
    allocation_mode::unified_global;

#if (GINKGO_HIP_PLATFORM_HCC == 1)

// HIP on AMD GPUs does not support UM, so always prefer device allocations.
constexpr allocation_mode default_hip_alloc_mode = allocation_mode::device;

#else

// HIP on NVIDIA GPUs supports UM, so prefer UM allocations.
constexpr allocation_mode default_hip_alloc_mode =
    allocation_mode::unified_global;

#endif

#endif


}  // namespace gko


/**
 * The enum class is for the dpcpp queue property. It's legal to use a binary
 * or(|) operation to combine several properties.
 */
enum class dpcpp_queue_property {
    /**
     * queue executes in order
     */
    in_order = 1,

    /**
     * queue enables the profiling
     */
    enable_profiling = 2
};

GKO_ATTRIBUTES GKO_INLINE dpcpp_queue_property operator|(dpcpp_queue_property a,
                                                         dpcpp_queue_property b)
{
    return static_cast<dpcpp_queue_property>(static_cast<int>(a) |
                                             static_cast<int>(b));
}


namespace gko {


#define GKO_FORWARD_DECLARE(_type, ...) class _type

GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_FORWARD_DECLARE);

#undef GKO_FORWARD_DECLARE


class ReferenceExecutor;


namespace detail {


template <typename>
class ExecutorBase;


}  // namespace detail


/**
 * Operations can be used to define functionalities whose implementations differ
 * among devices.
 *
 * This is done by extending the Operation class and implementing the overloads
 * of the Operation::run() method for all Executor types. When invoking the
 * Executor::run() method with the Operation as input, the library will select
 * the Operation::run() overload corresponding to the dynamic type of the
 * Executor instance.
 *
 * Consider an overload of `operator<<` for Executors, which prints some basic
 * device information (e.g. device type and id) of the Executor to a C++ stream:
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec);
 * ```
 *
 * One possible implementation would be to use RTTI to find the dynamic type of
 * the Executor, However, using the Operation feature of Ginkgo, there is a
 * more elegant approach which utilizes polymorphism. The first step is to
 * define an Operation that will print the desired information for each Executor
 * type.
 *
 * ```
 * class DeviceInfoPrinter : public gko::Operation {
 * public:
 *     explicit DeviceInfoPrinter(std::ostream &os) : os_(os) {}
 *
 *     void run(const gko::OmpExecutor *) const override { os_ << "OMP"; }
 *
 *     void run(const gko::CudaExecutor *exec) const override
 *     { os_ << "CUDA(" << exec->get_device_id() << ")"; }
 *
 *     void run(const gko::HipExecutor *exec) const override
 *     { os_ << "HIP(" << exec->get_device_id() << ")"; }
 *
 *     void run(const gko::DpcppExecutor *exec) const override
 *     { os_ << "DPC++(" << exec->get_device_id() << ")"; }
 *
 *     // This is optional, if not overloaded, defaults to OmpExecutor overload
 *     void run(const gko::ReferenceExecutor *) const override
 *     { os_ << "Reference CPU"; }
 *
 * private:
 *     std::ostream &os_;
 * };
 * ```
 *
 * Using DeviceInfoPrinter, the implementation of `operator<<` is as simple as
 * calling the run() method of the executor.
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec)
 * {
 *     DeviceInfoPrinter printer(os);
 *     exec.run(printer);
 *     return os;
 * }
 * ```
 *
 * Now it is possible to write the following code:
 *
 * ```
 * auto omp = gko::OmpExecutor::create();
 * std::cout << *omp << std::endl
 *           << *gko::CudaExecutor::create(0, omp) << std::endl
 *           << *gko::HipExecutor::create(0, omp) << std::endl
 *           << *gko::DpcppExecutor::create(0, omp) << std::endl
 *           << *gko::ReferenceExecutor::create() << std::endl;
 * ```
 *
 * which produces the expected output:
 *
 * ```
 * OMP
 * CUDA(0)
 * HIP(0)
 * DPC++(0)
 * Reference CPU
 * ```
 *
 * One might feel that this code is too complicated for such a simple task.
 * Luckily, there is an overload of the Executor::run() method, which is
 * designed to facilitate writing simple operations like this one. The method
 * takes four closures as input: one which is run for OMP, one for CUDA
 * executors, one for HIP executors, and the last one for DPC++ executors. Using
 * this method, there is no need to implement an Operation subclass:
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec)
 * {
 *     exec.run(
 *         [&]() { os << "OMP"; },  // OMP closure
 *         [&]() { os << "CUDA("    // CUDA closure
 *                    << static_cast<gko::CudaExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; },
 *         [&]() { os << "HIP("    // HIP closure
 *                    << static_cast<gko::HipExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; });
 *         [&]() { os << "DPC++("    // DPC++ closure
 *                    << static_cast<gko::DpcppExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; });
 *     return os;
 * }
 * ```
 *
 * Using this approach, however, it is impossible to distinguish between
 * a OmpExecutor and ReferenceExecutor, as both of them call the OMP closure.
 *
 * @ingroup Executor
 */
class Operation {
public:
#define GKO_DECLARE_RUN_OVERLOAD(_type, ...) \
    virtual void run(std::shared_ptr<const _type>) const

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_DECLARE_RUN_OVERLOAD);

#undef GKO_DECLARE_RUN_OVERLOAD

    // ReferenceExecutor overload can be defaulted to OmpExecutor's
    virtual void run(std::shared_ptr<const ReferenceExecutor> executor) const;

    /**
     * Returns the operation's name.
     *
     * @return the operation's name
     */
    virtual const char* get_name() const noexcept;
};


namespace detail {


/**
 * The RegisteredOperation class wraps a functor that will be called with the
 * executor statically cast to its dynamic type.
 *
 * It is used to implement the @ref GKO_REGISTER_OPERATION macro.
 *
 * @tparam Closure  the type of the functor of taking parameters
 *                  (std::shared_ptr<const Executor>)
 */
template <typename Closure>
class RegisteredOperation : public Operation {
public:
    /**
     * Creates a RegisteredOperation object from a functor and a name.
     *
     * @param name  the name to be used for this operation
     * @param op  a functor object which will be called with the executor.
     */
    RegisteredOperation(const char* name, Closure op)
        : name_(name), op_(std::move(op))
    {}

    const char* get_name() const noexcept override { return name_; }

    void run(std::shared_ptr<const ReferenceExecutor> exec) const override
    {
        op_(exec);
    }

    void run(std::shared_ptr<const OmpExecutor> exec) const override
    {
        op_(exec);
    }

    void run(std::shared_ptr<const CudaExecutor> exec) const override
    {
        op_(exec);
    }

    void run(std::shared_ptr<const HipExecutor> exec) const override
    {
        op_(exec);
    }

    void run(std::shared_ptr<const DpcppExecutor> exec) const override
    {
        op_(exec);
    }

private:
    const char* name_;
    Closure op_;
};


template <typename Closure>
RegisteredOperation<Closure> make_register_operation(const char* name,
                                                     Closure op)
{
    return RegisteredOperation<Closure>{name, std::move(op)};
}


}  // namespace detail


/**
 * Binds a set of device-specific kernels to an Operation.
 *
 * It also defines a helper function which creates the associated operation.
 * Any input arguments passed to the helper function are forwarded to the
 * kernel when the operation is executed.
 *
 * The kernels used to bind the operation are searched in `kernels::DEV_TYPE`
 * namespace, where `DEV_TYPE` is replaced by `omp`, `cuda`, `hip`, `dpcpp` and
 * `reference`.
 *
 * @param _name  operation name
 * @param _kernel  kernel which will be bound to the operation
 *
 * Example
 * -------
 *
 * ```c++
 * // define the omp, cuda, hip and reference kernels which will be bound to the
 * // operation
 * namespace kernels {
 * namespace omp {
 * void my_kernel(int x) {
 *      // omp code
 * }
 * }
 * namespace cuda {
 * void my_kernel(int x) {
 *      // cuda code
 * }
 * }
 * namespace hip {
 * void my_kernel(int x) {
 *      // hip code
 * }
 * }
 * namespace dpcpp {
 * void my_kernel(int x) {
 *      // dpcpp code
 * }
 * }
 * namespace reference {
 * void my_kernel(int x) {
 *     // reference code
 * }
 * }
 *
 * // Bind the kernels to the operation
 * GKO_REGISTER_OPERATION(my_op, my_kernel);
 *
 * int main() {
 *     // create executors
 *     auto omp = OmpExecutor::create();
 *     auto cuda = CudaExecutor::create(0, omp);
 *     auto hip = HipExecutor::create(0, omp);
 *     auto dpcpp = DpcppExecutor::create(0, omp);
 *     auto ref = ReferenceExecutor::create();
 *
 *     // create the operation
 *     auto op = make_my_op(5); // x = 5
 *
 *     omp->run(op);  // run omp kernel
 *     cuda->run(op);  // run cuda kernel
 *     hip->run(op);  // run hip kernel
 *     dpcpp->run(op);  // run DPC++ kernel
 *     ref->run(op);  // run reference kernel
 * }
 * ```
 *
 * @ingroup Executor
 */
#define GKO_REGISTER_OPERATION(_name, _kernel)                                 \
    template <typename... Args>                                                \
    auto make_##_name(Args&&... args)                                          \
    {                                                                          \
        return ::gko::detail::make_register_operation(                         \
            #_kernel, [&args...](auto exec) {                                  \
                using exec_type = decltype(exec);                              \
                if (std::is_same<                                              \
                        exec_type,                                             \
                        std::shared_ptr<const ::gko::ReferenceExecutor>>::     \
                        value) {                                               \
                    ::gko::kernels::reference::_kernel(                        \
                        std::dynamic_pointer_cast<                             \
                            const ::gko::ReferenceExecutor>(exec),             \
                        std::forward<Args>(args)...);                          \
                } else if (std::is_same<                                       \
                               exec_type,                                      \
                               std::shared_ptr<const ::gko::OmpExecutor>>::    \
                               value) {                                        \
                    ::gko::kernels::omp::_kernel(                              \
                        std::dynamic_pointer_cast<const ::gko::OmpExecutor>(   \
                            exec),                                             \
                        std::forward<Args>(args)...);                          \
                } else if (std::is_same<                                       \
                               exec_type,                                      \
                               std::shared_ptr<const ::gko::CudaExecutor>>::   \
                               value) {                                        \
                    ::gko::kernels::cuda::_kernel(                             \
                        std::dynamic_pointer_cast<const ::gko::CudaExecutor>(  \
                            exec),                                             \
                        std::forward<Args>(args)...);                          \
                } else if (std::is_same<                                       \
                               exec_type,                                      \
                               std::shared_ptr<const ::gko::HipExecutor>>::    \
                               value) {                                        \
                    ::gko::kernels::hip::_kernel(                              \
                        std::dynamic_pointer_cast<const ::gko::HipExecutor>(   \
                            exec),                                             \
                        std::forward<Args>(args)...);                          \
                } else if (std::is_same<                                       \
                               exec_type,                                      \
                               std::shared_ptr<const ::gko::DpcppExecutor>>::  \
                               value) {                                        \
                    ::gko::kernels::dpcpp::_kernel(                            \
                        std::dynamic_pointer_cast<const ::gko::DpcppExecutor>( \
                            exec),                                             \
                        std::forward<Args>(args)...);                          \
                } else {                                                       \
                    GKO_NOT_IMPLEMENTED;                                       \
                }                                                              \
            });                                                                \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


/**
 * Binds a host-side kernel (independent of executor type) to an Operation.
 *
 * It also defines a helper function which creates the associated operation.
 * Any input arguments passed to the helper function are forwarded to the
 * kernel when the operation is executed.
 * The kernel name is searched for in the namespace where this macro is called.
 * Host operations are used to make computations that are not part of the device
 * kernels visible to profiling loggers and benchmarks.
 *
 * @param _name  operation name
 * @param _kernel  kernel which will be bound to the operation
 *
 * Example
 * -------
 *
 * ```c++
 * void host_kernel(int) {
 *     // do some expensive computations
 * }
 *
 * // Bind the kernels to the operation
 * GKO_REGISTER_HOST_OPERATION(my_op, host_kernel);
 *
 * int main() {
 *     // create executor
 *     auto ref = ReferenceExecutor::create();
 *
 *     // create the operation
 *     auto op = make_my_op(5); // x = 5
 *
 *     ref->run(op);  // run host kernel
 * }
 * ```
 *
 * @ingroup Executor
 */
#define GKO_REGISTER_HOST_OPERATION(_name, _kernel)                          \
    template <typename... Args>                                              \
    auto make_##_name(Args&&... args)                                        \
    {                                                                        \
        return ::gko::detail::make_register_operation(                       \
            #_kernel,                                                        \
            [&args...](auto) { _kernel(std::forward<Args>(args)...); });     \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define GKO_DECLARE_EXECUTOR_FRIEND(_type, ...) friend class _type

/**
 * The first step in using the Ginkgo library consists of creating an
 * executor. Executors are used to specify the location for the data of linear
 * algebra objects, and to determine where the operations will be executed.
 * Ginkgo currently supports five different executor types:
 *
 * +    OmpExecutor specifies that the data should be stored and the associated
 *      operations executed on an OpenMP-supporting device (e.g. host CPU);
 * +    CudaExecutor specifies that the data should be stored and the
 *      operations executed on the NVIDIA GPU accelerator;
 * +    HipExecutor specifies that the data should be stored and the
 *      operations executed on either an NVIDIA or AMD GPU accelerator;
 * +    DpcppExecutor specifies that the data should be stored and the
 *      operations executed on an hardware supporting DPC++;
 * +    ReferenceExecutor executes a non-optimized reference implementation,
 *      which can be used to debug the library.
 *
 * The following code snippet demonstrates the simplest possible use of the
 * Ginkgo library:
 *
 * ```cpp
 * auto omp = gko::create<gko::OmpExecutor>();
 * auto A = gko::read_from_mtx<gko::matrix::Csr<float>>("A.mtx", omp);
 * ```
 *
 * First, we create a OMP executor, which will be used in the next line to
 * specify where we want the data for the matrix A to be stored.
 * The second line will read a matrix from the matrix market file 'A.mtx',
 * and store the data on the CPU in CSR format (gko::matrix::Csr is a
 * Ginkgo matrix class which stores its data in CSR format).
 * At this point, matrix A is bound to the CPU, and any routines called on it
 * will be performed on the CPU. This approach is usually desired in sparse
 * linear algebra, as the cost of individual operations is several orders of
 * magnitude lower than the cost of copying the matrix to the GPU.
 *
 * If matrix A is going to be reused multiple times, it could be beneficial to
 * copy it over to the accelerator, and perform the operations there, as
 * demonstrated by the next code snippet:
 *
 * ```cpp
 * auto cuda = gko::create<gko::CudaExecutor>(0, omp);
 * auto dA = gko::copy_to<gko::matrix::Csr<float>>(A.get(), cuda);
 * ```
 *
 * The first line of the snippet creates a new CUDA executor. Since there may be
 * multiple NVIDIA GPUs present on the system, the first parameter instructs the
 * library to use the first device (i.e. the one with device ID zero, as in
 * cudaSetDevice() routine from the CUDA runtime API). In addition, since GPUs
 * are not stand-alone processors, it is required to pass a "master" OmpExecutor
 * which will be used to schedule the requested CUDA kernels on the accelerator.
 *
 * The second command creates a copy of the matrix A on the GPU. Notice the use
 * of the get() method. As Ginkgo aims to provide automatic memory
 * management of its objects, the result of calling gko::read_from_mtx()
 * is a smart pointer (std::unique_ptr) to the created object. On the other
 * hand, as the library will not hold a reference to A once the copy is
 * completed, the input parameter for gko::copy_to() is a plain pointer.
 * Thus, the get() method is used to convert from a std::unique_ptr to a
 * plain pointer, as expected by gko::copy_to().
 *
 * As a side note, the gko::copy_to routine is far more powerful than just
 * copying data between different devices. It can also be used to convert data
 * between different formats. For example, if the above code used
 * gko::matrix::Ell as the template parameter, dA would be stored on the GPU,
 * in ELLPACK format.
 *
 * Finally, if all the processing of the matrix is supposed to be done on the
 * GPU, and a CPU copy of the matrix is not required, we could have read the
 * matrix to the GPU directly:
 *
 * ```cpp
 * auto omp = gko::create<gko::OmpExecutor>();
 * auto cuda = gko::create<gko::CudaExecutor>(0, omp);
 * auto dA = gko::read_from_mtx<gko::matrix::Csr<float>>("A.mtx", cuda);
 * ```
 * Notice that even though reading the matrix directly from a file to the
 * accelerator is not supported, the library is designed to abstract away the
 * intermediate step of reading the matrix to the CPU memory. This is a general
 * design approach taken by the library: in case an operation is not supported
 * by the device, the data will be copied to the CPU, the operation performed
 * there, and finally the results copied back to the device.
 * This approach makes using the library more concise, as explicit copies are
 * not required by the user. Nevertheless, this feature should be taken into
 * account when considering performance implications of using such operations.
 *
 * @ingroup Executor
 */
class Executor : public log::EnableLogging<Executor> {
    template <typename T>
    friend class detail::ExecutorBase;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_DECLARE_EXECUTOR_FRIEND);
    friend class ReferenceExecutor;

public:
    virtual ~Executor() = default;

    Executor() = default;
    Executor(Executor&) = delete;
    Executor(Executor&&) = delete;
    Executor& operator=(Executor&) = delete;
    Executor& operator=(Executor&&) = delete;

    /**
     * Runs the specified Operation using this Executor.
     *
     * @param op  the operation to run
     */
    virtual void run(const Operation& op) const = 0;

    /**
     * Runs one of the passed in functors, depending on the Executor type.
     *
     * @tparam ClosureOmp  type of op_omp
     * @tparam ClosureCuda  type of op_cuda
     * @tparam ClosureHip  type of op_hip
     * @tparam ClosureDpcpp  type of op_dpcpp
     *
     * @param op_omp  functor to run in case of a OmpExecutor or
     *                ReferenceExecutor
     * @param op_cuda  functor to run in case of a CudaExecutor
     * @param op_hip  functor to run in case of a HipExecutor
     * @param op_dpcpp  functor to run in case of a DpcppExecutor
     */
    template <typename ClosureOmp, typename ClosureCuda, typename ClosureHip,
              typename ClosureDpcpp>
    void run(const ClosureOmp& op_omp, const ClosureCuda& op_cuda,
             const ClosureHip& op_hip, const ClosureDpcpp& op_dpcpp) const
    {
        LambdaOperation<ClosureOmp, ClosureCuda, ClosureHip, ClosureDpcpp> op(
            op_omp, op_cuda, op_hip, op_dpcpp);
        this->run(op);
    }

    /**
     * Allocates memory in this Executor.
     *
     * @tparam T  datatype to allocate
     *
     * @param num_elems  number of elements of type T to allocate
     *
     * @throw AllocationError  if the allocation failed
     *
     * @return pointer to allocated memory
     */
    template <typename T>
    T* alloc(size_type num_elems) const
    {
        this->template log<log::Logger::allocation_started>(
            this, num_elems * sizeof(T));
        T* allocated = static_cast<T*>(this->raw_alloc(num_elems * sizeof(T)));
        this->template log<log::Logger::allocation_completed>(
            this, num_elems * sizeof(T), reinterpret_cast<uintptr>(allocated));
        return allocated;
    }

    /**
     * Frees memory previously allocated with Executor::alloc().
     *
     * If `ptr` is a `nullptr`, the function has no effect.
     *
     * @param ptr  pointer to the allocated memory block
     */
    void free(void* ptr) const noexcept
    {
        this->template log<log::Logger::free_started>(
            this, reinterpret_cast<uintptr>(ptr));
        this->raw_free(ptr);
        this->template log<log::Logger::free_completed>(
            this, reinterpret_cast<uintptr>(ptr));
    }

    /**
     * Copies data from another Executor.
     *
     * @tparam T  datatype to copy
     *
     * @param src_exec  Executor from which the memory will be copied
     * @param num_elems  number of elements of type T to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory
     *                  where the data will be copied to
     */
    template <typename T>
    void copy_from(ptr_param<const Executor> src_exec, size_type num_elems,
                   const T* src_ptr, T* dest_ptr) const
    {
        const auto src_loc = reinterpret_cast<uintptr>(src_ptr);
        const auto dest_loc = reinterpret_cast<uintptr>(dest_ptr);
        this->template log<log::Logger::copy_started>(
            src_exec.get(), this, src_loc, dest_loc, num_elems * sizeof(T));
        if (this != src_exec.get()) {
            src_exec->template log<log::Logger::copy_started>(
                src_exec.get(), this, src_loc, dest_loc, num_elems * sizeof(T));
        }
        try {
            this->raw_copy_from(src_exec.get(), num_elems * sizeof(T), src_ptr,
                                dest_ptr);
        } catch (NotSupported&) {
#if (GKO_VERBOSE_LEVEL >= 1) && !defined(NDEBUG)
            // Unoptimized copy. Try to go through the masters.
            // output to log when verbose >= 1 and debug build
            std::clog << "Not direct copy. Try to copy data from the masters."
                      << std::endl;
#endif
            auto src_master = src_exec->get_master().get();
            if (num_elems > 0 && src_master != src_exec.get()) {
                auto* master_ptr = src_exec->get_master()->alloc<T>(num_elems);
                src_master->copy_from<T>(src_exec, num_elems, src_ptr,
                                         master_ptr);
                this->copy_from<T>(src_master, num_elems, master_ptr, dest_ptr);
                src_master->free(master_ptr);
            }
        }
        this->template log<log::Logger::copy_completed>(
            src_exec.get(), this, src_loc, dest_loc, num_elems * sizeof(T));
        if (this != src_exec.get()) {
            src_exec->template log<log::Logger::copy_completed>(
                src_exec.get(), this, src_loc, dest_loc, num_elems * sizeof(T));
        }
    }

    /**
     * Copies data within this Executor.
     *
     * @tparam T  datatype to copy
     *
     * @param num_elems  number of elements of type T to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory
     *                  where the data will be copied to
     */
    template <typename T>
    void copy(size_type num_elems, const T* src_ptr, T* dest_ptr) const
    {
        this->copy_from(this, num_elems, src_ptr, dest_ptr);
    }

    /**
     * Retrieves a single element at the given location from executor memory.
     *
     * @tparam T  datatype to copy
     *
     * @param ptr  the pointer to the element to be copied
     *
     * @return the value stored at ptr
     */
    template <typename T>
    T copy_val_to_host(const T* ptr) const
    {
        T out{};
        this->get_master()->copy_from(this, 1, ptr, &out);
        return out;
    }

    /**
     * Returns the master OmpExecutor of this Executor.
     * @return the master OmpExecutor of this Executor.
     */
    virtual std::shared_ptr<Executor> get_master() noexcept = 0;

    /**
     * @copydoc get_master
     */
    virtual std::shared_ptr<const Executor> get_master() const noexcept = 0;

    /**
     * Synchronize the operations launched on the executor with its master.
     */
    virtual void synchronize() const = 0;

    /**
     * @copydoc Loggable::add_logger
     * @note This specialization keeps track of whether any propagating loggers
     *       were attached to the executor.
     * @see Logger::needs_propagation()
     */
    void add_logger(std::shared_ptr<const log::Logger> logger) override
    {
        this->propagating_logger_refcount_.fetch_add(
            logger->needs_propagation() ? 1 : 0);
        this->EnableLogging<Executor>::add_logger(logger);
    }

    /**
     * @copydoc Loggable::remove_logger
     * @note This specialization keeps track of whether any propagating loggers
     *       were attached to the executor.
     * @see Logger::needs_propagation()
     */
    void remove_logger(const log::Logger* logger) override
    {
        this->propagating_logger_refcount_.fetch_sub(
            logger->needs_propagation() ? 1 : 0);
        this->EnableLogging<Executor>::remove_logger(logger);
    }

    using EnableLogging<Executor>::remove_logger;

    /**
     * Sets the logger event propagation mode for the executor.
     * This controls whether events that happen at objects created on this
     * executor will also be logged at propagating loggers attached to the
     * executor.
     * @see Logger::needs_propagation()
     */
    void set_log_propagation_mode(log_propagation_mode mode)
    {
        log_propagation_mode_ = mode;
    }

    /**
     * Returns true iff events occurring at an object created on this executor
     * should be logged at propagating loggers attached to this executor, and
     * there is at least one such propagating logger.
     * @see Logger::needs_propagation()
     * @see Executor::set_log_propagation_mode(log_propagation_mode)
     */
    bool should_propagate_log() const
    {
        return this->propagating_logger_refcount_.load() > 0 &&
               log_propagation_mode_ == log_propagation_mode::automatic;
    }

    /**
     * Verifies whether the executors share the same memory.
     *
     * @param other  the other Executor to compare against
     *
     * @return whether the executors this and other share the same memory.
     */
    bool memory_accessible(const std::shared_ptr<const Executor>& other) const
    {
        return this->verify_memory_from(other.get());
    }

    virtual scoped_device_id_guard get_scoped_device_id_guard() const = 0;

protected:
    /**
     * A struct that abstracts the executor info for different executors
     * classes.
     */
    struct exec_info {
        /**
         * The id of the device.
         */
        int device_id = -1;

        /**
         * The type of the device, relevant only for the dpcpp executor.
         */
        std::string device_type;

        /**
         * The numa node of the executor.
         */
        int numa_node = -1;

        /**
         * The number of computing units in the executor.
         *
         * @note In CPU executors this is equivalent to the number of cores.
         *       In CUDA and HIP executors this is the number of Streaming
         *       Multiprocessors. In DPCPP, this is the number of computing
         *       units.
         */
        int num_computing_units = -1;

        /**
         * The number of processing units per computing unit in the executor.
         *
         * @note In CPU executors this is equivalent to the number of SIMD units
         *       per core.
         *       In CUDA and HIP executors this is the number of warps
         *       per SM.
         *       In DPCPP, this is currently number of hardware threads per eu.
         *       If the device does not support, it will be set as 1.
         *       (TODO: check)
         */
        int num_pu_per_cu = -1;

        /**
         * The sizes of the subgroup for the executor.
         *
         * @note In CPU executors this is invalid.
         *       In CUDA and HIP executors this is invalid.
         *       In DPCPP, this is the subgroup sizes for the device associated
         *       with the dpcpp executor.
         */
        std::vector<int> subgroup_sizes{};

        /**
         * The maximum subgroup size for the executor.
         *
         * @note In CPU executors this is invalid.
         *       In CUDA and HIP executors this is the warp size.
         *       In DPCPP, this is the maximum subgroup size for the device
         *       associated with the dpcpp executor.
         */
        int max_subgroup_size = -1;

        /**
         * The sizes of the work items for the executor.
         *
         * @note In CPU executors this is invalid.
         *       In CUDA and HIP executors this is the maximum number of threads
         *       in each dimension of a block (x, y, z).
         *       In DPCPP, this is the maximum number of workitems, in each
         *       direction of the workgroup for the device associated with the
         *       dpcpp executor.
         */
        std::vector<int> max_workitem_sizes{};

        /**
         * The sizes of the work items for the executor.
         *
         * @note In CPU executors this is invalid.
         *       In CUDA and HIP executors this is the maximum number of threads
         *       in block.
         *       In DPCPP, this is the maximum number of workitems that are
         *       permitted in a workgroup.
         */
        int max_workgroup_size;

        /**
         * The major version for CUDA/HIP device.
         */
        int major = -1;

        /**
         * The minor version for CUDA/HIP device.
         */
        int minor = -1;

        /**
         * The PCI bus id of the device.
         *
         * @note Only relevant for I/O devices (GPUs).
         */
        std::string pci_bus_id = std::string(13, 'x');

        /**
         * The host processing units closest to the device.
         *
         * @note Currently only relevant for CUDA, HIP and DPCPP executors.
         *       [Definition from hwloc
         *       documentation:](https://www.open-mpi.org/projects/hwloc/doc/v2.4.0/a00350.php)
         *       "The smallest processing element that can be represented by a
         *       hwloc object. It may be a single-core processor, a core of a
         *       multicore processor, or a single thread in a SMT processor"
         */
        std::vector<int> closest_pu_ids{};
    };

    /**
     * Gets the exec info struct
     *
     * @return  the exec_info struct
     */
    const exec_info& get_exec_info() const { return this->exec_info_; }

    /**
     * Allocates raw memory in this Executor.
     *
     * @param size  number of bytes to allocate
     *
     * @throw AllocationError  if the allocation failed
     *
     * @return raw pointer to allocated memory
     */
    virtual void* raw_alloc(size_type size) const = 0;

    /**
     * Frees memory previously allocated with Executor::alloc().
     *
     * If `ptr` is a `nullptr`, the function has no effect.
     *
     * @param ptr  pointer to the allocated memory block
     */
    virtual void raw_free(void* ptr) const noexcept = 0;

    /**
     * Copies raw data from another Executor.
     *
     * @param src_exec  Executor from which the memory will be copied
     * @param n_bytes  number of bytes to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory where the data
     *                  will be copied to
     */
    virtual void raw_copy_from(const Executor* src_exec, size_type n_bytes,
                               const void* src_ptr, void* dest_ptr) const = 0;

/**
 * @internal
 * Declares a raw_copy_to() overload for a specified Executor subclass.
 *
 * This is the second stage of the double dispatch emulation required to
 * implement raw_copy_from().
 *
 * @param _exec_type  the Executor subclass
 */
#define GKO_ENABLE_RAW_COPY_TO(_exec_type, ...)                              \
    virtual void raw_copy_to(const _exec_type* dest_exec, size_type n_bytes, \
                             const void* src_ptr, void* dest_ptr) const = 0

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_ENABLE_RAW_COPY_TO);

#undef GKO_ENABLE_RAW_COPY_TO

    /**
     * Verify the memory from another Executor.
     *
     * @param src_exec  Executor from which to verify the memory.
     *
     * @return whether this executor and src_exec share the same memory.
     */
    virtual bool verify_memory_from(const Executor* src_exec) const = 0;

/**
 * @internal
 * Declares a verify_memory_to() overload for a specified Executor subclass.
 *
 * This is the second stage of the double dispatch emulation required to
 * implement verify_memory_from().
 *
 * @param _exec_type  the Executor subclass
 */
#define GKO_ENABLE_VERIFY_MEMORY_TO(_exec_type, ...) \
    virtual bool verify_memory_to(const _exec_type* dest_exec) const = 0

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_ENABLE_VERIFY_MEMORY_TO);

    GKO_ENABLE_VERIFY_MEMORY_TO(ReferenceExecutor, ref);

#undef GKO_ENABLE_VERIFY_MEMORY_TO

    /**
     * Populates the executor specific info from the global machine topology
     * object.
     *
     * @param mach_topo the machine topology object.
     */
    virtual void populate_exec_info(const machine_topology* mach_topo) = 0;

    /**
     * Gets the modifiable exec info object
     *
     * @return  the pointer to the exec_info object
     */
    exec_info& get_exec_info() { return this->exec_info_; }

    exec_info exec_info_;

    log_propagation_mode log_propagation_mode_{log_propagation_mode::automatic};

    std::atomic<int> propagating_logger_refcount_{};

private:
    /**
     * The LambdaOperation class wraps four functor objects into an
     * Operation.
     *
     * The first object is called by the OmpExecutor, the second one by the
     * CudaExecutor, the third one by the HipExecutor and the last one by
     * the DpcppExecutor. When run on the
     * ReferenceExecutor, the implementation will launch the OpenMP version.
     *
     * @tparam ClosureOmp  the type of the first functor
     * @tparam ClosureCuda  the type of the second functor
     * @tparam ClosureHip  the type of the third functor
     * @tparam ClosureDpcpp  the type of the fourth functor
     */
    template <typename ClosureOmp, typename ClosureCuda, typename ClosureHip,
              typename ClosureDpcpp>
    class LambdaOperation : public Operation {
    public:
        /**
         * Creates an LambdaOperation object from four functors.
         *
         * @param op_omp  a functor object which will be called by OmpExecutor
         *                and ReferenceExecutor
         * @param op_cuda  a functor object which will be called by CudaExecutor
         * @param op_hip  a functor object which will be called by HipExecutor
         * @param op_dpcpp  a functor object which will be called by
         *                  DpcppExecutor
         */
        LambdaOperation(const ClosureOmp& op_omp, const ClosureCuda& op_cuda,
                        const ClosureHip& op_hip, const ClosureDpcpp& op_dpcpp)
            : op_omp_(op_omp),
              op_cuda_(op_cuda),
              op_hip_(op_hip),
              op_dpcpp_(op_dpcpp)
        {}

        void run(std::shared_ptr<const OmpExecutor>) const override
        {
            op_omp_();
        }

        void run(std::shared_ptr<const ReferenceExecutor>) const override
        {
            op_omp_();
        }

        void run(std::shared_ptr<const CudaExecutor>) const override
        {
            op_cuda_();
        }

        void run(std::shared_ptr<const HipExecutor>) const override
        {
            op_hip_();
        }

        void run(std::shared_ptr<const DpcppExecutor>) const override
        {
            op_dpcpp_();
        }

    private:
        ClosureOmp op_omp_;
        ClosureCuda op_cuda_;
        ClosureHip op_hip_;
        ClosureDpcpp op_dpcpp_;
    };
};


/**
 * This is a deleter that uses an executor's `free` method to deallocate the
 * data.
 *
 * @tparam T  the type of object being deleted
 *
 * @ingroup Executor
 */
template <typename T>
class executor_deleter {
public:
    using pointer = T*;

    /**
     * Creates a new deleter.
     *
     * @param exec  the executor used to free the data
     */
    explicit executor_deleter(std::shared_ptr<const Executor> exec)
        : exec_{exec}
    {}

    /**
     * Deletes the object.
     *
     * @param ptr  pointer to the object being deleted
     */
    void operator()(pointer ptr) const
    {
        if (exec_) {
            exec_->free(ptr);
        }
    }

private:
    std::shared_ptr<const Executor> exec_;
};

// a specialization for arrays
template <typename T>
class executor_deleter<T[]> {
public:
    using pointer = T[];

    explicit executor_deleter(std::shared_ptr<const Executor> exec)
        : exec_{exec}
    {}

    void operator()(pointer ptr) const
    {
        if (exec_) {
            exec_->free(ptr);
        }
    }

private:
    std::shared_ptr<const Executor> exec_;
};


namespace detail {


template <typename ConcreteExecutor>
class ExecutorBase : public Executor {
    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_DECLARE_EXECUTOR_FRIEND);
    friend class ReferenceExecutor;

public:
    using Executor::run;

    void run(const Operation& op) const override
    {
        this->template log<log::Logger::operation_launched>(this, &op);
        auto scope_guard = get_scoped_device_id_guard();
        op.run(self()->shared_from_this());
        this->template log<log::Logger::operation_completed>(this, &op);
    }

protected:
    void raw_copy_from(const Executor* src_exec, size_type n_bytes,
                       const void* src_ptr, void* dest_ptr) const override
    {
        src_exec->raw_copy_to(self(), n_bytes, src_ptr, dest_ptr);
    }

    virtual bool verify_memory_from(const Executor* src_exec) const override
    {
        return src_exec->verify_memory_to(self());
    }

private:
    ConcreteExecutor* self() noexcept
    {
        return static_cast<ConcreteExecutor*>(this);
    }

    const ConcreteExecutor* self() const noexcept
    {
        return static_cast<const ConcreteExecutor*>(this);
    }
};

#undef GKO_DECLARE_EXECUTOR_FRIEND


/**
 * Controls whether the DeviceReset function should be called thanks to a
 * boolean. Note that in any case, `DeviceReset` is called only after destroying
 * the last Ginkgo executor. Therefore, it is sufficient to set this flag to the
 * last living executor in Ginkgo. Setting this flag to an executor which is not
 * destroyed last has no effect.
 */
class EnableDeviceReset {
public:
    /**
     * Set the device reset capability.
     *
     * @param device_reset  whether to allow a device reset or not
     */
    GKO_DEPRECATED(
        "device_reset is no longer supported, call "
        "cudaDeviceReset/hipDeviceReset manually")
    void set_device_reset(bool device_reset) {}

    /**
     * Returns the current status of the device reset boolean for this executor.
     *
     * @return the current status of the device reset boolean for this executor.
     */
    GKO_DEPRECATED(
        "device_reset is no longer supported, call "
        "cudaDeviceReset/hipDeviceReset manually")
    bool get_device_reset() { return false; }

protected:
    /**
     * Instantiate an EnableDeviceReset class
     *
     * @param device_reset  the starting device_reset status. Defaults to false.
     */
    EnableDeviceReset() {}

    GKO_DEPRECATED(
        "device_reset is no longer supported, call "
        "cudaDeviceReset/hipDeviceReset manually")
    EnableDeviceReset(bool device_reset) {}
};


}  // namespace detail


#define GKO_OVERRIDE_RAW_COPY_TO(_executor_type, ...)                    \
    void raw_copy_to(const _executor_type* dest_exec, size_type n_bytes, \
                     const void* src_ptr, void* dest_ptr) const override


#define GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(dest_, bool_)                     \
    virtual bool verify_memory_to(const dest_* other) const override         \
    {                                                                        \
        return bool_;                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * This is the Executor subclass which represents the OpenMP device
 * (typically CPU).
 *
 * @ingroup exec_omp
 * @ingroup Executor
 */
class OmpExecutor : public detail::ExecutorBase<OmpExecutor>,
                    public std::enable_shared_from_this<OmpExecutor> {
    friend class detail::ExecutorBase<OmpExecutor>;

public:
    /**
     * Creates a new OmpExecutor.
     */
    static std::shared_ptr<OmpExecutor> create(
        std::shared_ptr<CpuAllocatorBase> alloc =
            std::make_shared<CpuAllocator>())
    {
        return std::shared_ptr<OmpExecutor>(new OmpExecutor(std::move(alloc)));
    }

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    void synchronize() const override;

    int get_num_cores() const
    {
        return this->get_exec_info().num_computing_units;
    }

    int get_num_threads_per_core() const
    {
        return this->get_exec_info().num_pu_per_cu;
    }

    static int get_num_omp_threads();

    scoped_device_id_guard get_scoped_device_id_guard() const override;

protected:
    OmpExecutor(std::shared_ptr<CpuAllocatorBase> alloc)
        : alloc_{std::move(alloc)}
    {
        this->OmpExecutor::populate_exec_info(machine_topology::get_instance());
    }

    void populate_exec_info(const machine_topology* mach_topo) override;

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(OmpExecutor, true);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HipExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaExecutor, false);

    bool verify_memory_to(const DpcppExecutor* dest_exec) const override;

    std::shared_ptr<CpuAllocatorBase> alloc_;
};


namespace kernels {
namespace omp {
using DefaultExecutor = OmpExecutor;
}  // namespace omp
}  // namespace kernels


/**
 * This is a specialization of the OmpExecutor, which runs the reference
 * implementations of the kernels used for debugging purposes.
 *
 * @ingroup exec_ref
 * @ingroup Executor
 */
class ReferenceExecutor : public OmpExecutor {
public:
    static std::shared_ptr<ReferenceExecutor> create(
        std::shared_ptr<CpuAllocatorBase> alloc =
            std::make_shared<CpuAllocator>())
    {
        return std::shared_ptr<ReferenceExecutor>(
            new ReferenceExecutor(std::move(alloc)));
    }

    scoped_device_id_guard get_scoped_device_id_guard() const override
    {
        return {this, 0};
    }

    void run(const Operation& op) const override
    {
        this->template log<log::Logger::operation_launched>(this, &op);
        op.run(std::static_pointer_cast<const ReferenceExecutor>(
            this->shared_from_this()));
        this->template log<log::Logger::operation_completed>(this, &op);
    }

protected:
    ReferenceExecutor(std::shared_ptr<CpuAllocatorBase> alloc)
        : OmpExecutor{std::move(alloc)}
    {
        this->ReferenceExecutor::populate_exec_info(
            machine_topology::get_instance());
    }

    void populate_exec_info(const machine_topology*) override
    {
        this->get_exec_info().device_id = -1;
        this->get_exec_info().num_computing_units = 1;
        this->get_exec_info().num_pu_per_cu = 1;
    }

    bool verify_memory_from(const Executor* src_exec) const override
    {
        return src_exec->verify_memory_to(this);
    }

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, true);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(OmpExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HipExecutor, false);
};


namespace kernels {
namespace reference {
using DefaultExecutor = ReferenceExecutor;
}  // namespace reference
}  // namespace kernels


/**
 * This is the Executor subclass which represents the CUDA device.
 *
 * @ingroup exec_cuda
 * @ingroup Executor
 */
class CudaExecutor : public detail::ExecutorBase<CudaExecutor>,
                     public std::enable_shared_from_this<CudaExecutor>,
                     public detail::EnableDeviceReset {
    friend class detail::ExecutorBase<CudaExecutor>;

public:
    /**
     * Creates a new CudaExecutor.
     *
     * @param device_id  the CUDA device id of this device
     * @param master  an executor on the host that is used to invoke the device
     * kernels
     * @param device_reset  this option no longer has any effect.
     * @param alloc_mode  the allocation mode that the executor should operate
     *                    on. See @allocation_mode for more details
     * @param stream  the stream to execute operations on.
     */
    GKO_DEPRECATED(
        "device_reset is deprecated entirely, call cudaDeviceReset directly. "
        "alloc_mode was replaced by the Allocator type "
        "hierarchy.")
    static std::shared_ptr<CudaExecutor> create(
        int device_id, std::shared_ptr<Executor> master, bool device_reset,
        allocation_mode alloc_mode = default_cuda_alloc_mode,
        CUstream_st* stream = nullptr);

    /**
     * Creates a new CudaExecutor with a custom allocator and device stream.
     *
     * @param device_id  the CUDA device id of this device
     * @param master  an executor on the host that is used to invoke the device
     *                kernels.
     * @param alloc  the allocator to use for device memory allocations.
     * @param stream  the stream to execute operations on.
     */
    static std::shared_ptr<CudaExecutor> create(
        int device_id, std::shared_ptr<Executor> master,
        std::shared_ptr<CudaAllocatorBase> alloc =
            std::make_shared<CudaAllocator>(),
        CUstream_st* stream = nullptr);

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    void synchronize() const override;

    scoped_device_id_guard get_scoped_device_id_guard() const override;

    /**
     * Get the CUDA device id of the device associated to this executor.
     */
    int get_device_id() const noexcept
    {
        return this->get_exec_info().device_id;
    }

    /**
     * Get the number of devices present on the system.
     */
    static int get_num_devices();

    /**
     * Get the number of warps per SM of this executor.
     */
    int get_num_warps_per_sm() const noexcept
    {
        return this->get_exec_info().num_pu_per_cu;
    }

    /**
     * Get the number of multiprocessor of this executor.
     */
    int get_num_multiprocessor() const noexcept
    {
        return this->get_exec_info().num_computing_units;
    }

    /**
     * Get the number of warps of this executor.
     */
    int get_num_warps() const noexcept
    {
        return this->get_exec_info().num_computing_units *
               this->get_exec_info().num_pu_per_cu;
    }

    /**
     * Get the warp size of this executor.
     */
    int get_warp_size() const noexcept
    {
        return this->get_exec_info().max_subgroup_size;
    }

    /**
     * Get the major version of compute capability.
     */
    int get_major_version() const noexcept
    {
        return this->get_exec_info().major;
    }

    /**
     * Get the minor version of compute capability.
     */
    int get_minor_version() const noexcept
    {
        return this->get_exec_info().minor;
    }

    /**
     * Get the cublas handle for this executor
     *
     * @return  the cublas handle (cublasContext*) for this executor
     */
    cublasContext* get_cublas_handle() const { return cublas_handle_.get(); }

    /**
     * Get the cusparse handle for this executor
     *
     * @return the cusparse handle (cusparseContext*) for this executor
     */
    cusparseContext* get_cusparse_handle() const
    {
        return cusparse_handle_.get();
    }

    /**
     * Get the closest PUs
     *
     * @return  the array of PUs closest to this device
     */
    std::vector<int> get_closest_pus() const
    {
        return this->get_exec_info().closest_pu_ids;
    }

    /**
     * Get the closest NUMA node
     *
     * @return  the closest NUMA node closest to this device
     */
    int get_closest_numa() const { return this->get_exec_info().numa_node; }

    /**
     * Returns the CUDA stream used by this executor. Can be nullptr for the
     * default stream.
     *
     * @return  the stream used to execute kernels and memory operations.
     */
    CUstream_st* get_stream() const { return stream_; }

protected:
    void set_gpu_property();

    void init_handles();

    CudaExecutor(int device_id, std::shared_ptr<Executor> master,
                 std::shared_ptr<CudaAllocatorBase> alloc, CUstream_st* stream)
        : alloc_{std::move(alloc)}, master_(master), stream_{stream}
    {
        this->get_exec_info().device_id = device_id;
        this->get_exec_info().num_computing_units = 0;
        this->get_exec_info().num_pu_per_cu = 0;
        this->CudaExecutor::populate_exec_info(
            machine_topology::get_instance());
        this->set_gpu_property();
        this->init_handles();
    }

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(OmpExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppExecutor, false);

    bool verify_memory_to(const HipExecutor* dest_exec) const override;

    bool verify_memory_to(const CudaExecutor* dest_exec) const override;

    void populate_exec_info(const machine_topology* mach_topo) override;

private:
    std::shared_ptr<Executor> master_;

    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T*)>>;
    handle_manager<cublasContext> cublas_handle_;
    handle_manager<cusparseContext> cusparse_handle_;
    std::shared_ptr<CudaAllocatorBase> alloc_;
    CUstream_st* stream_;
};


namespace kernels {
namespace cuda {
using DefaultExecutor = CudaExecutor;
}  // namespace cuda
}  // namespace kernels


/**
 * This is the Executor subclass which represents the HIP enhanced device.
 *
 * @ingroup exec_hip
 * @ingroup Executor
 */
class HipExecutor : public detail::ExecutorBase<HipExecutor>,
                    public std::enable_shared_from_this<HipExecutor>,
                    public detail::EnableDeviceReset {
    friend class detail::ExecutorBase<HipExecutor>;

public:
    /**
     * Creates a new HipExecutor.
     *
     * @param device_id  the HIP device id of this device
     * @param master  an executor on the host that is used to invoke the device
     *                kernels
     * @param device_reset  whether to reset the device after the object exits
     *                      the scope.
     * @param alloc_mode  the allocation mode that the executor should operate
     *                    on. See @allocation_mode for more details
     */
    GKO_DEPRECATED(
        "device_reset is deprecated entirely, call hipDeviceReset directly. "
        "alloc_mode was replaced by the Allocator type "
        "hierarchy.")
    static std::shared_ptr<HipExecutor> create(
        int device_id, std::shared_ptr<Executor> master, bool device_reset,
        allocation_mode alloc_mode = default_hip_alloc_mode,
        GKO_HIP_STREAM_STRUCT* stream = nullptr);

    static std::shared_ptr<HipExecutor> create(
        int device_id, std::shared_ptr<Executor> master,
        std::shared_ptr<HipAllocatorBase> alloc =
            std::make_shared<HipAllocator>(),
        GKO_HIP_STREAM_STRUCT* stream = nullptr);

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    void synchronize() const override;

    scoped_device_id_guard get_scoped_device_id_guard() const override;

    /**
     * Get the HIP device id of the device associated to this executor.
     */
    int get_device_id() const noexcept
    {
        return this->get_exec_info().device_id;
    }

    /**
     * Get the number of devices present on the system.
     */
    static int get_num_devices();

    /**
     * Get the number of warps per SM of this executor.
     */
    int get_num_warps_per_sm() const noexcept
    {
        return this->get_exec_info().num_pu_per_cu;
    }

    /**
     * Get the number of multiprocessor of this executor.
     */
    int get_num_multiprocessor() const noexcept
    {
        return this->get_exec_info().num_computing_units;
    }

    /**
     * Get the major version of compute capability.
     */
    int get_major_version() const noexcept
    {
        return this->get_exec_info().major;
    }

    /**
     * Get the minor version of compute capability.
     */
    int get_minor_version() const noexcept
    {
        return this->get_exec_info().minor;
    }

    /**
     * Get the number of warps of this executor.
     */
    int get_num_warps() const noexcept
    {
        return this->get_exec_info().num_computing_units *
               this->get_exec_info().num_pu_per_cu;
    }

    /**
     * Get the warp size of this executor.
     */
    int get_warp_size() const noexcept
    {
        return this->get_exec_info().max_subgroup_size;
    }

    /**
     * Get the hipblas handle for this executor
     *
     * @return  the hipblas handle (hipblasContext*) for this executor
     */
    hipblasContext* get_hipblas_handle() const { return hipblas_handle_.get(); }

    /**
     * Get the hipsparse handle for this executor
     *
     * @return the hipsparse handle (hipsparseContext*) for this executor
     */
    hipsparseContext* get_hipsparse_handle() const
    {
        return hipsparse_handle_.get();
    }

    /**
     * Get the closest NUMA node
     *
     * @return  the closest NUMA node closest to this device
     */
    int get_closest_numa() const { return this->get_exec_info().numa_node; }

    /**
     * Get the closest PUs
     *
     * @return  the array of PUs closest to this device
     */
    std::vector<int> get_closest_pus() const
    {
        return this->get_exec_info().closest_pu_ids;
    }

    GKO_HIP_STREAM_STRUCT* get_stream() const { return stream_; }

protected:
    void set_gpu_property();

    void init_handles();

    HipExecutor(int device_id, std::shared_ptr<Executor> master,
                std::shared_ptr<HipAllocatorBase> alloc,
                GKO_HIP_STREAM_STRUCT* stream)
        : master_{std::move(master)}, alloc_{std::move(alloc)}, stream_{stream}
    {
        this->get_exec_info().device_id = device_id;
        this->get_exec_info().num_computing_units = 0;
        this->get_exec_info().num_pu_per_cu = 0;
        this->HipExecutor::populate_exec_info(machine_topology::get_instance());
        this->set_gpu_property();
        this->init_handles();
    }

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(OmpExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(DpcppExecutor, false);

    bool verify_memory_to(const CudaExecutor* dest_exec) const override;

    bool verify_memory_to(const HipExecutor* dest_exec) const override;

    void populate_exec_info(const machine_topology* mach_topo) override;

private:
    std::shared_ptr<Executor> master_;

    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T*)>>;
    handle_manager<hipblasContext> hipblas_handle_;
    handle_manager<hipsparseContext> hipsparse_handle_;
    std::shared_ptr<HipAllocatorBase> alloc_;
    GKO_HIP_STREAM_STRUCT* stream_;
};


namespace kernels {
namespace hip {
using DefaultExecutor = HipExecutor;
}  // namespace hip
}  // namespace kernels


/**
 * This is the Executor subclass which represents a DPC++ enhanced device.
 *
 * @ingroup exec_dpcpp
 * @ingroup Executor
 */
class DpcppExecutor : public detail::ExecutorBase<DpcppExecutor>,
                      public std::enable_shared_from_this<DpcppExecutor> {
    friend class detail::ExecutorBase<DpcppExecutor>;

public:
    /**
     * Creates a new DpcppExecutor.
     *
     * @param device_id  the DPCPP device id of this device
     * @param master  an executor on the host that is used to invoke the device
     *                kernels
     * @param device_type  a string representing the type of device to consider
     *                     (accelerator, cpu, gpu or all).
     */
    static std::shared_ptr<DpcppExecutor> create(
        int device_id, std::shared_ptr<Executor> master,
        std::string device_type = "all",
        dpcpp_queue_property property = dpcpp_queue_property::in_order);

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    void synchronize() const override;

    scoped_device_id_guard get_scoped_device_id_guard() const override;

    /**
     * Get the DPCPP device id of the device associated to this executor.
     *
     * @return the DPCPP device id of the device associated to this executor
     */
    int get_device_id() const noexcept
    {
        return this->get_exec_info().device_id;
    }

    sycl::queue* get_queue() const { return queue_.get(); }

    /**
     * Get the number of devices present on the system.
     *
     * @param device_type  a string representing the device type
     *
     * @return the number of devices present on the system
     */
    static int get_num_devices(std::string device_type);

    /**
     * Get the available subgroup sizes for this device.
     *
     * @return the available subgroup sizes for this device
     */
    const std::vector<int>& get_subgroup_sizes() const noexcept
    {
        return this->get_exec_info().subgroup_sizes;
    }

    /**
     * Get the number of Computing Units of this executor.
     *
     * @return the number of Computing Units of this executor
     */
    int get_num_computing_units() const noexcept
    {
        return this->get_exec_info().num_computing_units;
    }

    /**
     * Get the number of subgroups of this executor.
     */
    int get_num_subgroups() const noexcept
    {
        return this->get_exec_info().num_computing_units *
               this->get_exec_info().num_pu_per_cu;
    }

    /**
     * Get the maximum work item sizes.
     *
     * @return the maximum work item sizes
     */
    const std::vector<int>& get_max_workitem_sizes() const noexcept
    {
        return this->get_exec_info().max_workitem_sizes;
    }

    /**
     * Get the maximum workgroup size.
     *
     * @return the maximum workgroup size
     */
    int get_max_workgroup_size() const noexcept
    {
        return this->get_exec_info().max_workgroup_size;
    }

    /**
     * Get the maximum subgroup size.
     *
     * @return the maximum subgroup size
     */
    int get_max_subgroup_size() const noexcept
    {
        return this->get_exec_info().max_subgroup_size;
    }

    /**
     * Get a string representing the device type.
     *
     * @return a string representing the device type
     */
    std::string get_device_type() const noexcept
    {
        return this->get_exec_info().device_type;
    }

protected:
    void set_device_property(
        dpcpp_queue_property property = dpcpp_queue_property::in_order);

    DpcppExecutor(
        int device_id, std::shared_ptr<Executor> master,
        std::string device_type = "all",
        dpcpp_queue_property property = dpcpp_queue_property::in_order)
        : master_(master)
    {
        std::for_each(device_type.begin(), device_type.end(),
                      [](char& c) { c = std::tolower(c); });
        this->get_exec_info().device_type = std::string(device_type);
        this->get_exec_info().device_id = device_id;
        this->set_device_property(property);
    }

    void populate_exec_info(const machine_topology* mach_topo) override;

    void* raw_alloc(size_type size) const override;

    void raw_free(void* ptr) const noexcept override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_OVERRIDE_RAW_COPY_TO);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(CudaExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(HipExecutor, false);

    GKO_DEFAULT_OVERRIDE_VERIFY_MEMORY(ReferenceExecutor, false);

    bool verify_memory_to(const OmpExecutor* dest_exec) const override;

    bool verify_memory_to(const DpcppExecutor* dest_exec) const override;

private:
    std::shared_ptr<Executor> master_;

    template <typename T>
    using queue_manager = std::unique_ptr<T, std::function<void(T*)>>;
    queue_manager<sycl::queue> queue_;
};


namespace kernels {
namespace dpcpp {
using DefaultExecutor = DpcppExecutor;
}  // namespace dpcpp
}  // namespace kernels


#undef GKO_OVERRIDE_RAW_COPY_TO


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_EXECUTOR_HPP_
