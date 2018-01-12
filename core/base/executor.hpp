#ifndef GKO_CORE_EXECUTOR_HPP_
#define GKO_CORE_EXECUTOR_HPP_


#include "core/base/types.hpp"


#include <memory>
#include <tuple>
#include <type_traits>


namespace gko {


#define FORWARD_DECLARE(_type, _unused) class _type

GKO_ENABLE_FOR_ALL_EXECUTORS(FORWARD_DECLARE);

#undef FORWARD_DECLARE


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
 *     void run(const gko::CpuExecutor *) const override { os_ << "CPU"; }
 *
 *     void run(const gko::GpuExecutor *exec) const override
 *     { os_ << "GPU(" << exec->get_device_id() << ")"; }
 *
 *     // This is optional, if not overloaded, defaults to CpuExecutor overload
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
 * auto cpu = gko::CpuExecutor::create();
 * std::cout << *cpu << std::endl
 *           << *gko::GpuExecutor::create(0, cpu) << std::endl
 *           << *gko::ReferenceExecutor::create() << std::endl;
 * ```
 *
 * which produces the expected output:
 *
 * ```
 * CPU
 * GPU(0)
 * Reference CPU
 * ```
 *
 * One might feel that this code is too complicated for such a simple task.
 * Luckily, there is an overload of the Executor::run() method, which is
 * designed to facilitate writing simple operations like this one. The method
 * takes two closures as input: one which is run for CPU, and the other one for
 * GPU executors. Using this method, there is no need to implement an Operation
 * subclass:
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec)
 * {
 *     exec.run(
 *         [&]() { os << "CPU"; },  // CPU closure
 *         [&]() { os << "GPU("     // GPU closure
 *                    << static_cast<gko::GpuExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; });
 *     return os;
 * }
 * ```
 *
 * Using this approach, however, it is impossible to distinguish between
 * a CpuExecutor and ReferenceExecutor, as both of them call the CPU closure.
 */
class Operation {
public:
#define DECLARE_RUN_OVERLOAD(_type, _unused) \
    virtual void run(const _type *) const;

    GKO_ENABLE_FOR_ALL_EXECUTORS(DECLARE_RUN_OVERLOAD);

#undef DECLARE_RUN_OVERLOAD

    // ReferenceExecutor overload can be defaulted to CpuExecutor's
    virtual void run(const ReferenceExecutor *executor) const;
};


namespace detail {


template <int K, int... Ns, typename F, typename Tuple>
typename std::enable_if<(K == 0)>::type call_impl(F f, Tuple &data)
{
    f(std::get<Ns>(data)...);
}

template <int K, int... Ns, typename F, typename Tuple>
typename std::enable_if<(K > 0)>::type call_impl(F f, Tuple &data)
{
    call_impl<K - 1, K - 1, Ns...>(f, data);
}

template <typename F, typename... Args>
void call(F f, std::tuple<Args...> &data)
{
    call_impl<sizeof...(Args)>(f, data);
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
 * namespace, where `DEV_TYPE` is replaced by `cpu`, `gpu` and `reference`.
 *
 * @param _name  operation name
 * @param _kernel  kernel which will be bound to the operation
 *
 * Example
 * -------
 *
 * ```c++
 * // define the cpu, gpu and reference kernels which will be bound to the
 * // operation
 * namespace kernels {
 * namespace cpu {
 * void my_kernel(int x) {
 *      // cpu code
 * }
 * }
 * namespace gpu {
 * void my_kernel(int x) {
 *      // gpu code
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
 *     auto cpu = CpuExecutor::create();
 *     auto gpu = GpuExecutor::create(cpu, 0);
 *     auto ref = ReferenceExecutor::create();
 *
 *     // create the operation
 *     auto op = make_my_op_operation(5); // x = 5
 *
 *     cpu->run(op);  // run cpu kernel
 *     gpu->run(op);  // run gpu kernel
 *     ref->run(op);  // run reference kernel
 * }
 * ```
 */
#define GKO_REGISTER_OPERATION(_name, _kernel)                                 \
    template <typename... Args>                                                \
    class _name##_operation : public Operation {                               \
    public:                                                                    \
        _name##_operation(Args &&... args) : data(std::forward<Args>(args)...) \
        {}                                                                     \
                                                                               \
        void run(const CpuExecutor *) const override                           \
        {                                                                      \
            detail::call(kernels::cpu::_kernel, data);                         \
        }                                                                      \
                                                                               \
        void run(const GpuExecutor *) const override                           \
        {                                                                      \
            detail::call(kernels::gpu::_kernel, data);                         \
        }                                                                      \
                                                                               \
        void run(const ReferenceExecutor *) const override                     \
        {                                                                      \
            detail::call(kernels::reference::_kernel, data);                   \
        }                                                                      \
                                                                               \
    private:                                                                   \
        mutable std::tuple<Args &&...> data;                                   \
    };                                                                         \
                                                                               \
    template <typename... Args>                                                \
    static _name##_operation<Args...> make_##_name##_operation(                \
        Args &&... args)                                                       \
    {                                                                          \
        return _name##_operation<Args...>(std::forward<Args>(args)...);        \
    }


/**
 * The first step in using the Ginkgo library consists of creating an
 * executor. Executors are used to specify the location for the data of linear
 * algebra objects, and to determine where the operations will be executed.
 * Ginkgo currently supports three different executor types:
 *
 * +    CpuExecutor specifies that the data should be stored and the associated
 *      operations executed on the host CPU;
 * +    GpuExecutor specifies that the data should be stored and the
 *      operations executed on the NVIDIA GPU accelerator;
 * +    ReferenceExecutor executes a non-optimized reference implementation,
 *      which can be used to debug the library.
 *
 * The following code snippet demonstrates the simplest possible use of the
 * Ginkgo library:
 *
 * ```cpp
 * auto cpu = gko::create<gko::CpuExecutor>();
 * auto A = gko::read_from_mtx<gko::matrix::Csr<float>>("A.mtx", cpu);
 * ```
 *
 * First, we create a CPU executor, which will be used in the next line to
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
 * auto gpu = gko::create<gko::GpuExecutor>(0, cpu);
 * auto dA = gko::copy_to<gko::matrix::Csr<float>>(A.get(), gpu);
 * ```
 *
 * The first line of the snippet creates a new GPU executor. Since there may be
 * multiple GPUs present on the system, the first parameter instructs the
 * library to use the first device (i.e. the one with device ID zero, as in
 * cudaSetDevice() routine from the CUDA runtime API). In addition, since GPUs
 * are not stand-alone processors, it is required to pass a "master" CpuExecutor
 * which will be used to schedule the requested GPU kernels on the accelerator.
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
 * auto cpu = gko::create<gko::CpuExecutor>();
 * auto gpu = gko::create<gko::GpuExecutor>(0, cpu);
 * auto dA = gko::read_from_mtx<gko::matrix::Csr<float>>("A.mtx", gpu);
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
 */
class Executor {
    template <typename T>
    friend class detail::ExecutorBase;

public:
    virtual ~Executor() = default;

    Executor() = default;
    Executor(Executor &) = delete;
    Executor(Executor &&) = default;
    Executor &operator=(Executor &) = delete;
    Executor &operator=(Executor &&) = default;

    /**
     * Runs the specified Operation using this Executor.
     *
     * @param op  the operation to run
     */
    virtual void run(const Operation &op) const = 0;

    /**
     * Runs one of the passed in functors, depending on the Executor type.
     *
     * @tparam ClosureCpu  type of op_cpu
     * @tparam ClosureGpu  type of op_gpu
     *
     * @param op_cpu  functor to run in case of a CpuExecutor or
     *                ReferenceExecutor
     * @param op_gpu  functor to run in case of a GpuExecutor
     */
    template <typename ClosureCpu, typename ClosureGpu>
    void run(const ClosureCpu &op_cpu, const ClosureGpu &op_gpu) const
    {
        LambdaOperation<ClosureCpu, ClosureGpu> op(op_cpu, op_gpu);
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
    T *alloc(size_type num_elems) const
    {
        return static_cast<T *>(this->raw_alloc(num_elems * sizeof(T)));
    }

    /**
     * Frees memory previously allocated with Executor::alloc().
     *
     * If `ptr` is a `nullptr`, the function has no effect.
     *
     * @param ptr  pointer to the allocated memory block
     */
    virtual void free(void *ptr) const noexcept = 0;

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
    void copy_from(const Executor *src_exec, size_type num_elems,
                   const T *src_ptr, T *dest_ptr) const
    {
        this->raw_copy_from(src_exec, num_elems * sizeof(T), src_ptr, dest_ptr);
    }

    /**
     * Returns the master CpuExecutor of this Executor.
     */
    virtual std::shared_ptr<CpuExecutor> get_master() noexcept = 0;

    /**
     * @copydoc get_master
     */
    virtual std::shared_ptr<const CpuExecutor> get_master() const noexcept = 0;

protected:
    /**
     * Allocates raw memory in this Executor.
     *
     * @param size  number of bytes to allocate
     *
     * @throw AllocationError  if the allocation failed
     *
     * @return raw pointer to allocated memory
     */
    virtual void *raw_alloc(size_type size) const = 0;

    /**
     * Copies raw data from another Executor.
     *
     * @param dest_exec  Executor from which the memory will be copied
     * @param n_bytes  number of bytes to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory where the data
     *                  will be copied to
     */
    virtual void raw_copy_from(const Executor *src_exec, size_type n_bytes,
                               const void *src_ptr, void *dest_ptr) const = 0;

/**
 * @internal
 * Declares a raw_copy_to() overload for a specified Executor subclass.
 *
 * This is the second stage of the double dispatch emulation required to
 * implement raw_copy_from().
 *
 * @param _exec_type  the Executor subclass
 */
#define ENABLE_RAW_COPY_TO(_exec_type, _unused)                              \
    virtual void raw_copy_to(const _exec_type *dest_exec, size_type n_bytes, \
                             const void *src_ptr, void *dest_ptr) const = 0

    GKO_ENABLE_FOR_ALL_EXECUTORS(ENABLE_RAW_COPY_TO);

#undef ENABLE_RAW_COPY_TO

private:
    /**
     * The LambdaOperation class wraps two functor objects into an Operation.
     *
     * The first object is called by the CpuExecutor, while the other one by the
     * GpuExecutor. When run on the ReferenceExecutor, the implementation will
     * launch the CPU version.
     *
     * @tparam ClosureCpu  the type of the first functor
     * @tparam ClosureGpu  the type of the second functor
     */
    template <typename ClosureCpu, typename ClosureGpu>
    class LambdaOperation : public Operation {
    public:
        /**
         * Creates an LambdaOperation object from two functors.
         *
         * @param op_cpu  a functor object which will be called by CpuExecutor
         *                and ReferenceExecutor
         * @param op_gpu  a functor object which will be called by GpuExecutor
         */
        LambdaOperation(const ClosureCpu &op_cpu, const ClosureGpu &op_gpu)
            : op_cpu_(op_cpu), op_gpu_(op_gpu)
        {}

        void run(const CpuExecutor *) const override { op_cpu_(); }

        void run(const GpuExecutor *) const override { op_gpu_(); }

    private:
        ClosureCpu op_cpu_;
        ClosureGpu op_gpu_;
    };
};


namespace detail {


template <typename ConcreteExecutor>
class ExecutorBase : public Executor {
public:
    void run(const Operation &op) const override { op.run(self()); }

protected:
    void raw_copy_from(const Executor *src_exec, size_type n_bytes,
                       const void *src_ptr, void *dest_ptr) const override
    {
        src_exec->raw_copy_to(self(), n_bytes, src_ptr, dest_ptr);
    }

private:
    ConcreteExecutor *self() noexcept
    {
        return static_cast<ConcreteExecutor *>(this);
    }

    const ConcreteExecutor *self() const noexcept
    {
        return static_cast<const ConcreteExecutor *>(this);
    }
};


}  // namespace detail


#define OVERRIDE_RAW_COPY_TO(_executor_type, _unused)                    \
    void raw_copy_to(const _executor_type *dest_exec, size_type n_bytes, \
                     const void *src_ptr, void *dest_ptr) const override


/**
 * This is the Executor subclass which represents the CPU device.
 */
class CpuExecutor : public detail::ExecutorBase<CpuExecutor>,
                    public std::enable_shared_from_this<CpuExecutor> {
public:
    /**
     * Creates a new CpuExecutor.
     */
    static std::shared_ptr<CpuExecutor> create()
    {
        return std::shared_ptr<CpuExecutor>(new CpuExecutor());
    }

    void free(void *ptr) const noexcept override;

    std::shared_ptr<CpuExecutor> get_master() noexcept override;

    std::shared_ptr<const CpuExecutor> get_master() const noexcept override;

protected:
    CpuExecutor() = default;

    void *raw_alloc(size_type size) const override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(OVERRIDE_RAW_COPY_TO);
};


/**
 * This is a specialization of the CpuExecutor, which runs the reference
 * implementations of the kernels used for debugging purposes.
 */
class ReferenceExecutor : public CpuExecutor {
public:
    static std::shared_ptr<ReferenceExecutor> create()
    {
        return std::shared_ptr<ReferenceExecutor>(new ReferenceExecutor());
    }

    void run(const Operation &op) const override { op.run(this); }

protected:
    ReferenceExecutor() = default;
};


/**
 * This is the Executor subclass which represents the GPU device.
 */
class GpuExecutor : public detail::ExecutorBase<GpuExecutor> {
public:
    /**
     * Creates a new GpuExecutor.
     *
     * @param device  the CUDA device number of this device
     * @param master  a CPU executor used to invoke the device kernels
     */
    static std::shared_ptr<GpuExecutor> create(
        int device_id, std::shared_ptr<CpuExecutor> master)
    {
        return std::shared_ptr<GpuExecutor>(
            new GpuExecutor(device_id, std::move(master)));
    }

    void free(void *ptr) const noexcept override;

    std::shared_ptr<CpuExecutor> get_master() noexcept override;

    std::shared_ptr<const CpuExecutor> get_master() const noexcept override;

    int get_device_id() const noexcept { return device_id_; }

protected:
    GpuExecutor(int device_id, std::shared_ptr<CpuExecutor> master)
        : device_id_(device_id), master_(master)
    {}

    void *raw_alloc(size_type size) const override;

    GKO_ENABLE_FOR_ALL_EXECUTORS(OVERRIDE_RAW_COPY_TO);

private:
    int device_id_;
    std::shared_ptr<CpuExecutor> master_;
};


#undef OVERRIDE_RAW_COPY_TO


}  // namespace gko


#endif  // GKO_CORE_EXECUTOR_HPP_
