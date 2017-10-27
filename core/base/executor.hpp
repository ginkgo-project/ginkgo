#ifndef MSPARSE_CORE_EXECUTOR_HPP_
#define MSPARSE_CORE_EXECUTOR_HPP_


#include "core/base/types.hpp"


#include <memory>


namespace msparse {


#define MSPARSE_ENABLE_FOR_ALL_EXECUTORS(_enable_macro) \
    _enable_macro(CpuExecutor, cpu);                    \
    _enable_macro(GpuExecutor, gpu)


#define FORWARD_DECLARE(_type, _unused) class _type

MSPARSE_ENABLE_FOR_ALL_EXECUTORS(FORWARD_DECLARE);

#undef FORWARD_DECLARE


class ReferenceExecutor;


/**
 * The Operation class is a base class for all operations which should be run
 * on supported devices.
 *
 * Inheriting this class is only needed if the implementation differs for
 * different Executors. If this is the case, the implementer has to override
 * each of the run() overloads with the appropriate implementation.
 *
 * Finally, an implementer instance should be passed to the
 * Executor::run() method of the Executor containing the input/output data.
 */
class Operation {
public:
#define DECLARE_RUN_OVERLOAD(_type, _unused) \
    virtual void run(const _type *) const = 0

    MSPARSE_ENABLE_FOR_ALL_EXECUTORS(DECLARE_RUN_OVERLOAD);

#undef DECLARE_RUN_OVERLOAD

    virtual void run(const ReferenceExecutor *executor) const;
};


/**
 * The first step in using the MAGMA-sparse library consists of creating an
 * executor. Executors are used to specify the location for the data of linear
 * algebra objects, and to determine where the operations will be executed.
 * MAGMA-sparse currently supports two different executor types:
 * +    CpuExecutor specifies that the data should be stored and the associated
 *      operations executed on the host CPU,
 * +    GpuExecutor specifies that the data should be stored and the
 *      operations executed on the NVIDIA GPU accelerator.
 * +    DebugExecutor executes a non-optimized reference implementation, which
 *      can be used to debug the library.
 *
 * The following code snippet demonstrates the simplest possible use of the
 * MAGMA-sparse library:
 *
 * ```cpp
 * auto cpu = msparse::create<msparse::CpuExecutor>();
 * auto A = msparse::read_from_mtx<msparse::CsrMatrix<float>>("A.mtx", cpu);
 * ```
 *
 * First, we create a CPU executor, which will be used in the next line to
 * specify where we want the data for the matrix A to be stored.
 * The second line will read a matrix from a matrix market file 'A.mtx',
 * and store the data on the CPU in CSR format (msparse::CsrMatrix is a
 * MAGMA-sparse Matrix class which stores its data in CSR format).
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
 * auto gpu = msparse::create<msparse::GpuExecutor>(0, cpu);
 * auto dA = msparse::copy_to<msparse::CsrMatrix<float>>(A.get(), gpu);
 * ```
 * The first line of the snippet creates a new GPU executor. Since there may be
 * multiple GPUs present on the system, the first parameter instructs the
 * library to use the first device (i.e. the one with device ID zero, as in
 * cudaSetDevice() routine from the CUDA runtime API). In addition, since GPUs
 * are not stand-alone processors, it is required to pass a CpuExecutor which
 * will be used to schedule the requested GPU kernels on the accelerator.
 *
 * The second command creates a copy of the matrix A on the GPU. Notice the use
 * of the get() method. As MAGMA-sparse aims to provide automatic memory
 * management of its objects, the result of calling msparse::read_from_mtx()
 * is a smart pointer (std::unique_ptr) to the created object. On the other
 * hand, as the library will not hold a reference to A once the copy is
 * completed, the input parameter for msparse::copy_to is a plain pointer.
 * Thus, the get() routine is used to convert from a std::unique_ptr to a
 * plain pointer expected by the routine.
 *
 * As a side note, the msparse::copy_to routine is far more powerful than just
 * copying data between different devices. It can also be used to convert data
 * between different formats. For example, if the above code used
 * msparse::EllMatrix as the template parameter, dA would be stored on the GPU,
 * in ELLPACK format.
 *
 * Finally, if all the processing of the matrix is supposed to be done on the
 * GPU, and a CPU copy of the matrix is not required, we could have read the
 * matrix to the GPU directly:
 *
 * ```cpp
 * auto cpu = msparse::create<msparse::CpuExecutor>();
 * auto gpu = msparse::create<msparse::GpuExecutor>(0, cpu);
 * auto dA = msparse::read_from_mtx<msparse::CsrMatrix<float>>("A.mtx", gpu);
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
    friend class ExecutorBase;

public:
    virtual ~Executor() = default;

    Executor() = default;
    Executor(Executor &) = delete;
    Executor(Executor &&) = default;
    Executor &operator=(Executor &) = delete;
    Executor &operator=(Executor &&) = default;

    /**
     * Runs the specified Operation using this executor.
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
     * @param n_elems  number of elements of type T to allocate
     *
     * @throw AllocationError  if the allocation failed
     *
     * @return pointer to allocated memory
     */
    template <typename T>
    T *alloc(size_type n_elems) const
    {
        return static_cast<T *>(this->raw_alloc(n_elems * sizeof(T)));
    }

    /**
     * Frees memory previously allocated with Executor::alloc().
     *
     * @param ptr  pointer to the allocated memory block
     */
    virtual void free(void *ptr) const noexcept = 0;

    /**
     * Copies data from another Executor.
     *
     * @tparam T  datatype to copy
     *
     * @param srch_exec  Executor from which the memory will be copied
     * @param n_elems  number of elements of type T to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory
     *                  where the data will be copied to
     */
    template <typename T>
    void copy_from(const Executor *src_exec, size_type n_elems,
                   const T *src_ptr, T *dest_ptr) const
    {
        this->raw_copy_from(src_exec, n_elems * sizeof(T), src_ptr, dest_ptr);
    }

    /**
     * @internal
     * Returns the master CpuExecutor of this Executor.
     */
    virtual std::shared_ptr<CpuExecutor> get_master() noexcept = 0;

    /**
     * @internal
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
     * Copies raw data to another Executor.
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

    MSPARSE_ENABLE_FOR_ALL_EXECUTORS(ENABLE_RAW_COPY_TO);

#undef ENABLE_RAW_COPY_TO

private:
    /**
     * The LambdaOperation class wraps two functor objects into an Operation.
     *
     * The first object is called by the CpuExecutor, while the other by the
     * GpuExecutor. When run on the DebugExecutor, the implementation will
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


template <typename ConcreteExecutor>
class ExecutorBase : public Executor {
public:
    ExecutorBase() = default;

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


#define OVERRIDE_RAW_COPY_TO(_executor_type, _unused)                    \
    void raw_copy_to(const _executor_type *dest_exec, size_type n_bytes, \
                     const void *src_ptr, void *dest_ptr) const override


/**
 * This is the Executor subclass which represents the CPU device.
 */
class CpuExecutor : public ExecutorBase<CpuExecutor>,
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

    MSPARSE_ENABLE_FOR_ALL_EXECUTORS(OVERRIDE_RAW_COPY_TO);
};


/**
 * This is a specialization of the CpuExecutor, which runs the reference
 * implementations of the kernels used for debugging purposes.
 */
class ReferenceExecutor : public CpuExecutor {
    static std::shared_ptr<ReferenceExecutor> create()
    {
        return std::shared_ptr<ReferenceExecutor>(new ReferenceExecutor());
    }

    void run(const Operation &op) const override { op.run(this); }

protected:
    ReferenceExecutor() = default;
};


void Operation::run(const ReferenceExecutor *executor) const
{
    this->run(static_cast<const CpuExecutor *>(executor));
}


/**
 * This is the Executor subclass which represents the GPU device.
 */
class GpuExecutor : public ExecutorBase<GpuExecutor> {
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
    GpuExecutor(int device_id, std::shared_ptr<CpuExecutor> master);

    void *raw_alloc(size_type size) const override;

    MSPARSE_ENABLE_FOR_ALL_EXECUTORS(OVERRIDE_RAW_COPY_TO);

private:
    int device_id_;
    std::shared_ptr<CpuExecutor> master_;
};


#undef OVERRIDE_RAW_COPY_TO


}  // namespace msparse


#endif  // MSPARSE_CORE_EXECUTOR_HPP_
