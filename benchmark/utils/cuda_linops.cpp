// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <memory>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>


#include "benchmark/utils/sparselib_linops.hpp"
#include "benchmark/utils/types.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"


class cusparse_csr {};
class cusparse_csrmp {};
class cusparse_csrmm {};
class cusparse_hybrid {};
class cusparse_coo {};
class cusparse_ell {};
class cusparse_gcsr {};
class cusparse_gcoo {};
class cusparse_csrex {};
class cusparse_gcsr2 {};


namespace detail {


class CusparseBase : public gko::LinOp {
public:
    cusparseMatDescr_t get_descr() const { return this->descr_.get(); }

    // Return shared pointer not plain pointer such that CusparseGenericSpMV
    // uses gko::array to allocate buffer.
    std::shared_ptr<const gko::CudaExecutor> get_gpu_exec() const
    {
        return std::dynamic_pointer_cast<const gko::CudaExecutor>(
            this->get_executor());
    }

protected:
    CusparseBase(std::shared_ptr<const gko::Executor> exec,
                 const gko::dim<2>& size = gko::dim<2>{})
        : gko::LinOp(exec, size)
    {
        if (this->get_gpu_exec() == nullptr) {
            GKO_NOT_IMPLEMENTED;
        }
        this->initialize_descr();
    }

    ~CusparseBase() = default;

    CusparseBase(const CusparseBase& other) = delete;

    CusparseBase& operator=(const CusparseBase& other)
    {
        if (this != &other) {
            gko::LinOp::operator=(other);
            this->initialize_descr();
        }
        return *this;
    }

    void initialize_descr()
    {
        auto exec = this->get_gpu_exec();
        auto guard = exec->get_scoped_device_id_guard();
        this->descr_ = handle_manager<cusparseMatDescr>(
            gko::kernels::cuda::cusparse::create_mat_descr(),
            [exec](cusparseMatDescr_t descr) {
                auto guard = exec->get_scoped_device_id_guard();
                gko::kernels::cuda::cusparse::destroy(descr);
            });
    }

private:
    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T*)>>;
    handle_manager<cusparseMatDescr> descr_;
};


#if CUDA_VERSION < 11000


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CusparseCsrmp
    : public gko::EnableLinOp<CusparseCsrmp<ValueType, IndexType>,
                              CusparseBase>,
      public gko::ReadableFromMatrixData<ValueType, IndexType>,
      public gko::EnableCreateMethod<CusparseCsrmp<ValueType, IndexType>> {
    friend class gko::EnableCreateMethod<CusparseCsrmp>;
    friend class gko::EnablePolymorphicObject<CusparseCsrmp, CusparseBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    using device_mat_data = gko::device_matrix_data<ValueType, IndexType>;

    void read(const device_mat_data& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(device_mat_data&& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(const mat_data& data) override
    {
        csr_->read(data);
        this->set_size(csr_->get_size());
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
        gko::kernels::cuda::cusparse::spmv_mp(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &scalars.get_const_data()[0],
            this->get_descr(), csr_->get_const_values(),
            csr_->get_const_row_ptrs(), csr_->get_const_col_idxs(), db,
            &scalars.get_const_data()[1], dx);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta,
                    gko::LinOp* x) const override GKO_NOT_IMPLEMENTED;

    CusparseCsrmp(std::shared_ptr<const gko::Executor> exec,
                  const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<CusparseCsrmp, CusparseBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    cusparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CusparseCsr
    : public gko::EnableLinOp<CusparseCsr<ValueType, IndexType>, CusparseBase>,
      public gko::EnableCreateMethod<CusparseCsr<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CusparseCsr>;
    friend class gko::EnablePolymorphicObject<CusparseCsr, CusparseBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    using device_mat_data = gko::device_matrix_data<ValueType, IndexType>;

    void read(const device_mat_data& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(device_mat_data&& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(const mat_data& data) override
    {
        csr_->read(data);
        this->set_size(csr_->get_size());
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
        gko::kernels::cuda::cusparse::spmv(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &scalars.get_const_data()[0],
            this->get_descr(), csr_->get_const_values(),
            csr_->get_const_row_ptrs(), csr_->get_const_col_idxs(), db,
            &scalars.get_const_data()[1], dx);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta,
                    gko::LinOp* x) const override GKO_NOT_IMPLEMENTED;

    CusparseCsr(std::shared_ptr<const gko::Executor> exec,
                const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<CusparseCsr, CusparseBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    cusparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CusparseCsrmm
    : public gko::EnableLinOp<CusparseCsrmm<ValueType, IndexType>,
                              CusparseBase>,
      public gko::EnableCreateMethod<CusparseCsrmm<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CusparseCsrmm>;
    friend class gko::EnablePolymorphicObject<CusparseCsrmm, CusparseBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    using device_mat_data = gko::device_matrix_data<ValueType, IndexType>;

    void read(const device_mat_data& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(device_mat_data&& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(const mat_data& data) override
    {
        csr_->read(data);
        this->set_size(csr_->get_size());
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
        gko::kernels::cuda::cusparse::spmm(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            this->get_size()[0], dense_b->get_size()[1], this->get_size()[1],
            csr_->get_num_stored_elements(), &scalars.get_const_data()[0],
            this->get_descr(), csr_->get_const_values(),
            csr_->get_const_row_ptrs(), csr_->get_const_col_idxs(), db,
            dense_b->get_size()[0], &scalars.get_const_data()[1], dx,
            dense_x->get_size()[0]);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta,
                    gko::LinOp* x) const override GKO_NOT_IMPLEMENTED;

    CusparseCsrmm(std::shared_ptr<const gko::Executor> exec,
                  const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<CusparseCsrmm, CusparseBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    cusparseOperation_t trans_;
};


#endif  // CUDA_VERSION < 11000


#if CUDA_VERSION < 11021


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CusparseCsrEx
    : public gko::EnableLinOp<CusparseCsrEx<ValueType, IndexType>,
                              CusparseBase>,
      public gko::EnableCreateMethod<CusparseCsrEx<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CusparseCsrEx>;
    friend class gko::EnablePolymorphicObject<CusparseCsrEx, CusparseBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    using device_mat_data = gko::device_matrix_data<ValueType, IndexType>;

    void read(const device_mat_data& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(device_mat_data&& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(const mat_data& data) override
    {
        csr_->read(data);
        this->set_size(csr_->get_size());
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

    CusparseCsrEx(const CusparseCsrEx& other) = delete;

    CusparseCsrEx& operator=(const CusparseCsrEx& other) = default;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        ValueType alpha = gko::one<ValueType>();
        ValueType beta = gko::zero<ValueType>();
        gko::size_type buffer_size = 0;

        auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
        auto handle = this->get_gpu_exec()->get_cusparse_handle();
        // This function seems to require the pointer mode to be set to HOST.
        // Ginkgo use pointer mode DEVICE by default, so we change this
        // temporarily.
        gko::kernels::cuda::cusparse::pointer_mode_guard pm_guard(handle);
        gko::kernels::cuda::cusparse::spmv_buffersize<ValueType, IndexType>(
            handle, algmode_, trans_, this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &alpha, this->get_descr(),
            csr_->get_const_values(), csr_->get_const_row_ptrs(),
            csr_->get_const_col_idxs(), db, &beta, dx, &buffer_size);
        buffer_.resize_and_reset(buffer_size);

        gko::kernels::cuda::cusparse::spmv<ValueType, IndexType>(
            handle, algmode_, trans_, this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &alpha, this->get_descr(),
            csr_->get_const_values(), csr_->get_const_row_ptrs(),
            csr_->get_const_col_idxs(), db, &beta, dx, buffer_.get_data());

        // Exiting the scope sets the pointer mode back to the default
        // DEVICE for Ginkgo
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta,
                    gko::LinOp* x) const override GKO_NOT_IMPLEMENTED;

    CusparseCsrEx(std::shared_ptr<const gko::Executor> exec,
                  const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<CusparseCsrEx, CusparseBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE),
          buffer_(exec)
    {
        algmode_ = CUSPARSE_ALG_MERGE_PATH;
    }

private:
    std::shared_ptr<csr> csr_;
    cusparseOperation_t trans_;
    cusparseAlgMode_t algmode_;
    mutable gko::array<char> buffer_;
};


#endif  // CUDA_VERSION < 11021


#if CUDA_VERSION < 11000


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32,
          cusparseHybPartition_t Partition = CUSPARSE_HYB_PARTITION_AUTO,
          int Threshold = 0>
class CusparseHybrid
    : public gko::EnableLinOp<
          CusparseHybrid<ValueType, IndexType, Partition, Threshold>,
          CusparseBase>,
      public gko::EnableCreateMethod<
          CusparseHybrid<ValueType, IndexType, Partition, Threshold>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CusparseHybrid>;
    friend class gko::EnablePolymorphicObject<CusparseHybrid, CusparseBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    using device_mat_data = gko::device_matrix_data<ValueType, IndexType>;

    void read(const device_mat_data& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(device_mat_data&& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(const mat_data& data) override
    {
        auto t_csr = csr::create(this->get_executor(),
                                 std::make_shared<typename csr::classical>());
        t_csr->read(data);
        this->set_size(t_csr->get_size());

        auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
        gko::kernels::cuda::cusparse::csr2hyb(
            this->get_gpu_exec()->get_cusparse_handle(), this->get_size()[0],
            this->get_size()[1], this->get_descr(), t_csr->get_const_values(),
            t_csr->get_const_row_ptrs(), t_csr->get_const_col_idxs(), hyb_,
            Threshold, Partition);
    }

    ~CusparseHybrid() override
    {
        try {
            auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
            GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyHybMat(hyb_));
        } catch (const std::exception& e) {
            std::cerr << "Error when unallocating CusparseHybrid hyb_ matrix: "
                      << e.what() << std::endl;
        }
    }

    CusparseHybrid(const CusparseHybrid& other) = delete;

    CusparseHybrid& operator=(const CusparseHybrid& other) = default;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
        gko::kernels::cuda::cusparse::spmv(
            this->get_gpu_exec()->get_cusparse_handle(), trans_,
            &scalars.get_const_data()[0], this->get_descr(), hyb_, db,
            &scalars.get_const_data()[1], dx);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta,
                    gko::LinOp* x) const override GKO_NOT_IMPLEMENTED;

    CusparseHybrid(std::shared_ptr<const gko::Executor> exec,
                   const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<CusparseHybrid, CusparseBase>(exec, size),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateHybMat(&hyb_));
    }

private:
    // Contains {alpha, beta}
    gko::array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    cusparseOperation_t trans_;
    cusparseHybMat_t hyb_;
};


#endif  // CUDA_VERSION < 11000


#if CUDA_VERSION >= 11000 || \
    ((CUDA_VERSION >= 10020) && !(defined(_WIN32) || defined(__CYGWIN__)))


template <typename ValueType>
void cusparse_generic_spmv(std::shared_ptr<const gko::CudaExecutor> gpu_exec,
                           const cusparseSpMatDescr_t mat,
                           const gko::array<ValueType>& scalars,
                           const gko::LinOp* b, gko::LinOp* x,
                           cusparseOperation_t trans, cusparseSpMVAlg_t alg)
{
    cudaDataType_t cu_value = gko::kernels::cuda::cuda_data_type<ValueType>();
    using gko::kernels::cuda::as_culibs_type;
    auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
    auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
    auto db = dense_b->get_const_values();
    auto dx = dense_x->get_values();
    auto guard = gpu_exec->get_scoped_device_id_guard();
    cusparseDnVecDescr_t vecb, vecx;
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseCreateDnVec(&vecx, dense_x->get_num_stored_elements(),
                            as_culibs_type(dx), cu_value));
    // cusparseCreateDnVec only allows non-const pointer
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateDnVec(
        &vecb, dense_b->get_num_stored_elements(),
        as_culibs_type(const_cast<ValueType*>(db)), cu_value));

    gko::size_type buffer_size = 0;
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpMV_bufferSize(
        gpu_exec->get_cusparse_handle(), trans, &scalars.get_const_data()[0],
        mat, vecb, &scalars.get_const_data()[1], vecx, cu_value, alg,
        &buffer_size));
    gko::array<char> buffer_array(gpu_exec, buffer_size);
    auto dbuffer = buffer_array.get_data();
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpMV(
        gpu_exec->get_cusparse_handle(), trans, &scalars.get_const_data()[0],
        mat, vecb, &scalars.get_const_data()[1], vecx, cu_value, alg, dbuffer));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyDnVec(vecx));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyDnVec(vecb));
}

#if CUDA_VERSION < 11021
constexpr auto default_csr_alg = CUSPARSE_MV_ALG_DEFAULT;
#else
constexpr auto default_csr_alg = CUSPARSE_SPMV_ALG_DEFAULT;
#endif

template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32,
          cusparseSpMVAlg_t Alg = default_csr_alg>
class CusparseGenericCsr
    : public gko::EnableLinOp<CusparseGenericCsr<ValueType, IndexType, Alg>,
                              CusparseBase>,
      public gko::EnableCreateMethod<
          CusparseGenericCsr<ValueType, IndexType, Alg>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CusparseGenericCsr>;
    friend class gko::EnablePolymorphicObject<CusparseGenericCsr, CusparseBase>;

public:
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    using device_mat_data = gko::device_matrix_data<ValueType, IndexType>;
    cusparseIndexType_t cu_index =
        gko::kernels::cuda::cusparse_index_type<IndexType>();
    cudaDataType_t cu_value = gko::kernels::cuda::cuda_data_type<ValueType>();

    void read(const device_mat_data& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(device_mat_data&& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(const mat_data& data) override
    {
        using gko::kernels::cuda::as_culibs_type;
        csr_->read(data);
        this->set_size(csr_->get_size());
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseCreateCsr(&mat_, csr_->get_size()[0], csr_->get_size()[1],
                              csr_->get_num_stored_elements(),
                              as_culibs_type(csr_->get_row_ptrs()),
                              as_culibs_type(csr_->get_col_idxs()),
                              as_culibs_type(csr_->get_values()), cu_index,
                              cu_index, CUSPARSE_INDEX_BASE_ZERO, cu_value));
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

    ~CusparseGenericCsr() override
    {
        try {
            auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
            GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroySpMat(mat_));
        } catch (const std::exception& e) {
            std::cerr
                << "Error when unallocating CusparseGenericCsr mat_ matrix: "
                << e.what() << std::endl;
        }
    }

    CusparseGenericCsr(const CusparseGenericCsr& other) = delete;

    CusparseGenericCsr& operator=(const CusparseGenericCsr& other) = default;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        cusparse_generic_spmv(this->get_gpu_exec(), mat_, scalars, b, x, trans_,
                              Alg);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta,
                    gko::LinOp* x) const override GKO_NOT_IMPLEMENTED;

    CusparseGenericCsr(std::shared_ptr<const gko::Executor> exec,
                       const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<CusparseGenericCsr, CusparseBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    cusparseOperation_t trans_;
    cusparseSpMatDescr_t mat_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class CusparseGenericCoo
    : public gko::EnableLinOp<CusparseGenericCoo<ValueType, IndexType>,
                              CusparseBase>,
      public gko::EnableCreateMethod<CusparseGenericCoo<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<CusparseGenericCoo>;
    friend class gko::EnablePolymorphicObject<CusparseGenericCoo, CusparseBase>;

public:
    using coo = gko::matrix::Coo<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    using device_mat_data = gko::device_matrix_data<ValueType, IndexType>;
    cusparseIndexType_t cu_index =
        gko::kernels::cuda::cusparse_index_type<IndexType>();
    cudaDataType_t cu_value = gko::kernels::cuda::cuda_data_type<ValueType>();

    void read(const device_mat_data& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(device_mat_data&& data) override
    {
        this->read(data.copy_to_host());
    }

    void read(const mat_data& data) override
    {
        using gko::kernels::cuda::as_culibs_type;
        coo_->read(data);
        this->set_size(coo_->get_size());
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseCreateCoo(&mat_, coo_->get_size()[0], coo_->get_size()[1],
                              coo_->get_num_stored_elements(),
                              as_culibs_type(coo_->get_row_idxs()),
                              as_culibs_type(coo_->get_col_idxs()),
                              as_culibs_type(coo_->get_values()), cu_index,
                              CUSPARSE_INDEX_BASE_ZERO, cu_value));
    }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return coo_->get_num_stored_elements();
    }

    ~CusparseGenericCoo() override
    {
        try {
            auto guard = this->get_gpu_exec()->get_scoped_device_id_guard();
            GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroySpMat(mat_));
        } catch (const std::exception& e) {
            std::cerr
                << "Error when unallocating CusparseGenericCoo mat_ matrix: "
                << e.what() << std::endl;
        }
    }

    CusparseGenericCoo(const CusparseGenericCoo& other) = delete;

    CusparseGenericCoo& operator=(const CusparseGenericCoo& other) = default;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        cusparse_generic_spmv(this->get_gpu_exec(), mat_, scalars, b, x, trans_,
                              default_csr_alg);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta,
                    gko::LinOp* x) const override GKO_NOT_IMPLEMENTED;

    CusparseGenericCoo(std::shared_ptr<const gko::Executor> exec,
                       const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<CusparseGenericCoo, CusparseBase>(exec, size),
          coo_(std::move(coo::create(exec))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<coo> coo_;
    cusparseOperation_t trans_;
    cusparseSpMatDescr_t mat_;
};


#endif  // CUDA_VERSION >= 11000 || ((CUDA_VERSION >= 10020) &&
        // !(defined(_WIN32) || defined(__CYGWIN__)))


}  // namespace detail


#if CUDA_VERSION < 11021
IMPL_CREATE_SPARSELIB_LINOP(cusparse_csrex,
                            detail::CusparseCsrEx<etype, itype>);
#else
STUB_CREATE_SPARSELIB_LINOP(cusparse_csrex);
#endif

#if CUDA_VERSION < 11000
IMPL_CREATE_SPARSELIB_LINOP(cusparse_csr, detail::CusparseCsr<etype, itype>);
IMPL_CREATE_SPARSELIB_LINOP(cusparse_csrmp,
                            detail::CusparseCsrmp<etype, itype>);
IMPL_CREATE_SPARSELIB_LINOP(cusparse_csrmm,
                            detail::CusparseCsrmm<etype, itype>);
#else   // CUDA_VERSION >= 11000
IMPL_CREATE_SPARSELIB_LINOP(cusparse_csr,
                            detail::CusparseGenericCsr<etype, itype>);
STUB_CREATE_SPARSELIB_LINOP(cusparse_csrmp);
STUB_CREATE_SPARSELIB_LINOP(cusparse_csrmm);
#endif  // CUDA_VERSION >= 11000


#if CUDA_VERSION >= 11000 || \
    ((CUDA_VERSION >= 10020) && !(defined(_WIN32) || defined(__CYGWIN__)))
IMPL_CREATE_SPARSELIB_LINOP(cusparse_gcsr,
                            detail::CusparseGenericCsr<etype, itype>);
#if CUDA_VERSION >= 11021
constexpr auto csr_algo = CUSPARSE_SPMV_CSR_ALG2;
#else
constexpr auto csr_algo = CUSPARSE_CSRMV_ALG2;
#endif
IMPL_CREATE_SPARSELIB_LINOP(cusparse_gcsr2,
                            detail::CusparseGenericCsr<etype, itype, csr_algo>);
IMPL_CREATE_SPARSELIB_LINOP(cusparse_gcoo,
                            detail::CusparseGenericCoo<etype, itype>);
#else
STUB_CREATE_SPARSELIB_LINOP(cusparse_gcsr);
STUB_CREATE_SPARSELIB_LINOP(cusparse_gcsr2);
STUB_CREATE_SPARSELIB_LINOP(cusparse_gcoo);
#endif  // CUDA_VERSION < 11000 && ((CUDA_VERSION < 10020) || (defined(_WIN32)
        // && defined(__CYGWIN__))))


#if CUDA_VERSION < 11000
IMPL_CREATE_SPARSELIB_LINOP(
    cusparse_coo,
    detail::CusparseHybrid<etype, itype, CUSPARSE_HYB_PARTITION_USER, 0>);
IMPL_CREATE_SPARSELIB_LINOP(
    cusparse_ell,
    detail::CusparseHybrid<etype, itype, CUSPARSE_HYB_PARTITION_MAX, 0>);
IMPL_CREATE_SPARSELIB_LINOP(cusparse_hybrid,
                            detail::CusparseHybrid<etype, itype>);
#else   // CUDA_VERSION >= 11000
IMPL_CREATE_SPARSELIB_LINOP(cusparse_coo,
                            detail::CusparseGenericCoo<etype, itype>);
STUB_CREATE_SPARSELIB_LINOP(cusparse_ell);
STUB_CREATE_SPARSELIB_LINOP(cusparse_hybrid);
#endif  // CUDA_VERSION >= 11000
