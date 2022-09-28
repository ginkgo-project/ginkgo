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

#include <ginkgo/ginkgo.hpp>


#include <memory>


#include <hipsparse.h>


#include "benchmark/utils/sparselib_linops.hpp"
#include "benchmark/utils/types.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"


class hipsparse_csr {};
class hipsparse_csrmm {};
class hipsparse_hybrid {};
class hipsparse_coo {};
class hipsparse_ell {};


namespace detail {


struct hipsparseMatDescr;


class HipsparseBase : public gko::LinOp {
public:
    hipsparseMatDescr_t get_descr() const { return this->descr_.get(); }

    std::shared_ptr<const gko::HipExecutor> get_gpu_exec() const
    {
        return std::dynamic_pointer_cast<const gko::HipExecutor>(
            this->get_executor());
    }

protected:
    HipsparseBase(std::shared_ptr<const gko::Executor> exec,
                  const gko::dim<2>& size = gko::dim<2>{})
        : gko::LinOp(exec, size)
    {
        if (this->get_gpu_exec() == nullptr) {
            GKO_NOT_IMPLEMENTED;
        }
        this->initialize_descr();
    }

    ~HipsparseBase() = default;

    HipsparseBase(const HipsparseBase& other) = delete;

    HipsparseBase& operator=(const HipsparseBase& other)
    {
        if (this != &other) {
            gko::LinOp::operator=(other);
            this->initialize_descr();
        }
        return *this;
    }

    void initialize_descr()
    {
        const auto id = this->get_gpu_exec()->get_device_id();
        gko::detail::hip_scoped_device_id g{id};
        this->descr_ = handle_manager<hipsparseMatDescr>(
            reinterpret_cast<hipsparseMatDescr*>(
                gko::kernels::hip::hipsparse::create_mat_descr()),
            [id](hipsparseMatDescr* descr) {
                gko::detail::hip_scoped_device_id g{id};
                gko::kernels::hip::hipsparse::destroy(descr);
            });
    }

private:
    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T*)>>;
    handle_manager<hipsparseMatDescr> descr_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class HipsparseCsr
    : public gko::EnableLinOp<HipsparseCsr<ValueType, IndexType>,
                              HipsparseBase>,
      public gko::EnableCreateMethod<HipsparseCsr<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<HipsparseCsr>;
    friend class gko::polymorphic_object_traits<HipsparseCsr>;

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

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::detail::hip_scoped_device_id g{id};
        gko::kernels::hip::hipsparse::spmv(
            this->get_gpu_exec()->get_hipsparse_handle(), trans_,
            this->get_size()[0], this->get_size()[1],
            csr_->get_num_stored_elements(), &scalars.get_const_data()[0],
            this->get_descr(), csr_->get_const_values(),
            csr_->get_const_row_ptrs(), csr_->get_const_col_idxs(), db,
            &scalars.get_const_data()[1], dx);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta,
                    gko::LinOp* x) const override GKO_NOT_IMPLEMENTED;

    HipsparseCsr(std::shared_ptr<const gko::Executor> exec,
                 const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<HipsparseCsr, HipsparseBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    hipsparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class HipsparseCsrmm
    : public gko::EnableLinOp<HipsparseCsrmm<ValueType, IndexType>,
                              HipsparseBase>,
      public gko::EnableCreateMethod<HipsparseCsrmm<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<HipsparseCsrmm>;
    friend class gko::polymorphic_object_traits<HipsparseCsrmm>;

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

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::detail::hip_scoped_device_id g{id};
        gko::kernels::hip::hipsparse::spmm(
            this->get_gpu_exec()->get_hipsparse_handle(), trans_,
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

    HipsparseCsrmm(std::shared_ptr<const gko::Executor> exec,
                   const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<HipsparseCsrmm, HipsparseBase>(exec, size),
          csr_(std::move(
              csr::create(exec, std::make_shared<typename csr::classical>()))),
          trans_(HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {}

private:
    // Contains {alpha, beta}
    gko::array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    std::shared_ptr<csr> csr_;
    hipsparseOperation_t trans_;
};


template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32,
          hipsparseHybPartition_t Partition = HIPSPARSE_HYB_PARTITION_AUTO,
          int Threshold = 0>
class HipsparseHybrid
    : public gko::EnableLinOp<
          HipsparseHybrid<ValueType, IndexType, Partition, Threshold>,
          HipsparseBase>,
      public gko::EnableCreateMethod<
          HipsparseHybrid<ValueType, IndexType, Partition, Threshold>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<HipsparseHybrid>;
    friend class gko::polymorphic_object_traits<HipsparseHybrid>;

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

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::detail::hip_scoped_device_id g{id};
        gko::kernels::hip::hipsparse::csr2hyb(
            this->get_gpu_exec()->get_hipsparse_handle(), this->get_size()[0],
            this->get_size()[1], this->get_descr(), t_csr->get_const_values(),
            t_csr->get_const_row_ptrs(), t_csr->get_const_col_idxs(), hyb_,
            Threshold, Partition);
    }

    ~HipsparseHybrid() override
    {
        const auto id = this->get_gpu_exec()->get_device_id();
        try {
            gko::detail::hip_scoped_device_id g{id};
            GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseDestroyHybMat(hyb_));
        } catch (const std::exception& e) {
            std::cerr << "Error when unallocating HipsparseHybrid hyb_ matrix: "
                      << e.what() << std::endl;
        }
    }

    HipsparseHybrid(const HipsparseHybrid& other) = delete;

    HipsparseHybrid& operator=(const HipsparseHybrid& other) = default;

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();

        const auto id = this->get_gpu_exec()->get_device_id();
        gko::detail::hip_scoped_device_id g{id};
        gko::kernels::hip::hipsparse::spmv(
            this->get_gpu_exec()->get_hipsparse_handle(), trans_,
            &scalars.get_const_data()[0], this->get_descr(), hyb_, db,
            &scalars.get_const_data()[1], dx);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta,
                    gko::LinOp* x) const override GKO_NOT_IMPLEMENTED;

    HipsparseHybrid(std::shared_ptr<const gko::Executor> exec,
                    const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<HipsparseHybrid, HipsparseBase>(exec, size),
          trans_(HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        const auto id = this->get_gpu_exec()->get_device_id();
        gko::detail::hip_scoped_device_id g{id};
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreateHybMat(&hyb_));
    }

private:
    // Contains {alpha, beta}
    gko::array<ValueType> scalars{
        this->get_executor(), {gko::one<ValueType>(), gko::zero<ValueType>()}};
    hipsparseOperation_t trans_;
    hipsparseHybMat_t hyb_;
};


}  // namespace detail


IMPL_CREATE_SPARSELIB_LINOP(hipsparse_csr, detail::HipsparseCsr<etype, itype>);
IMPL_CREATE_SPARSELIB_LINOP(hipsparse_csrmm,
                            detail::HipsparseCsrmm<etype, itype>);
IMPL_CREATE_SPARSELIB_LINOP(
    hipsparse_coo,
    detail::HipsparseHybrid<etype, itype, HIPSPARSE_HYB_PARTITION_USER, 0>);
IMPL_CREATE_SPARSELIB_LINOP(
    hipsparse_ell,
    detail::HipsparseHybrid<etype, itype, HIPSPARSE_HYB_PARTITION_MAX, 0>);
IMPL_CREATE_SPARSELIB_LINOP(hipsparse_hybrid,
                            detail::HipsparseHybrid<etype, itype>);
