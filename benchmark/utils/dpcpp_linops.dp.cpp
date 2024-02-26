// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <memory>


#include <oneapi/mkl.hpp>


#include "benchmark/utils/sparselib_linops.hpp"
#include "benchmark/utils/types.hpp"


class onemkl_csr {};
class onemkl_optimized_csr {};


namespace detail {


inline oneapi::mkl::sparse::matrix_handle_t create_mat_handle()
{
    oneapi::mkl::sparse::matrix_handle_t mat_handle;
    oneapi::mkl::sparse::init_matrix_handle(&mat_handle);
    return mat_handle;
}


class OnemklBase : public gko::LinOp {
public:
    oneapi::mkl::sparse::matrix_handle_t get_mat_handle() const
    {
        return this->mat_handle_.get();
    }

    std::shared_ptr<const gko::DpcppExecutor> get_device_exec() const
    {
        return std::dynamic_pointer_cast<const gko::DpcppExecutor>(
            this->get_executor());
    }

protected:
    void apply_impl(const gko::LinOp*, const gko::LinOp*, const gko::LinOp*,
                    gko::LinOp*) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    void initialize_mat_handle()
    {
        mat_handle_ = handle_manager<oneapi::mkl::sparse::matrix_handle>(
            create_mat_handle(),
            [](oneapi::mkl::sparse::matrix_handle_t mat_handle) {
                oneapi::mkl::sparse::release_matrix_handle(&mat_handle);
            });
    }

    OnemklBase(std::shared_ptr<const gko::Executor> exec,
               const gko::dim<2>& size = gko::dim<2>{})
        : gko::LinOp(exec, size)
    {
        if (this->get_device_exec() == nullptr) {
            GKO_NOT_IMPLEMENTED;
        }
        this->initialize_mat_handle();
    }

    ~OnemklBase() = default;

    OnemklBase(const OnemklBase& other) = delete;

    OnemklBase& operator=(const OnemklBase& other)
    {
        if (this != &other) {
            gko::LinOp::operator=(other);
            this->initialize_mat_handle();
        }
        return *this;
    }

private:
    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T*)>>;
    handle_manager<oneapi::mkl::sparse::matrix_handle> mat_handle_;
};


template <bool optimized = false, typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class OnemklCsr
    : public gko::EnableLinOp<OnemklCsr<optimized, ValueType, IndexType>,
                              OnemklBase>,
      public gko::EnableCreateMethod<
          OnemklCsr<optimized, ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::EnableCreateMethod<OnemklCsr>;
    friend class gko::EnablePolymorphicObject<OnemklCsr, OnemklBase>;

public:
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using mat_data = gko::matrix_data<ValueType, IndexType>;
    void read(const mat_data& data) override
    {
        csr_->read(data);
        this->set_size(csr_->get_size());

        oneapi::mkl::sparse::set_csr_data(
            this->get_mat_handle(), static_cast<int>(this->get_size()[0]),
            static_cast<int>(this->get_size()[1]),
            oneapi::mkl::index_base::zero, csr_->get_row_ptrs(),
            csr_->get_col_idxs(), csr_->get_values());
        if (optimized) {
            // need the last argument {} to make sure that it uses USM version.
            oneapi::mkl::sparse::optimize_gemv(
                *(this->get_device_exec()->get_queue()), trans_,
                this->get_mat_handle(), {});
        }
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

        oneapi::mkl::sparse::gemv(
            *(this->get_device_exec()->get_queue()), trans_,
            gko::one<ValueType>(), this->get_mat_handle(),
            const_cast<ValueType*>(db), gko::zero<ValueType>(), dx);
    }

    OnemklCsr(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<OnemklCsr, OnemklBase>(exec, size),
          csr_(std::move(
              Csr::create(exec, std::make_shared<typename Csr::classical>()))),
          trans_(oneapi::mkl::transpose::nontrans)
    {}


private:
    oneapi::mkl::transpose trans_;
    std::shared_ptr<Csr> csr_;
};


}  // namespace detail


IMPL_CREATE_SPARSELIB_LINOP(onemkl_csr, detail::OnemklCsr<false, etype, itype>);
IMPL_CREATE_SPARSELIB_LINOP(onemkl_optimized_csr,
                            detail::OnemklCsr<true, etype, itype>);
