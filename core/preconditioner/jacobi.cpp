// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/jacobi.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"


namespace gko {
namespace preconditioner {
namespace jacobi {
namespace {


GKO_REGISTER_OPERATION(simple_apply, jacobi::simple_apply);
GKO_REGISTER_OPERATION(simple_scalar_apply, jacobi::simple_scalar_apply);
GKO_REGISTER_OPERATION(apply, jacobi::apply);
GKO_REGISTER_OPERATION(scalar_apply, jacobi::scalar_apply);
GKO_REGISTER_OPERATION(find_blocks, jacobi::find_blocks);
GKO_REGISTER_OPERATION(generate, jacobi::generate);
GKO_REGISTER_OPERATION(scalar_conj, jacobi::scalar_conj);
GKO_REGISTER_OPERATION(invert_diagonal, jacobi::invert_diagonal);
GKO_REGISTER_OPERATION(transpose_jacobi, jacobi::transpose_jacobi);
GKO_REGISTER_OPERATION(conj_transpose_jacobi, jacobi::conj_transpose_jacobi);
GKO_REGISTER_OPERATION(convert_to_dense, jacobi::convert_to_dense);
GKO_REGISTER_OPERATION(scalar_convert_to_dense,
                       jacobi::scalar_convert_to_dense);
GKO_REGISTER_OPERATION(initialize_precisions, jacobi::initialize_precisions);


}  // anonymous namespace
}  // namespace jacobi


template <typename ValueType, typename IndexType>
Jacobi<ValueType, IndexType>& Jacobi<ValueType, IndexType>::operator=(
    const Jacobi& other)
{
    if (&other != this) {
        EnableLinOp<Jacobi>::operator=(other);
        storage_scheme_ = other.storage_scheme_;
        num_blocks_ = other.num_blocks_;
        blocks_ = other.blocks_;
        conditioning_ = other.conditioning_;
        parameters_ = other.parameters_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Jacobi<ValueType, IndexType>& Jacobi<ValueType, IndexType>::operator=(
    Jacobi&& other)
{
    if (&other != this) {
        EnableLinOp<Jacobi>::operator=(std::move(other));
        // reset size values to 0 in other
        storage_scheme_ =
            std::exchange(other.storage_scheme_,
                          block_interleaved_storage_scheme<index_type>{});
        num_blocks_ = std::exchange(other.num_blocks_, 0);
        blocks_ = std::move(other.blocks_);
        conditioning_ = std::move(other.conditioning_);
        parameters_ = std::exchange(other.parameters_, parameters_type{});
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Jacobi<ValueType, IndexType>::Jacobi(const Jacobi& other)
    : Jacobi{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Jacobi<ValueType, IndexType>::Jacobi(Jacobi&& other)
    : Jacobi{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            if (parameters_.max_block_size == 1) {
                this->get_executor()->run(jacobi::make_simple_scalar_apply(
                    this->blocks_, dense_b, dense_x));
            } else {
                this->get_executor()->run(jacobi::make_simple_apply(
                    num_blocks_, parameters_.max_block_size, storage_scheme_,
                    parameters_.storage_optimization.block_wise,
                    parameters_.block_pointers, blocks_, dense_b, dense_x));
            }
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                              const LinOp* b, const LinOp* beta,
                                              LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            if (parameters_.max_block_size == 1) {
                this->get_executor()->run(jacobi::make_scalar_apply(
                    this->blocks_, dense_alpha, dense_b, dense_beta, dense_x));
            } else {
                this->get_executor()->run(jacobi::make_apply(
                    num_blocks_, parameters_.max_block_size, storage_scheme_,
                    parameters_.storage_optimization.block_wise,
                    parameters_.block_pointers, blocks_, dense_alpha, dense_b,
                    dense_beta, dense_x));
            }
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::convert_to(
    matrix::Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp = matrix::Dense<ValueType>::create(exec, this->get_size());
    if (parameters_.max_block_size == 1) {
        exec->run(jacobi::make_scalar_convert_to_dense(blocks_, tmp.get()));
    } else {
        exec->run(jacobi::make_convert_to_dense(
            num_blocks_, parameters_.storage_optimization.block_wise,
            parameters_.block_pointers, blocks_, storage_scheme_,
            tmp->get_values(), tmp->get_stride()));
    }
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::move_to(matrix::Dense<ValueType>* result)
{
    this->convert_to(result);  // no special optimization possible here
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::write(mat_data& data) const
{
    auto local_clone =
        make_temporary_clone(this->get_executor()->get_master(), this);
    data = {local_clone->get_size(), {}};

    if (parameters_.max_block_size == 1) {
        for (IndexType row = 0; row < data.size[0]; ++row) {
            data.nonzeros.emplace_back(
                row, row,
                static_cast<ValueType>(local_clone->get_blocks()[row]));
        }
    } else {
        const auto ptrs =
            local_clone->parameters_.block_pointers.get_const_data();
        for (size_type block = 0; block < local_clone->get_num_blocks();
             ++block) {
            const auto scheme = local_clone->get_storage_scheme();
            const auto group_data = local_clone->blocks_.get_const_data() +
                                    scheme.get_group_offset(block);
            const auto block_size = ptrs[block + 1] - ptrs[block];
            const auto precisions =
                local_clone->parameters_.storage_optimization.block_wise
                    .get_const_data();
            const auto prec =
                precisions ? precisions[block] : precision_reduction();
            GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(ValueType, prec, {
                const auto block_data =
                    reinterpret_cast<const resolved_precision*>(group_data) +
                    scheme.get_block_offset(block);
                for (IndexType row = 0; row < block_size; ++row) {
                    for (IndexType col = 0; col < block_size; ++col) {
                        data.nonzeros.emplace_back(
                            ptrs[block] + row, ptrs[block] + col,
                            static_cast<ValueType>(
                                block_data[row + col * scheme.get_stride()]));
                    }
                }
            });
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Jacobi<ValueType, IndexType>::transpose() const
{
    auto res = std::unique_ptr<Jacobi<ValueType, IndexType>>(
        new Jacobi<ValueType, IndexType>(this->get_executor()));
    // Jacobi enforces square matrices, so no dim transposition necessary
    res->set_size(this->get_size());
    res->storage_scheme_ = storage_scheme_;
    res->num_blocks_ = num_blocks_;
    res->blocks_.resize_and_reset(blocks_.get_size());
    res->conditioning_ = conditioning_;
    res->parameters_ = parameters_;
    if (parameters_.max_block_size == 1) {
        res->blocks_ = blocks_;
    } else {
        this->get_executor()->run(jacobi::make_transpose_jacobi(
            num_blocks_, parameters_.max_block_size,
            parameters_.storage_optimization.block_wise,
            parameters_.block_pointers, blocks_, storage_scheme_,
            res->blocks_));
    }

    return std::move(res);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Jacobi<ValueType, IndexType>::conj_transpose() const
{
    auto res = std::unique_ptr<Jacobi<ValueType, IndexType>>(
        new Jacobi<ValueType, IndexType>(this->get_executor()));
    // Jacobi enforces square matrices, so no dim transposition necessary
    res->set_size(this->get_size());
    res->storage_scheme_ = storage_scheme_;
    res->num_blocks_ = num_blocks_;
    res->blocks_.resize_and_reset(blocks_.get_size());
    res->conditioning_ = conditioning_;
    res->parameters_ = parameters_;
    if (parameters_.max_block_size == 1) {
        this->get_executor()->run(
            jacobi::make_scalar_conj(this->blocks_, res->blocks_));
    } else {
        this->get_executor()->run(jacobi::make_conj_transpose_jacobi(
            num_blocks_, parameters_.max_block_size,
            parameters_.storage_optimization.block_wise,
            parameters_.block_pointers, blocks_, storage_scheme_,
            res->blocks_));
    }

    return std::move(res);
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::detect_blocks(
    const matrix::Csr<ValueType, IndexType>* system_matrix)
{
    parameters_.block_pointers.resize_and_reset(system_matrix->get_size()[0] +
                                                1);
    this->get_executor()->run(
        jacobi::make_find_blocks(system_matrix, parameters_.max_block_size,
                                 num_blocks_, parameters_.block_pointers));
    blocks_.resize_and_reset(
        storage_scheme_.compute_storage_space(num_blocks_));
}


template <typename ValueType, typename IndexType>
void Jacobi<ValueType, IndexType>::generate(const LinOp* system_matrix,
                                            bool skip_sorting)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    using csr_type = matrix::Csr<ValueType, IndexType>;
    const auto exec = this->get_executor();
    if (parameters_.max_block_size == 1) {
        auto diag = share(as<DiagonalLinOpExtractable>(system_matrix)
                              ->extract_diagonal_linop());
        auto diag_vt =
            ::gko::detail::temporary_conversion<matrix::Diagonal<ValueType>>::
                template create<matrix::Diagonal<next_precision<ValueType>>>(
                    diag.get());
        if (!diag_vt) {
            GKO_NOT_SUPPORTED(system_matrix);
        }
        auto temp =
            make_array_view(diag_vt->get_executor(), diag_vt->get_size()[0],
                            diag_vt->get_values());
        this->blocks_ = array<ValueType>(exec, temp.get_size());
        exec->run(jacobi::make_invert_diagonal(temp, this->blocks_));
        this->num_blocks_ = diag_vt->get_size()[0];
    } else {
        auto csr_mtx = convert_to_with_sorting<csr_type>(exec, system_matrix,
                                                         skip_sorting);
        if (parameters_.block_pointers.get_data() == nullptr) {
            this->detect_blocks(csr_mtx.get());
        }
        const auto all_block_opt =
            parameters_.storage_optimization.of_all_blocks;
        auto& precisions = parameters_.storage_optimization.block_wise;
        // if adaptive version is used, make sure that the precision array is of
        // the correct size by replicating it multiple times if needed
        if (parameters_.storage_optimization.is_block_wise ||
            all_block_opt != precision_reduction(0, 0)) {
            if (!parameters_.storage_optimization.is_block_wise) {
                precisions =
                    gko::array<precision_reduction>(exec, {all_block_opt});
            }
            array<precision_reduction> tmp(
                exec, parameters_.block_pointers.get_size() - 1);
            exec->run(jacobi::make_initialize_precisions(precisions, tmp));
            precisions = std::move(tmp);
            conditioning_.resize_and_reset(num_blocks_);
        }
        exec->run(jacobi::make_generate(
            csr_mtx.get(), num_blocks_, parameters_.max_block_size,
            parameters_.accuracy, storage_scheme_, conditioning_, precisions,
            parameters_.block_pointers, blocks_));
    }
}


#define GKO_DECLARE_JACOBI(ValueType, IndexType) \
    class Jacobi<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI);


}  // namespace preconditioner
}  // namespace gko
