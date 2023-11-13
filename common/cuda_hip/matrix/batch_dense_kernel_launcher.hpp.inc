// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const batch::matrix::Dense<ValueType>* mat,
                  const batch::MultiVector<ValueType>* b,
                  batch::MultiVector<ValueType>* x)
{
    const auto num_blocks = mat->get_num_batch_items();
    const auto b_ub = get_batch_struct(b);
    const auto x_ub = get_batch_struct(x);
    const auto mat_ub = get_batch_struct(mat);
    if (b->get_common_size()[1] > 1) {
        GKO_NOT_IMPLEMENTED;
    }
    simple_apply_kernel<<<num_blocks, default_block_size, 0,
                          exec->get_stream()>>>(mat_ub, b_ub, x_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void advanced_apply(std::shared_ptr<const DefaultExecutor> exec,
                    const batch::MultiVector<ValueType>* alpha,
                    const batch::matrix::Dense<ValueType>* mat,
                    const batch::MultiVector<ValueType>* b,
                    const batch::MultiVector<ValueType>* beta,
                    batch::MultiVector<ValueType>* x)
{
    const auto num_blocks = mat->get_num_batch_items();
    const auto b_ub = get_batch_struct(b);
    const auto x_ub = get_batch_struct(x);
    const auto mat_ub = get_batch_struct(mat);
    const auto alpha_ub = get_batch_struct(alpha);
    const auto beta_ub = get_batch_struct(beta);
    if (b->get_common_size()[1] > 1) {
        GKO_NOT_IMPLEMENTED;
    }
    advanced_apply_kernel<<<num_blocks, default_block_size, 0,
                            exec->get_stream()>>>(alpha_ub, mat_ub, b_ub,
                                                  beta_ub, x_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADVANCED_APPLY_KERNEL);
