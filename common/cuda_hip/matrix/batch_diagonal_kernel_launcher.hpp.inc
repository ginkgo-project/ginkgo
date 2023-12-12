// Copyright (c) 2017-2023, the Ginkgo authors
// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const batch::matrix::Diagonal<ValueType>* mat,
                  const batch::MultiVector<ValueType>* b,
                  batch::MultiVector<ValueType>* x)
{
    const auto num_blocks = mat->get_num_batch_items();
    simple_apply_kernel<<<num_blocks, default_block_size, 0,
                          exec->get_stream()>>>(
        num_blocks, static_cast<int>(mat->get_common_size()[0]),
        as_device_type(mat->get_const_values()),
        static_cast<int>(b->get_common_size()[1]), b->get_common_size()[1],
        as_device_type(b->get_const_values()), x->get_common_size()[1],
        as_device_type(x->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIAGONAL_SIMPLE_APPLY_KERNEL);
