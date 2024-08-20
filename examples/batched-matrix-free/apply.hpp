// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_external.hpp>


template <typename ValueType>
void advanced_apply_generic(gko::size_type id, gko::dim<2> size,
                            const void* alpha, const void* b, const void* beta,
                            void* x, void* payload);

template <typename ValueType>
void simple_apply_generic(gko::size_type id, gko::dim<2> size, const void* b,
                          void* x, void* payload);


gko::batch::matrix::external_apply::advanced_type get_gpu_advanced_apply_ptr();

gko::batch::matrix::external_apply::simple_type get_gpu_simple_apply_ptr();
