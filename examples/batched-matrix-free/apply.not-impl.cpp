// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "apply.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>


gko::batch::matrix::external_apply::advanced_type get_gpu_advanced_apply_ptr()
{
    return nullptr;
}


gko::batch::matrix::external_apply::simple_type get_gpu_simple_apply_ptr()
{
    return nullptr;
}
