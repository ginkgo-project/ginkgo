// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace gko {

struct reference_kernel {};
struct omp_kernel {};
struct cuda_kernel {};
struct hip_kernel {};
struct sycl_kernel {};

}  // namespace gko
