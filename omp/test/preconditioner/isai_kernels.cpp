// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/isai_kernels.hpp"

#include <cmath>

#include <omp.h>

#include <gtest/gtest.h>

#include <sys/resource.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/test/utils.hpp"


namespace {


long peak_rss_kib()
{
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}


TEST(SpdIsai, ScratchDoesNotGrowWithThreadCount)
{
    using Csr = gko::matrix::Csr<double, gko::int32>;
    auto exec = gko::OmpExecutor::create();
    const gko::int32 n = 2000;
    // diagonal plus a dense last row and column: one row of length n
    gko::matrix_data<double, gko::int32> data{gko::dim<2>(n, n)};
    gko::matrix_data<double, gko::int32> lower{gko::dim<2>(n, n)};
    const auto off = 1.0 / std::sqrt(static_cast<double>(n));
    for (gko::int32 i = 0; i < n; ++i) {
        if (i < n - 1) {
            data.nonzeros.emplace_back(i, n - 1, off);
            data.nonzeros.emplace_back(n - 1, i, off);
            lower.nonzeros.emplace_back(n - 1, i, -1.0);
        }
        data.nonzeros.emplace_back(i, i, i == n - 1 ? 5.0 : 4.0);
        lower.nonzeros.emplace_back(i, i, -1.0);
    }
    data.sort_row_major();
    lower.sort_row_major();
    auto a = Csr::create(exec);
    a->read(data);
    auto l = Csr::create(exec);
    l->read(lower);
    omp_set_num_threads(8);
    const auto before = peak_rss_kib();

    gko::kernels::omp::isai::generate_spd_inverse(
        exec, a->get_const_device_view(), l->get_device_view(), 0);

    // one n x n scratch per thread would add 8 n^2 doubles
    const long one_scratch_kib = long{n} * n * sizeof(double) / 1024;
    ASSERT_LT(peak_rss_kib() - before, 2 * one_scratch_kib);
    const auto vals = l->get_const_values();
    for (gko::size_type nz = 0; nz < l->get_num_stored_elements(); ++nz) {
        ASSERT_TRUE(std::isfinite(vals[nz]));
    }
}


}  // namespace
