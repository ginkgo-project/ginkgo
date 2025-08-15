// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_
#define GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_


#include <gflags/gflags.h>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"


std::string reordering_algorithm_desc =
    "Reordering algorithm to apply to the input matrices:\n"
    "    none - no reordering\n"
    "    amd - Approximate Minimum Degree reordering algorithm\n"
#if GKO_HAVE_METIS
    "    nd - Nested Dissection reordering algorithm\n"
#endif
    "    rcm - Reverse Cuthill-McKee reordering algorithm\n"
    "This is a preprocessing step whose runtime will not be included\n"
    "in the measurements.";


#ifndef GKO_BENCHMARK_DISTRIBUTED
DEFINE_string(reorder, "none", reordering_algorithm_desc.c_str());
#endif


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Permutation<IndexType>> reorder(
    gko::matrix_data<ValueType, IndexType>& data, json& test_case)
{
#ifndef GKO_BENCHMARK_DISTRIBUTED
    if (FLAGS_reorder == "none") {
        return nullptr;
    }
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto ref = gko::ReferenceExecutor::create();
    auto mtx = gko::share(Csr::create(ref));
    mtx->read(data);
    std::unique_ptr<gko::matrix::Permutation<IndexType>> perm;
    if (FLAGS_reorder == "amd") {
        perm = gko::experimental::reorder::Amd<IndexType>::build()
                   .on(ref)
                   ->generate(mtx);
#if GKO_HAVE_METIS
    } else if (FLAGS_reorder == "nd") {
        perm = gko::experimental::reorder::NestedDissection<ValueType,
                                                            IndexType>::build()
                   .on(ref)
                   ->generate(mtx);
#endif
    } else if (FLAGS_reorder == "rcm") {
        perm = gko::experimental::reorder::Rcm<IndexType>::build()
                   .on(ref)
                   ->generate(mtx);
    } else {
        throw std::runtime_error{"Unknown reordering algorithm " +
                                 FLAGS_reorder};
    }
    auto perm_arr =
        gko::array<IndexType>::view(ref, data.size[0], perm->get_permutation());
    gko::as<Csr>(mtx->permute(&perm_arr))->write(data);
    test_case["reordered"] = FLAGS_reorder;
    return perm;
#else
    // no reordering for distributed benchmarks
    return nullptr;
#endif
}


template <typename ValueType, typename IndexType>
void permute(std::unique_ptr<gko::matrix::Dense<ValueType>>& vec,
             gko::matrix::Permutation<IndexType>* perm)
{
    auto perm_arr = gko::array<IndexType>::view(
        perm->get_executor(), perm->get_size()[0], perm->get_permutation());
    vec = gko::as<gko::matrix::Dense<ValueType>>(vec->row_permute(&perm_arr));
}


template <typename ValueType, typename IndexType>
void permute(
    std::unique_ptr<gko::experimental::distributed::Vector<ValueType>>& vec,
    gko::matrix::Permutation<IndexType>* perm)
{}


#endif  // GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_
