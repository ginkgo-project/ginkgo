// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/components/load_balancing.hpp"

#include <memory>
#include <numeric>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/base/index_range.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


class LoadBalancing : public HipTestFixture {
protected:
};


__global__ void test_1_work_per_chunk(int* output, int* size_once, bool* err,
                                      int size)
{
    gko::kernels::hip::load_balance_subwarp<
        gko::kernels::hip::config::warp_size>(
        size,
        [&](int chunk) {
            if (chunk < 0 || chunk >= size) {
                printf("oob %d\n", chunk);
                *err = true;
            }
            int result = atomicCAS(size_once + chunk, 0, 1);
            if (result != 0) {
                printf("duplicate size %d\n", chunk);
                *err = true;
            }
            return 1;
        },
        [&](int chunk, int work, int global_pos) {
            if (chunk < 0 || chunk >= size || work != 0) {
                printf("oob %d %d\n", chunk, work);
                *err = true;
            }
            if (global_pos != chunk) {
                printf("incorrect global_pos %d %d\n", global_pos, chunk);
                *err = true;
            }
            // make sure every work item happens only once
            int prev = atomicCAS(output + chunk, -1, chunk);
            if (prev != -1) {
                printf("duplicate %d %d\n", chunk, work);
                *err = true;
            }
        });
}


TEST_F(LoadBalancing, WorksWith1WorkPerChunk)
{
    for (const auto size : {0, 1, 2, 3, 7, 8, 63, 64, 65, 127, 128, 129, 200}) {
        SCOPED_TRACE(size);
        const auto usize = static_cast<gko::size_type>(size);
        gko::array<int> data{this->exec, usize};
        data.fill(-1);
        gko::array<bool> err{this->exec, 1};
        err.fill(false);
        gko::array<int> size_once{this->exec, usize};
        size_once.fill(0);
        gko::array<int> ref_data{this->exec, usize};
        gko::kernels::hip::components::fill_seq_array(exec, ref_data.get_data(),
                                                      usize);

        test_1_work_per_chunk<<<1, gko::kernels::hip::config::warp_size>>>(
            data.get_data(), size_once.get_data(), err.get_data(), size);

        GKO_ASSERT_ARRAY_EQ(data, ref_data);
        ASSERT_FALSE(this->exec->copy_val_to_host(err.get_const_data()));
    }
}


__global__ void test_3_work_per_chunk(int* output, int* size_once, bool* err,
                                      int size)
{
    gko::kernels::hip::load_balance_subwarp<
        gko::kernels::hip::config::warp_size>(
        size,
        [&](int chunk) {
            if (chunk < 0 || chunk >= size) {
                printf("oob %d\n", chunk);
                *err = true;
            }
            int result = atomicCAS(size_once + chunk, 0, 1);
            if (result != 0) {
                printf("duplicate size %d\n", chunk);
                *err = true;
            }
            return 3;
        },
        [&](int chunk, int work, int global_pos) {
            if (chunk < 0 || chunk >= size || work < 0 || work >= 3) {
                printf("oob %d %d\n", chunk, work);
                *err = true;
            }
            // make sure every work item happens only once
            int prev = atomicCAS(output + global_pos, -1, 3 * chunk + work);
            if (prev != -1) {
                printf("duplicate %d %d\n", chunk, work);
                *err = true;
            }
        });
}


TEST_F(LoadBalancing, WorksWith3WorkPerChunk)
{
    for (const auto size : {0, 1, 2, 3, 21, 22, 42, 43, 63, 64, 65, 100}) {
        SCOPED_TRACE(size);
        const auto usize = static_cast<gko::size_type>(size);
        gko::array<int> data{this->exec, 3 * usize};
        data.fill(-1);
        gko::array<bool> err{this->exec, 1};
        err.fill(false);
        gko::array<int> size_once{this->exec, usize};
        size_once.fill(0);
        gko::array<int> ref_data{this->exec, 3 * usize};
        gko::kernels::hip::components::fill_seq_array(exec, ref_data.get_data(),
                                                      3 * usize);

        test_3_work_per_chunk<<<1, gko::kernels::hip::config::warp_size>>>(
            data.get_data(), size_once.get_data(), err.get_data(), size);

        GKO_ASSERT_ARRAY_EQ(data, ref_data);
        ASSERT_FALSE(this->exec->copy_val_to_host(err.get_const_data()));
    }
}


__global__ void test_dynamic_work_per_chunk(const int* chunk_sizes, int size,
                                            int* output_chunk, int* output_work,
                                            int* size_once, bool* err)
{
    gko::kernels::hip::load_balance_subwarp<
        gko::kernels::hip::config::warp_size>(
        size,
        [&](int chunk) {
            if (chunk < 0 || chunk >= size) {
                printf("oob %d\n", chunk);
                *err = true;
            }
            int result = atomicCAS(size_once + chunk, 0, 1);
            if (result != 0) {
                printf("duplicate size %d\n", chunk);
                *err = true;
            }
            return chunk_sizes[chunk];
        },
        [&](int chunk, int work, int global_pos) {
            if (chunk < 0 || chunk >= size || work < 0 ||
                work >= chunk_sizes[chunk]) {
                printf("oob %d %d\n", chunk, work);
                *err = true;
            }
            // make sure every work item happens only once
            int prev = atomicCAS(output_chunk + global_pos, -1, chunk);
            int prev2 = atomicCAS(output_work + global_pos, -1, work);
            if (prev != -1 || prev2 != -1) {
                printf("duplicate %d %d\n", chunk, work);
                *err = true;
            }
        });
}


TEST_F(LoadBalancing, WorksWithRandomWorkPerChunk)
{
    std::default_random_engine rng{123468};
    std::uniform_int_distribution<int> dist{1, 100};
    for (const auto size :
         {0, 1, 2, 3, 7, 8, 31, 32, 33, 34, 63, 64, 65, 100}) {
        SCOPED_TRACE(size);
        const auto usize = static_cast<gko::size_type>(size);
        gko::array<int> chunk_sizes{this->ref, usize};
        std::generate_n(chunk_sizes.get_data(), size,
                        [&] { return dist(rng); });
        const auto total_work = static_cast<gko::size_type>(std::reduce(
            chunk_sizes.get_const_data(), chunk_sizes.get_const_data() + size));
        gko::array<int> data_chunk{this->ref, total_work};
        gko::array<int> data_work{this->ref, total_work};
        gko::array<int> ddata_chunk{this->exec, total_work};
        gko::array<int> ddata_work{this->exec, total_work};
        gko::array<bool> derr{this->exec, 1};
        gko::array<int> dsize_once{this->exec, usize};
        gko::array<int> dchunk_sizes{this->exec, chunk_sizes};
        ddata_chunk.fill(-1);
        ddata_work.fill(-1);
        derr.fill(false);
        dsize_once.fill(0);
        int i{};
        for (const auto chunk : gko::irange{size}) {
            const auto chunk_size = chunk_sizes.get_const_data()[chunk];
            for (const auto work : gko::irange{chunk_size}) {
                data_chunk.get_data()[i] = chunk;
                data_work.get_data()[i] = work;
                i++;
            }
        }

        test_dynamic_work_per_chunk<<<1,
                                      gko::kernels::hip::config::warp_size>>>(
            dchunk_sizes.get_data(), size, ddata_chunk.get_data(),
            ddata_work.get_data(), dsize_once.get_data(), derr.get_data());

        GKO_ASSERT_ARRAY_EQ(ddata_chunk, data_chunk);
        GKO_ASSERT_ARRAY_EQ(ddata_work, data_work);
        ASSERT_FALSE(this->exec->copy_val_to_host(derr.get_const_data()));
    }
}


TEST_F(LoadBalancing, WorksWithRandomWorkPerChunkEmptyAllowed)
{
    // empty chunks aligned with warp size
    std::vector<int> empty_aligned(64, 0);
    // empty chunks not aligned with warp size
    std::vector<int> empty_unaligned(65, 0);
    // some empty chunks, not enough work initially
    std::vector<int> incomplete{0, 0, 63, 0, 0};
    // some empty chunks, not enough work in second iteration
    std::vector<int> incomplete2{0, 0, 65, 0, 0};
    // large gaps, never enough work
    std::vector<int> gaps_incomplete(128);
    gaps_incomplete[4] = 31;
    gaps_incomplete[4 + 64] = 31;
    // large gaps, enough work
    std::vector<int> gaps_complete(128);
    gaps_incomplete[4] = 64;
    gaps_incomplete[4 + 64] = 64;
    // large gaps, chunks overlap between iterations
    std::vector<int> gaps_overlap(96);
    gaps_overlap[31] = 127;
    gaps_overlap[31 + 64] = 64;
    // large gaps, work at the end of the chunk
    std::vector<int> gaps_complete_end(96);
    gaps_overlap[31] = 128;
    gaps_overlap[31 + 64] = 64;

    for (auto input :
         {empty_aligned, empty_unaligned, incomplete, incomplete2}) {
        gko::array<int> chunk_sizes{this->ref, input.begin(), input.end()};
        auto size = static_cast<int>(input.size());
        const auto total_work = static_cast<gko::size_type>(std::reduce(
            chunk_sizes.get_const_data(), chunk_sizes.get_const_data() + size));
        gko::array<int> data_chunk{this->ref, total_work};
        gko::array<int> data_work{this->ref, total_work};
        gko::array<int> ddata_chunk{this->exec, total_work};
        gko::array<int> ddata_work{this->exec, total_work};
        gko::array<bool> derr{this->exec, 1};
        gko::array<int> dsize_once{this->exec, static_cast<unsigned>(size)};
        gko::array<int> dchunk_sizes{this->exec, chunk_sizes};
        ddata_chunk.fill(-1);
        ddata_work.fill(-1);
        derr.fill(false);
        dsize_once.fill(0);
        int i{};
        for (const auto chunk : gko::irange{size}) {
            const auto chunk_size = chunk_sizes.get_const_data()[chunk];
            for (const auto work : gko::irange{chunk_size}) {
                data_chunk.get_data()[i] = chunk;
                data_work.get_data()[i] = work;
                i++;
            }
        }

        test_dynamic_work_per_chunk<<<1,
                                      gko::kernels::hip::config::warp_size>>>(
            dchunk_sizes.get_data(), size, ddata_chunk.get_data(),
            ddata_work.get_data(), dsize_once.get_data(), derr.get_data());

        GKO_ASSERT_ARRAY_EQ(ddata_chunk, data_chunk);
        GKO_ASSERT_ARRAY_EQ(ddata_work, data_work);
        ASSERT_FALSE(this->exec->copy_val_to_host(derr.get_const_data()));
    }
}


}  // namespace
