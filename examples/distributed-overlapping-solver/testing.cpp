//
// Created by marcel on 28.03.23.
//

#include <gtest/gtest.h>
#include <ginkgo/ginkgo.hpp>
#include "overlap.hpp"

void test_non_ovlp_apply(gko::experimental::mpi::communicator comm)
{
    auto rank = comm.rank();
    auto exec = gko::ReferenceExecutor::create();
    auto global = gko::initialize<mtx>({{1, 2, 0, 3, 0},
                                        {0, 4, 0, 0, 5},
                                        {0, 0, 6, 0, 7},
                                        {3, 0, 0, 8, 9},
                                        {0, 5, 7, 10, 11}},
                                       exec);
    std::array<std::shared_ptr<mtx>, 2> locals = {
        gko::initialize<mtx>(
            {
                {1, 2, 3, 0},
                {0, 4, 0, 5},
                {3, 0, 8 / 2.0, 9 / 2.0},
                {0, 5, 10 / 2.0, 11 / 2.0},
            },
            exec),
        gko::initialize<mtx>(
            {{6, 0, 7}, {0, 8 / 2.0, 9 / 2.0}, {7, 10 / 2.0, 11 / 2.0}}, exec)};
    auto global_x = gko::initialize<vec>({1, -2, -3, 4, 5}, exec);
    std::array<std::shared_ptr<vec>, 2> local_xs = {
        gko::initialize<vec>({1, -2, 4, 5}, exec),
        gko::initialize<vec>({-3, 4, 5}, exec)};
    auto global_y = global_x->clone();
    auto local_y = local_xs[rank]->clone();

    std::array<gko::array<shared_idx_t>, 2> shared_idxs = {
        gko::array<shared_idx_t>{
            exec,
            std::initializer_list<shared_idx_t>{{2, 1, 1, 0}, {3, 2, 1, 0}}},
        gko::array<shared_idx_t>{exec, std::initializer_list<shared_idx_t>{
                                           {1, 2, 0, 1}, {2, 3, 0, 1}}}};
    gko::comm_info_t comm_info{comm, shared_idxs[rank]};

    global->apply(global_x, global_y);
    locals[rank]->apply(local_xs[rank], local_y);
    gko::overlapping_vec{exec, comm, gko::make_dense_view(local_y), comm_info}
        .make_consistent(gko::overlapping_vec::operation::add);

    std::array<std::vector<int>, 2> local_to_global = {
        std::vector<int>{0, 1, 3, 4}, std::vector<int>{2, 3, 4}};

    for (int i = 0; i < local_to_global[rank].size(); ++i) {
        auto diff =
            std::abs(local_y->at(i) - global_y->at(local_to_global[rank][i]));
        if (diff > 1e-12) {
            std::cout << i << "|" << local_to_global[rank][i] << ": "
                      << local_y->at(i) << " "
                      << global_y->at(local_to_global[rank][i]) << std::endl;
        }
    }
}

void test_non_ovlp_norm(gko::experimental::mpi::communicator comm)
{
    auto rank = comm.rank();
    auto exec = gko::ReferenceExecutor::create();
    auto global_x = gko::initialize<vec>({1, -2, -3, 4, 5}, exec);
    std::array<std::shared_ptr<vec>, 2> local_xs = {
        gko::initialize<vec>({1, -2, 4, 5}, exec),
        gko::initialize<vec>({-3, 4, 5}, exec)};

    auto result_g = gko::initialize<vec>({0.0}, exec);
    auto result_l = gko::initialize<vec>({0.0}, exec);


    std::array<gko::array<shared_idx_t>, 2> shared_idxs = {
        gko::array<shared_idx_t>{
            exec,
            std::initializer_list<shared_idx_t>{{2, 1, 1, 0}, {3, 2, 1, 0}}},
        gko::array<shared_idx_t>{exec, std::initializer_list<shared_idx_t>{
                                           {1, 2, 0, 1}, {2, 3, 0, 1}}}};
    gko::comm_info_t comm_info{comm, shared_idxs[rank]};

    global_x->compute_norm2()
}


int main(int argc, char** argv)
{
    gko::experimental::mpi::environment env(argc, argv);
    gko::experimental::mpi::communicator comm(MPI_COMM_WORLD);

    test_non_ovlp_apply(comm);
}
