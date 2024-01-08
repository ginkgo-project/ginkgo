// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/******************************************************************************
 *
 * Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC
 * and other gtest-mpi-listener developers. See the COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *******************************************************************************/

#include <cassert>
#include <sstream>
#include <string>
#include <vector>


#include <mpi.h>


#include <gtest/gtest.h>


#include <ginkgo/core/base/mpi.hpp>


#include "core/test/gtest/environments.hpp"


namespace GTestMPIListener {

// This class sets up the global test environment, which is needed
// to finalize MPI.
class MPIEnvironment : public ::testing::Environment {
public:
    MPIEnvironment() : ::testing::Environment() {}

    virtual ~MPIEnvironment() {}

    virtual void SetUp()
    {
        int is_mpi_initialized;
        ASSERT_EQ(MPI_Initialized(&is_mpi_initialized), MPI_SUCCESS);
        if (!is_mpi_initialized) {
            printf("MPI must be initialized before RUN_ALL_TESTS!\n");
            printf("Add '::testing::InitGoogleTest(&argc, argv);\n");
            printf("     MPI_Init(&argc, &argv);' to your 'main' function!\n");
            FAIL();
        }
    }

    virtual void TearDown()
    {
        int is_mpi_finalized;
        ASSERT_EQ(MPI_Finalized(&is_mpi_finalized), MPI_SUCCESS);
        if (!is_mpi_finalized) {
            int rank;
            ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
            if (rank == 0) {
                printf("Finalizing MPI...\n");
            }
            ASSERT_EQ(MPI_Finalize(), MPI_SUCCESS);
        }
        ASSERT_EQ(MPI_Finalized(&is_mpi_finalized), MPI_SUCCESS);
        ASSERT_TRUE(is_mpi_finalized);
    }

private:
    // Disallow copying
    MPIEnvironment(const MPIEnvironment& env) {}
};  // class MPIEnvironment


// This class more or less takes the code in Google Test's
// MinimalistPrinter example and wraps certain parts of it in MPI calls,
// gathering all results onto rank zero.
class MPIWrapperPrinter : public ::testing::TestEventListener {
public:
    MPIWrapperPrinter(::testing::TestEventListener* l, MPI_Comm comm_)
        : ::testing::TestEventListener(), listener(l), result_vector()
    {
        int is_mpi_initialized;
        MPI_Initialized(&is_mpi_initialized);
        if (!is_mpi_initialized) {
            printf("MPI must be initialized before RUN_ALL_TESTS!\n");
            printf("Add '::testing::InitGoogleTest(&argc, argv);\n");
            printf("     MPI_Init(&argc, &argv);' to your 'main' function!\n");
            assert(0);
        }

        MPI_Comm_dup(comm_, &comm);
        UpdateCommState();
    }

    MPIWrapperPrinter(const MPIWrapperPrinter& printer)
        : listener(printer.listener), result_vector(printer.result_vector)
    {
        int is_mpi_initialized;
        MPI_Initialized(&is_mpi_initialized);
        if (!is_mpi_initialized) {
            printf("MPI must be initialized before RUN_ALL_TESTS!\n");
            printf("Add '::testing::InitGoogleTest(&argc, argv);\n");
            printf("     MPI_Init(&argc, &argv);' to your 'main' function!\n");
            assert(0);
        }

        MPI_Comm_dup(printer.comm, &comm);
        UpdateCommState();
    }

    // Called before test activity starts
    virtual void OnTestProgramStart(const ::testing::UnitTest& unit_test)
    {
        if (rank == 0) {
            listener->OnTestProgramStart(unit_test);
        }
    }


    // Called before each test iteration starts, where iteration is
    // the iterate index. There could be more than one iteration if
    // GTEST_FLAG(repeat) is used.
    virtual void OnTestIterationStart(const ::testing::UnitTest& unit_test,
                                      int iteration)
    {
        if (rank == 0) {
            listener->OnTestIterationStart(unit_test, iteration);
        }
    }


    // Called before environment setup before start of each test iteration
    virtual void OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test)
    {
        if (rank == 0) {
            listener->OnEnvironmentsSetUpStart(unit_test);
        }
    }

    virtual void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& unit_test)
    {
        if (rank == 0) {
            listener->OnEnvironmentsSetUpEnd(unit_test);
        }
    }

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
    virtual void OnTestCaseStart(const ::testing::TestCase& test_case)
    {
        if (rank == 0) {
            listener->OnTestCaseStart(test_case);
        }
    }
#endif  // GTEST_REMOVE_LEGACY_TEST_CASEAPI_

    // Called before a test starts.
    virtual void OnTestStart(const ::testing::TestInfo& test_info)
    {
        // Only need to report test start info on rank 0
        if (rank == 0) {
            listener->OnTestStart(test_info);
        }
    }

    // Called after an assertion failure or an explicit SUCCESS() macro.
    // In an MPI program, this means that certain ranks may not call this
    // function if a test part does not fail on all ranks. Consequently, it
    // is difficult to have explicit synchronization points here.
    virtual void OnTestPartResult(
        const ::testing::TestPartResult& test_part_result)
    {
        result_vector.push_back(test_part_result);
        if (rank == 0) {
            listener->OnTestPartResult(test_part_result);
        }
    }

    // Called after a test ends.
    virtual void OnTestEnd(const ::testing::TestInfo& test_info)
    {
        int localResultCount = result_vector.size();
        std::vector<int> resultCountOnRank(size, 0);
        MPI_Gather(&localResultCount, 1, MPI_INT, &resultCountOnRank[0], 1,
                   MPI_INT, 0, comm);

        if (rank != 0) {
            // Nonzero ranks send constituent parts of each result to rank 0
            for (int i = 0; i < localResultCount; i++) {
                const ::testing::TestPartResult test_part_result =
                    result_vector.at(i);
                int resultStatus(test_part_result.failed());
                std::string resultFileName(test_part_result.file_name());
                int resultLineNumber(test_part_result.line_number());
                std::string resultMessage(test_part_result.message());

                int resultFileNameSize(resultFileName.size());
                int resultMessageSize(resultMessage.size());

                MPI_Send(&resultStatus, 1, MPI_INT, 0, rank, comm);
                MPI_Send(&resultFileNameSize, 1, MPI_INT, 0, rank, comm);
                MPI_Send(&resultLineNumber, 1, MPI_INT, 0, rank, comm);
                MPI_Send(&resultMessageSize, 1, MPI_INT, 0, rank, comm);
                MPI_Send(resultFileName.c_str(), resultFileNameSize, MPI_CHAR,
                         0, rank, comm);
                MPI_Send(resultMessage.c_str(), resultMessageSize, MPI_CHAR, 0,
                         rank, comm);
            }
        } else {
            // Rank 0 first prints its local result data
            for (int i = 0; i < localResultCount; i++) {
                const ::testing::TestPartResult test_part_result =
                    result_vector.at(i);
                if (test_part_result.failed()) {
                    std::string message(test_part_result.message());
                    std::istringstream input_stream(message);
                    std::stringstream to_stream_into_failure;
                    std::string line_as_string;
                    while (std::getline(input_stream, line_as_string)) {
                        to_stream_into_failure << "[Rank 0/" << size << "] "
                                               << line_as_string << std::endl;
                    }

                    ADD_FAILURE_AT(test_part_result.file_name(),
                                   test_part_result.line_number())
                        << to_stream_into_failure.str();
                }
            }

            for (int r = 1; r < size; r++) {
                for (int i = 0; i < resultCountOnRank[r]; i++) {
                    int resultStatus, resultFileNameSize, resultLineNumber;
                    int resultMessageSize;
                    MPI_Recv(&resultStatus, 1, MPI_INT, r, r, comm,
                             MPI_STATUS_IGNORE);
                    MPI_Recv(&resultFileNameSize, 1, MPI_INT, r, r, comm,
                             MPI_STATUS_IGNORE);
                    MPI_Recv(&resultLineNumber, 1, MPI_INT, r, r, comm,
                             MPI_STATUS_IGNORE);
                    MPI_Recv(&resultMessageSize, 1, MPI_INT, r, r, comm,
                             MPI_STATUS_IGNORE);

                    std::vector<char> fileNameBuffer(resultFileNameSize);
                    std::vector<char> messageBuffer(resultMessageSize);
                    MPI_Recv(&fileNameBuffer[0], resultFileNameSize, MPI_CHAR,
                             r, r, comm, MPI_STATUS_IGNORE);
                    MPI_Recv(&messageBuffer[0], resultMessageSize, MPI_CHAR, r,
                             r, comm, MPI_STATUS_IGNORE);

                    std::string resultFileName(fileNameBuffer.begin(),
                                               fileNameBuffer.end());
                    std::string resultMessage(messageBuffer.begin(),
                                              messageBuffer.end());

                    bool testPartHasFailed = (resultStatus == 1);
                    if (testPartHasFailed) {
                        std::string message(resultMessage);
                        std::istringstream input_stream(message);
                        std::stringstream to_stream_into_failure;
                        std::string line_as_string;

                        while (std::getline(input_stream, line_as_string)) {
                            to_stream_into_failure
                                << "[Rank " << r << "/" << size << "] "
                                << line_as_string << std::endl;
                        }

                        ADD_FAILURE_AT(resultFileName.c_str(), resultLineNumber)
                            << to_stream_into_failure.str();
                    }
                }
            }
        }

        result_vector.clear();
        if (rank == 0) {
            listener->OnTestEnd(test_info);
        }
    }

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
    virtual void OnTestCaseEnd(const ::testing::TestCase& test_case)
    {
        if (rank == 0) {
            listener->OnTestCaseEnd(test_case);
        }
    }

#endif

    // Called before the Environment is torn down.
    virtual void OnEnvironmentsTearDownStart(
        const ::testing::UnitTest& unit_test)
    {
        int is_mpi_finalized;
        ASSERT_EQ(MPI_Finalized(&is_mpi_finalized), MPI_SUCCESS);
        if (!is_mpi_finalized) {
            MPI_Comm_free(&comm);
        }
        if (rank == 0) {
            listener->OnEnvironmentsTearDownStart(unit_test);
        }
    }

    virtual void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test)
    {
        if (rank == 0) {
            listener->OnEnvironmentsTearDownEnd(unit_test);
        }
    }

    virtual void OnTestIterationEnd(const ::testing::UnitTest& unit_test,
                                    int iteration)
    {
        if (rank == 0) {
            listener->OnTestIterationEnd(unit_test, iteration);
        }
    }

    // Called when test driver program ends
    virtual void OnTestProgramEnd(const ::testing::UnitTest& unit_test)
    {
        if (rank == 0) {
            listener->OnTestProgramEnd(unit_test);
        }
    }

private:
    // Use a pointer here instead of a reference because
    // ::testing::TestEventListeners::Release returns a pointer
    // (namely, one of type ::testing::TesteEventListener*).
    ::testing::TestEventListener* listener;
    MPI_Comm comm;
    int rank;
    int size;
    std::vector<::testing::TestPartResult> result_vector;

    int UpdateCommState()
    {
        int flag = MPI_Comm_rank(comm, &rank);
        if (flag != MPI_SUCCESS) {
            return flag;
        }
        flag = MPI_Comm_size(comm, &size);
        return flag;
    }
};

}  // namespace GTestMPIListener


int ResourceEnvironment::omp_threads = 0;
int ResourceEnvironment::cuda_device_id = 0;
int ResourceEnvironment::hip_device_id = 0;
int ResourceEnvironment::sycl_device_id = 0;


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);
    MPI_Comm comm(MPI_COMM_WORLD);
    int rank;
    int size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::AddGlobalTestEnvironment(new ResourceEnvironment(rank, size));
    ::testing::AddGlobalTestEnvironment(new DeviceEnvironment(rank));
    MPI_Barrier(comm);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    ::testing::TestEventListener* l =
        listeners.Release(listeners.default_result_printer());
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, comm));
    int result = RUN_ALL_TESTS();
    return result;
}
