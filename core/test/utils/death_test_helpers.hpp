// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_DEATH_TEST_HELPERS_HPP_
#define GKO_CORE_TEST_UTILS_DEATH_TEST_HELPERS_HPP_


inline bool check_assertion_exit_code(int exit_code)
{
#ifdef _MSC_VER
    // MSVC picks up the exit code incorrectly,
    // so we can only check that it exits
    return true;
#else
    return exit_code != 0;
#endif
}


#define EXPECT_ASSERT_FAILURE(_expression, ...) \
    EXPECT_EXIT((void)(_expression), check_assertion_exit_code, "")


#endif  // GKO_CORE_TEST_UTILS_DEATH_TEST_HELPERS_HPP_
