/*
 * Copyright 2017-2018
 *
 * Karlsruhe Institute of Technology
 *
 * Universitat Jaume I
 *
 * University of Tennessee
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <core/base/exception.hpp>


#include <gtest/gtest.h>


namespace {


TEST(ExceptionClasses, ErrorReturnsCorrectWhatMessage)
{
    gko::Error error("test_file.cpp", 1, "test error");
    ASSERT_EQ(std::string("test_file.cpp:1: test error"), error.what());
}


TEST(ExceptionClasses, NotImplementedReturnsCorrectWhatMessage)
{
    gko::NotImplemented error("test_file.cpp", 25, "test_func");
    ASSERT_EQ(std::string("test_file.cpp:25: test_func is not implemented"),
              error.what());
}


TEST(ExceptionClasses, NotCompiledReturnsCorrectWhatMessage)
{
    gko::NotCompiled error("test_file.cpp", 345, "test_func", "nvidia");
    ASSERT_EQ(std::string("test_file.cpp:345: feature test_func is part of the "
                          "nvidia module, which is not compiled on this "
                          "system"),
              error.what());
}


TEST(ExceptionClasses, NotSupportedReturnsCorrectWhatMessage)
{
    gko::NotSupported error("test_file.cpp", 123, "test_func", "test_obj");
    ASSERT_EQ(
        std::string("test_file.cpp:123: Operation test_func does not support "
                    "parameters of type test_obj"),
        error.what());
}


TEST(ExceptionClasses, DimensionMismatchReturnsCorrectWhatMessage)
{
    gko::DimensionMismatch error("test_file.cpp", 243, "test_func", "a", 3, 4,
                                 "b", 2, 5, "my_clarify");
    ASSERT_EQ(std::string("test_file.cpp:243: test_func: attempting to combine "
                          "operators a [3 x 4] and b [2 x 5]: my_clarify"),
              error.what());
}


TEST(ExceptionClasses, NotFoundReturnsCorrectWhatMessage)
{
    gko::NotFound error("test_file.cpp", 195, "my_func", "my error");
    ASSERT_EQ(std::string("test_file.cpp:195: my_func: my error"),
              error.what());
}


TEST(ExceptionClasses, AllocationErrorReturnsCorrectWhatMessage)
{
    gko::AllocationError error("test_file.cpp", 42, "CPU", 135);
    ASSERT_EQ(
        std::string("test_file.cpp:42: CPU: failed to allocate memory block "
                    "of 135B"),
        error.what());
}


}  // namespace
