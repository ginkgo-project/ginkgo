// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <limits>
#include <memory>
#include <thread>


#include <gtest/gtest.h>


TEST(Sanitizers, UseAfterFree)
{
    char* x = new char[50];
    x[0] = 'H';
    x[1] = 'I';
    x[2] = '\n';

    std::free(x);

    static volatile char z = x[0];
}


TEST(Sanitizers, MemoryLeak)
{
    char* x = new char[50];
    x[0] = 'H';
    x[1] = 'I';
    x[2] = '\n';
}


TEST(Sanitizers, UndefinedBehavior)
{
    int x = std::numeric_limits<int>::max();
    int y = 10001;

    static volatile int z = x + y;
}


int Global = 0;
void* Thread(void* x)
{
    Global = 42;
    return x;
}


TEST(Sanitizers, RaceCondition)
{
    std::thread t(Thread, &Global);

    Global = 43;
    t.join();
}
