/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <limits>
#include <memory>
#include <thread>


#include <gtest/gtest.h>


TEST(Sanitizers, UseAfterFree)
{
    char *x = new char[50];
    x[0] = 'H';
    x[1] = 'I';
    x[2] = '\n';

    std::free(x);

    static volatile char z = x[0];
}


TEST(Sanitizers, MemoryLeak)
{
    char *x = new char[50];
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
void *Thread(void *x)
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
