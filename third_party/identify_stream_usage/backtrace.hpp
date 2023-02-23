/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef STREAM_USAGE_BACKTRACE_HPP_
#define STREAM_USAGE_BACKTRACE_HPP_


#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <sstream>


inline bool check_backtrace(const char* ignore_symbol = nullptr)
{
#ifdef __GNUC__
    // If we're on the wrong stream, print the stack trace from the current
    // frame. Adapted from from
    // https://panthema.net/2008/0901-stacktrace-demangled/
    constexpr int kMaxStackDepth = 64;
    void* stack[kMaxStackDepth];
    auto depth = backtrace(stack, kMaxStackDepth);
    auto strings = backtrace_symbols(stack, depth);
    std::stringstream ss;

    if (strings == nullptr) {
        ss << "No stack trace could be found!" << std::endl;
    } else {
        // If we were able to extract a trace, parse it, demangle symbols,
        // and print a readable output.

        // allocate string which will be filled with the demangled function
        // name
        size_t funcnamesize = 256;
        char* funcname = (char*)malloc(funcnamesize);

        // Start at frame 1 to skip print_trace itself.
        for (int i = 1; i < depth; ++i) {
            char* begin_name = nullptr;
            char* begin_offset = nullptr;
            char* end_offset = nullptr;

            // find parentheses and +address offset surrounding the mangled
            // name:
            // ./module(function+0x15c) [0x8048a6d]
            for (char* p = strings[i]; *p; ++p) {
                if (*p == '(') {
                    begin_name = p;
                } else if (*p == '+') {
                    begin_offset = p;
                } else if (*p == ')' && begin_offset) {
                    end_offset = p;
                    break;
                }
            }

            if (begin_name && begin_offset && end_offset &&
                begin_name < begin_offset) {
                *begin_name++ = '\0';
                *begin_offset++ = '\0';
                *end_offset = '\0';

                if (ignore_symbol &&
                    std::string(begin_name, begin_offset - 1) ==
                        ignore_symbol) {
                    return false;
                }
                // mangled name is now in [begin_name, begin_offset) and
                // caller offset in [begin_offset, end_offset). now apply
                // __cxa_demangle():

                int status;
                char* ret = abi::__cxa_demangle(begin_name, funcname,
                                                &funcnamesize, &status);
                if (status == 0) {
                    funcname = ret;  // use possibly realloc()-ed string
                                     // (__cxa_demangle may realloc funcname)
                    ss << "#" << i << " in " << strings[i] << " : " << funcname
                       << "+" << begin_offset << std::endl;
                } else {
                    // demangling failed. Output function name as a C
                    // function with no arguments.
                    ss << "#" << i << " in " << strings[i] << " : "
                       << begin_name << "()+" << begin_offset << std::endl;
                }
            } else {
                ss << "#" << i << " in " << strings[i] << std::endl;
            }
        }

        free(funcname);
    }
    free(strings);
    std::cout << ss.str();
#else
    std::cout << "Backtraces are only when built with a GNU compiler."
              << std::endl;
#endif  // __GNUC__
    return true;
}


#endif  // STREAM_USAGE_BACKTRACE_HPP_