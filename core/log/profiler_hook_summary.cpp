/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <chrono>
#include <iomanip>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>


#include "core/log/profiler_hook.hpp"


namespace gko {
namespace log {
namespace profiler_hook {


using clock = std::chrono::steady_clock;
using duration = clock::duration;
using time_point = clock::time_point;


struct summary_entry {
    std::string name;
    duration inclusive{};
    duration exclusive{};
    int64 count{};

    summary_entry(std::string name) : name{std::move(name)} {}
};


struct partial_summary_entry {
    int64 name_id;
    time_point start;

    partial_summary_entry(int64 name_id, time_point start)
        : name_id{name_id}, start{start}
    {}
};


struct summary {
    duration overhead{};
    std::vector<partial_summary_entry> range_stack;
    std::unordered_map<std::string, int64> name_map;
    std::vector<summary_entry> entries;
    std::mutex mutex{};

    summary() { push("total"); }

    void push(const std::string& name)
    {
        const auto now = clock::now();
        std::lock_guard<std::mutex> guard{mutex};
        auto it = name_map.find(name);
        if (it == name_map.end()) {
            const auto new_id = static_cast<int64>(entries.size());
            it = name_map.emplace_hint(it, name, new_id);
            entries.emplace_back(name);
        }
        const auto id = it->second;
        range_stack.emplace_back(id, now);
        overhead += clock::now() - now;
    }

    void pop(const std::string& name)
    {
        const auto now = clock::now();
        std::lock_guard<std::mutex> guard{mutex};
        auto id = name_map.at(name);
        if (range_stack.empty() || range_stack.back().name_id != id) {
            throw std::runtime_error{"Ranges are not nested: trying to pop " +
                                     name + " from " +
                                     entries[range_stack.back().name_id].name};
        }
        const auto partial_entry = range_stack.back();
        range_stack.pop_back();
        auto& entry = entries[id];
        const auto elapsed = now - partial_entry.start;
        entry.count++;
        entry.inclusive += elapsed;
        entry.exclusive += elapsed;
        if (!range_stack.empty()) {
            entries[range_stack.back().name_id].exclusive -= elapsed;
        }
        overhead += clock::now() - now;
    }

    void pop_all(std::ostream& warn_stream)
    {
        while (!range_stack.empty()) {
            if (range_stack.back().name_id != 0) {
                warn_stream << "Warning: range "
                            << entries[range_stack.back().name_id].name
                            << " left open\n";
            }
            pop(entries[range_stack.back().name_id].name);
        }
    }
};


std::string fmt_duration(int64 time_ns)
{
    std::stringstream ss;
    ss << std::setprecision(1) << std::fixed;
    std::array<int64, 7> ranges{{1000, 1'000'000, 1'000'000'000, 60'000'000'000,
                                 3'600'000'000'000, 86'400'000'000'000,
                                 std::numeric_limits<int64>::max()}};
    std::array<const char*, 7> units{
        {"ns ", "us ", "ms ", "s  ", "min", "h  ", "d  "}};
    auto unit =
        std::distance(ranges.begin(),
                      std::lower_bound(ranges.begin(), ranges.end(), time_ns));
    if (unit == 0) {
        ss << time_ns << units[unit];
    } else {
        ss << (time_ns / double(ranges[unit - 1])) << units[unit];
    }
    return ss.str();
}


std::string fmt_duration(duration d)
{
    return fmt_duration(
        std::chrono::duration_cast<std::chrono::nanoseconds, int64>(d).count());
}


std::string fmt_avg_duration(duration d, int64 count)
{
    return fmt_duration(
        std::chrono::duration_cast<std::chrono::nanoseconds, int64>(d).count() /
        count);
}


void print_summary(std::ostream& stream, const std::string& name, summary& s)
{
    stream << name << '\n'
           << "Overhead estimate " << fmt_duration(s.overhead) << '\n';
    s.pop_all(stream);
    std::sort(s.entries.begin(), s.entries.end(),
              [](const auto& lhs, const auto& rhs) {
                  // reverse-sort by inclusive total time
                  return lhs.inclusive > rhs.inclusive;
              });
    std::vector<std::array<std::string, 6>> table;
    std::array<std::string, 6> headers({" name ", " total ", " total (self) ",
                                        " count ", " avg ", " avg (self) "});
    for (const auto& entry : s.entries) {
        table.emplace_back(std::array<std::string, 6>{
            entry.name + " ", " " + fmt_duration(entry.inclusive) + " ",
            " " + fmt_duration(entry.exclusive) + " ",
            " " + std::to_string(entry.count) + " ",
            " " + fmt_avg_duration(entry.inclusive, entry.count) + " ",
            " " + fmt_avg_duration(entry.exclusive, entry.count) + " "});
    }
    std::array<std::size_t, 6> widths;
    for (int i = 0; i < widths.size(); i++) {
        widths[i] = headers[i].size();
    }
    for (const auto& row : table) {
        for (int i = 0; i < widths.size(); i++) {
            widths[i] = std::max(widths[i], row[i].size());
        }
    }
    for (int i = 0; i < widths.size(); i++) {
        if (i > 0) {
            stream << '|';
        }
        const auto align1 = (widths[i] - headers[i].size()) / 2;
        const auto align2 = (widths[i] - headers[i].size()) - align1;
        stream << std::string(align1, ' ') << headers[i]
               << std::string(align2, ' ');
    }
    stream << '\n';
    for (int i = 0; i < widths.size(); i++) {
        if (i > 0) {
            stream << '+';
        }
        stream << std::string(widths[i], '-');
    }
    stream << '\n';
    for (const auto& row : table) {
        for (int i = 0; i < widths.size(); i++) {
            if (i > 0) {
                stream << '|';
            }
            stream << std::setw(widths[i]) << (i > 0 ? std::right : std::left)
                   << row[i];
        }
        stream << '\n';
    }
}


}  // namespace profiler_hook


std::pair<ProfilerHook::hook_function, ProfilerHook::hook_function>
create_summary_fns(std::ostream& stream, std::string name)
{
    std::shared_ptr<profiler_hook::summary> data{
        new profiler_hook::summary{},
        [&stream, name](profiler_hook::summary* ptr) {
            print_summary(stream, name, *ptr);
            delete ptr;
        }};
    return std::make_pair(
        [data](const char* name, profile_event_category) { data->push(name); },
        [data](const char* name, profile_event_category) { data->pop(name); });
}


}  // namespace log
}  // namespace gko
