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

#include <iomanip>
#include <numeric>


#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {
namespace log {
namespace {


std::string format_duration(int64 time_ns)
{
    std::stringstream ss;
    ss << std::setprecision(1) << std::fixed;
    std::array<int64, 7> ranges{{999, 999'999, 999'999'999, 59'999'999'999,
                                 3'599'999'999'999, 86'399'999'999'999,
                                 std::numeric_limits<int64>::max()}};
    std::array<const char*, 7> units{
        {"ns", "us", "ms", "s ", "m ", "h ", "d "}};
    auto unit =
        std::distance(ranges.begin(),
                      std::lower_bound(ranges.begin(), ranges.end(), time_ns));
    if (unit == 0) {
        ss << double(time_ns) << ' ' << units[unit];
    } else {
        ss << (time_ns / double(ranges[unit - 1] + 1)) << ' ' << units[unit];
    }
    return ss.str();
}


std::string format_avg_duration(int64 time_ns, int64 count)
{
    return format_duration(time_ns / std::max(count, int64{1}));
}


std::string format_fraction(int64 part, int64 whole)
{
    std::stringstream ss;
    ss << std::setprecision(1) << std::fixed;
    ss << (double(part) / double(whole)) * 100.0 << " %";
    return ss.str();
}


template <std::size_t size>
void print_table(const std::array<std::string, size>& headers,
                 const std::vector<std::array<std::string, size>>& table,
                 std::ostream& stream)
{
    std::array<std::size_t, size> widths;
    for (int i = 0; i < widths.size(); i++) {
        widths[i] = headers[i].size();
    }
    for (const auto& row : table) {
        for (int i = 0; i < widths.size(); i++) {
            widths[i] = std::max(widths[i], row[i].size());
        }
    }
    for (int i = 0; i < widths.size(); i++) {
        stream << '|';
        const auto align1 = (widths[i] - headers[i].size()) / 2;
        const auto align2 = (widths[i] - headers[i].size()) - align1;
        stream << std::string(align1, ' ') << headers[i]
               << std::string(align2, ' ');
    }
    stream << "|\n";
    for (int i = 0; i < widths.size(); i++) {
        stream << '|';
        // right-align for Markdown, assuming widths[i] > 0
        stream << std::string(widths[i] - 1, '-') << (i == 0 ? '-' : ':');
    }
    stream << "|\n";
    for (const auto& row : table) {
        for (int i = 0; i < widths.size(); i++) {
            stream << '|';
            stream << std::setw(widths[i]) << (i > 0 ? std::right : std::left)
                   << row[i];
        }
        stream << "|\n";
    }
}


}  // namespace


ProfilerHook::TableSummaryWriter::TableSummaryWriter(std::ostream& output,
                                                     std::string header)
    : output_{&output}, header_{std::move(header)}
{}


void ProfilerHook::TableSummaryWriter::write(
    const std::vector<summary_entry>& entries, int64 overhead_ns)
{
    (*output_) << header_ << '\n'
               << "Overhead estimate " << format_duration(overhead_ns) << '\n';
    auto sorted_entries = entries;
    std::sort(sorted_entries.begin(), sorted_entries.end(),
              [](const summary_entry& lhs, const summary_entry& rhs) {
                  // reverse-sort by inclusive total time
                  return lhs.inclusive_ns > rhs.inclusive_ns;
              });
    std::vector<std::array<std::string, 6>> table;
    std::array<std::string, 6> headers({" name ", " total ", " total (self) ",
                                        " count ", " avg ", " avg (self) "});
    for (const auto& entry : sorted_entries) {
        table.emplace_back(std::array<std::string, 6>{
            " " + entry.name + " ",
            " " + format_duration(entry.inclusive_ns) + " ",
            " " + format_duration(entry.exclusive_ns) + " ",
            " " + std::to_string(entry.count) + " ",
            " " + format_avg_duration(entry.inclusive_ns, entry.count) + " ",
            " " + format_avg_duration(entry.exclusive_ns, entry.count) + " "});
    }
    print_table(headers, table, *output_);
}


void ProfilerHook::TableSummaryWriter::write_nested(
    const nested_summary_entry& root, int64 overhead_ns)
{
    (*output_) << header_ << '\n'
               << "Overhead estimate " << format_duration(overhead_ns) << '\n';
    std::vector<std::array<std::string, 5>> table;
    std::array<std::string, 5> headers(
        {" name ", " total ", " fraction ", " count ", " avg "});
    auto visitor = [&table](auto visitor, const nested_summary_entry& node,
                            int64 parent_elapsed_ns,
                            std::size_t depth) -> void {
        std::vector<int64> child_permutation(node.children.size());
        const auto total_child_duration =
            std::accumulate(node.children.begin(), node.children.end(), int64{},
                            [](int64 acc, const nested_summary_entry& child) {
                                return acc + child.elapsed_ns;
                            });
        const auto self_duration = node.elapsed_ns - total_child_duration;
        std::iota(child_permutation.begin(), child_permutation.end(), 0);
        std::sort(child_permutation.begin(), child_permutation.end(),
                  [&node](int64 lhs, int64 rhs) {
                      // sort by elapsed time in descending order
                      return node.children[lhs].elapsed_ns >
                             node.children[rhs].elapsed_ns;
                  });
        table.emplace_back(std::array<std::string, 5>{
            std::string(2 * depth + 1, ' ') + node.name + " ",
            " " + format_duration(node.elapsed_ns) + " ",
            format_fraction(node.elapsed_ns, parent_elapsed_ns) + " ",
            " " + std::to_string(node.count) + " ",
            " " + format_avg_duration(node.elapsed_ns, node.count) + " "});
        nested_summary_entry self_entry{
            "(self)", self_duration, node.count, {}};
        bool printed_self = false;
        for (const auto child_id : child_permutation) {
            // print (self) entry before smaller entry ...
            if (!printed_self &&
                node.children[child_id].elapsed_ns < self_duration) {
                visitor(visitor, self_entry, node.elapsed_ns, depth + 1);
                printed_self = true;
            }
            visitor(visitor, node.children[child_id], node.elapsed_ns,
                    depth + 1);
        }
        // ... or at the end
        if (!printed_self && !node.children.empty()) {
            visitor(visitor, self_entry, node.elapsed_ns, depth + 1);
        }
    };
    visitor(visitor, root, root.elapsed_ns, 0);
    print_table(headers, table, *output_);
}


}  // namespace log
}  // namespace gko
