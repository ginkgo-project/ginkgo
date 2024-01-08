// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iomanip>
#include <numeric>


#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {
namespace log {
namespace {


std::string format_duration(std::chrono::nanoseconds time)
{
    const auto time_ns = time.count();
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


std::string format_avg_duration(std::chrono::nanoseconds time, int64 count)
{
    return format_duration(
        std::chrono::nanoseconds{time.count() / std::max(count, int64{1})});
}


std::string format_fraction(std::chrono::nanoseconds part,
                            std::chrono::nanoseconds whole)
{
    std::stringstream ss;
    ss << std::setprecision(1) << std::fixed;
    ss << (double(part.count()) / double(whole.count())) * 100.0 << " %";
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
    const std::vector<summary_entry>& entries,
    std::chrono::nanoseconds overhead)
{
    (*output_) << header_ << '\n'
               << "Overhead estimate " << format_duration(overhead) << '\n';
    auto sorted_entries = entries;
    std::sort(sorted_entries.begin(), sorted_entries.end(),
              [](const summary_entry& lhs, const summary_entry& rhs) {
                  // reverse-sort by inclusive total time
                  return lhs.inclusive > rhs.inclusive;
              });
    std::vector<std::array<std::string, 6>> table;
    std::array<std::string, 6> headers({" name ", " total ", " total (self) ",
                                        " count ", " avg ", " avg (self) "});
    for (const auto& entry : sorted_entries) {
        table.emplace_back(std::array<std::string, 6>{
            " " + entry.name + " ",
            " " + format_duration(entry.inclusive) + " ",
            " " + format_duration(entry.exclusive) + " ",
            " " + std::to_string(entry.count) + " ",
            " " + format_avg_duration(entry.inclusive, entry.count) + " ",
            " " + format_avg_duration(entry.exclusive, entry.count) + " "});
    }
    print_table(headers, table, *output_);
}


void ProfilerHook::TableSummaryWriter::write_nested(
    const nested_summary_entry& root, std::chrono::nanoseconds overhead)
{
    (*output_) << header_ << '\n'
               << "Overhead estimate " << format_duration(overhead) << '\n';
    std::vector<std::array<std::string, 5>> table;
    std::array<std::string, 5> headers(
        {" name ", " total ", " fraction ", " count ", " avg "});
    auto visitor = [&table](auto visitor, const nested_summary_entry& node,
                            std::chrono::nanoseconds parent_elapsed,
                            std::size_t depth) -> void {
        std::vector<int64> child_permutation(node.children.size());
        const auto total_child_duration =
            std::accumulate(node.children.begin(), node.children.end(),
                            std::chrono::nanoseconds{},
                            [](std::chrono::nanoseconds acc,
                               const nested_summary_entry& child) {
                                return acc + child.elapsed;
                            });
        const auto self_duration = node.elapsed - total_child_duration;
        std::iota(child_permutation.begin(), child_permutation.end(), 0);
        std::sort(child_permutation.begin(), child_permutation.end(),
                  [&node](int64 lhs, int64 rhs) {
                      // sort by elapsed time in descending order
                      return node.children[lhs].elapsed >
                             node.children[rhs].elapsed;
                  });
        table.emplace_back(std::array<std::string, 5>{
            std::string(2 * depth + 1, ' ') + node.name + " ",
            " " + format_duration(node.elapsed) + " ",
            format_fraction(node.elapsed, parent_elapsed) + " ",
            " " + std::to_string(node.count) + " ",
            " " + format_avg_duration(node.elapsed, node.count) + " "});
        nested_summary_entry self_entry{
            "(self)", self_duration, node.count, {}};
        bool printed_self = false;
        for (const auto child_id : child_permutation) {
            // print (self) entry before smaller entry ...
            if (!printed_self &&
                node.children[child_id].elapsed < self_duration) {
                visitor(visitor, self_entry, node.elapsed, depth + 1);
                printed_self = true;
            }
            visitor(visitor, node.children[child_id], node.elapsed, depth + 1);
        }
        // ... or at the end
        if (!printed_self && !node.children.empty()) {
            visitor(visitor, self_entry, node.elapsed, depth + 1);
        }
    };
    visitor(visitor, root, root.elapsed, 0);
    print_table(headers, table, *output_);
}


}  // namespace log
}  // namespace gko
