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
    std::vector<partial_summary_entry> stack;
    std::unordered_map<std::string, int64> name_map;
    std::vector<summary_entry> entries;
    std::mutex mutex{};

    summary() { push("total"); }

    void push(const std::string& name, const time_point now = clock::now())
    {
        std::lock_guard<std::mutex> guard{mutex};
        auto it = name_map.find(name);
        if (it == name_map.end()) {
            const auto new_id = static_cast<int64>(entries.size());
            it = name_map.emplace_hint(it, name, new_id);
            entries.emplace_back(name);
        }
        const auto id = it->second;
        stack.emplace_back(id, now);
        overhead += clock::now() - now;
    }

    void pop(const std::string& name, const time_point now = clock::now())
    {
        std::lock_guard<std::mutex> guard{mutex};
        auto id = name_map.at(name);
        if (stack.empty() || stack.back().name_id != id) {
            throw std::logic_error{"Ranges are not nested: trying to pop " +
                                   name + " from " +
                                   entries[stack.back().name_id].name};
        }
        const auto partial_entry = stack.back();
        stack.pop_back();
        auto& entry = entries[id];
        const auto elapsed = now - partial_entry.start;
        entry.count++;
        entry.inclusive += elapsed;
        entry.exclusive += elapsed;
        if (!stack.empty()) {
            entries[stack.back().name_id].exclusive -= elapsed;
        }
        overhead += clock::now() - now;
    }

    void pop_all(std::ostream& warn_stream)
    {
        while (!stack.empty()) {
            if (stack.back().name_id != 0) {
                warn_stream << "Warning: range "
                            << entries[stack.back().name_id].name
                            << " left open\n";
            }
            pop(entries[stack.back().name_id].name);
        }
    }
};


struct nested_summary_entry {
    int64 name_id;
    int64 node_id;
    int64 parent_id;
    duration elapsed{};
    int64 count{};

    nested_summary_entry(int64 name_id, int64 node_id, int64 parent_id)
        : name_id{name_id}, node_id{node_id}, parent_id{parent_id}
    {}
};


struct partial_nested_summary_entry {
    int64 name_id;
    int64 node_id;
    time_point start;

    partial_nested_summary_entry(int64 name_id, int64 node_id, time_point start)
        : name_id{name_id}, node_id{node_id}, start{start}
    {}
};


struct pair_hash {
    int64 operator()(std::pair<int64, int64> pair) const
    {
        return pair.first ^ (pair.second << 32);
    }
};


struct nested_summary {
    duration overhead{};
    std::vector<partial_nested_summary_entry> stack;
    std::unordered_map<std::pair<int64, int64>, int64, pair_hash> node_map;
    std::unordered_map<std::string, int64> name_map;
    std::vector<nested_summary_entry> nodes;
    std::vector<std::string> names;
    std::mutex mutex{};

    nested_summary()
    {
        const auto name_id = get_or_add_name_id("total");
        const auto node_id = get_or_add_node_id(name_id);
        stack.emplace_back(name_id, node_id, clock::now());
    }

    int64 get_or_add_name_id(const std::string& name)
    {
        const auto it = name_map.find(name);
        if (it != name_map.end()) {
            return it->second;
        }
        const auto name_id = static_cast<int64>(names.size());
        name_map.emplace_hint(it, name, name_id);
        names.push_back(name);
        return name_id;
    }

    int64 get_or_add_node_id(int64 name_id)
    {
        const auto parent_id = get_parent_id();
        const auto pair = std::make_pair(name_id, parent_id);
        const auto it = node_map.find(pair);
        if (it != node_map.end()) {
            return it->second;
        }
        const auto node_id = static_cast<int64>(nodes.size());
        node_map.emplace_hint(it, pair, node_id);
        nodes.emplace_back(name_id, node_id, parent_id);
        return node_id;
    }

    int64 get_parent_id() const
    {
        return stack.empty() ? int64{-1} : stack.back().node_id;
    }

    void push(const std::string& name, const time_point now = clock::now())
    {
        std::lock_guard<std::mutex> guard{mutex};
        const auto name_id = get_or_add_name_id(name);
        const auto node_id = get_or_add_node_id(name_id);
        stack.emplace_back(name_id, node_id, clock::now());
        overhead += clock::now() - now;
    }

    void pop(const std::string& name, bool allow_pop_root = false,
             const time_point now = clock::now())
    {
        std::lock_guard<std::mutex> guard{mutex};
        auto name_id = name_map.at(name);
        if (stack.empty()) {
            throw std::logic_error{"Trying to pop " + name +
                                   " from empty stack"};
        }
        if (!allow_pop_root && stack.size() == 1) {
            throw std::logic_error{"Trying to pop " + name + " from root"};
        }
        if (stack.back().name_id != name_id) {
            throw std::logic_error{"Ranges are not nested: trying to pop " +
                                   name + " from " +
                                   names[stack.back().name_id]};
        }
        const auto partial_entry = stack.back();
        stack.pop_back();
        const auto node_id =
            node_map.at(std::make_pair(name_id, get_parent_id()));
        auto& node = nodes[node_id];
        const auto elapsed = now - partial_entry.start;
        node.count++;
        node.elapsed += elapsed;
        overhead += clock::now() - now;
    }

    void pop_all(std::ostream& warn_stream)
    {
        while (!stack.empty()) {
            if (stack.back().name_id != 0) {
                warn_stream << "Warning: range " << names[stack.back().name_id]
                            << " left open\n";
            }
            pop(names[stack.back().name_id], true);
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
        {"ns", "us", "ms", "s ", "m ", "h ", "d "}};
    auto unit =
        std::distance(ranges.begin(),
                      std::lower_bound(ranges.begin(), ranges.end(), time_ns));
    if (unit == 0) {
        ss << time_ns << ' ' << units[unit];
    } else {
        ss << (time_ns / double(ranges[unit - 1])) << ' ' << units[unit];
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


std::string fmt_fraction(duration part, duration whole)
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
            " " + entry.name + " ", " " + fmt_duration(entry.inclusive) + " ",
            " " + fmt_duration(entry.exclusive) + " ",
            " " + std::to_string(entry.count) + " ",
            " " + fmt_avg_duration(entry.inclusive, entry.count) + " ",
            " " + fmt_avg_duration(entry.exclusive, entry.count) + " "});
    }
    print_table(headers, table, stream);
}


struct nested_summary_tree {
    struct node : nested_summary_entry {
        std::string name;
        std::vector<node*> children;
        node* parent;

        node(nested_summary_entry entry, std::string name)
            : nested_summary_entry{entry},
              name{std::move(name)},
              parent{nullptr}
        {}
    };

    nested_summary_tree(nested_summary& s)
    {
        // bring group by parent node
        std::sort(s.nodes.begin(), s.nodes.end(), [](auto lhs, auto rhs) {
            return lhs.parent_id < rhs.parent_id;
        });
        std::vector<std::pair<int64, int64>> child_ranges(
            s.nodes.size(), std::make_pair(s.nodes.size(), 0));
        nodes.emplace_back(std::make_unique<node>(s.nodes[0], s.names[0]));
        for (int64 i = 1; i < s.nodes.size(); i++) {
            const auto entry = s.nodes[i];
            assert(entry.parent_id >= 0);
            auto& range = child_ranges[entry.parent_id];
            range.first = std::min(i, range.first);
            range.second = std::max(i + 1, range.second);
            nodes.push_back(
                std::make_unique<node>(entry, s.names[entry.name_id]));
        }
        for (int64 i = 0; i < s.nodes.size(); i++) {
            auto cur_node = nodes[i].get();
            const auto child_range = child_ranges[cur_node->node_id];
            auto self_duration = cur_node->elapsed;
            for (auto c = child_range.first; c < child_range.second; c++) {
                auto& child_node = nodes[c];
                cur_node->children.push_back(child_node.get());
                self_duration -= child_node->elapsed;
                child_node->parent = cur_node;
            }
            if (!cur_node->children.empty()) {
                nested_summary_entry self_entry{-1, -1, cur_node->node_id};
                self_entry.elapsed = self_duration;
                self_entry.count = cur_node->count;
                nodes.push_back(std::make_unique<node>(self_entry, "(self)"));
                nodes.back()->parent = cur_node;
                cur_node->children.push_back(nodes.back().get());
            }
            std::sort(cur_node->children.begin(), cur_node->children.end(),
                      [](const node* lhs, const node* rhs) {
                          // sort descending by inclusive time
                          return lhs->elapsed > rhs->elapsed;
                      });
        }
    }

    std::vector<std::unique_ptr<node>> nodes;
};


void print_summary(std::ostream& stream, const std::string& name,
                   nested_summary& s)
{
    stream << name << '\n'
           << "Overhead estimate " << fmt_duration(s.overhead) << '\n';
    s.pop_all(stream);

    std::vector<std::array<std::string, 5>> table;
    std::array<std::string, 5> headers(
        {" name ", " total ", " fraction ", " count ", " avg "});
    auto visitor = [&table](auto visitor, const nested_summary_tree::node* node,
                            std::size_t depth) -> void {
        table.emplace_back(std::array<std::string, 5>{
            std::string(2 * depth + 1, ' ') + node->name + " ",
            " " + fmt_duration(node->elapsed) + " ",
            node->parent
                ? (" " + fmt_fraction(node->elapsed, node->parent->elapsed) +
                   " ")
                : std::string{""},
            " " + std::to_string(node->count) + " ",
            " " + fmt_avg_duration(node->elapsed, node->count) + " "});
        for (const auto* child : node->children) {
            visitor(visitor, child, depth + 1);
        }
    };
    nested_summary_tree tree{s};
    visitor(visitor, tree.nodes[0].get(), 0);
    print_table(headers, table, stream);
}


}  // namespace profiler_hook


std::pair<ProfilerHook::hook_function, ProfilerHook::hook_function>
create_summary_fns(std::ostream& stream, std::string name, bool nested)
{
    if (nested) {
        std::shared_ptr<profiler_hook::nested_summary> data{
            new profiler_hook::nested_summary{},
            [&stream, name](profiler_hook::nested_summary* ptr) {
                print_summary(stream, name, *ptr);
                delete ptr;
            }};
        return std::make_pair(
            [data](const char* name, profile_event_category) {
                data->push(name);
            },
            [data](const char* name, profile_event_category) {
                data->pop(name);
            });
    } else {
        std::shared_ptr<profiler_hook::summary> data{
            new profiler_hook::summary{},
            [&stream, name](profiler_hook::summary* ptr) {
                print_summary(stream, name, *ptr);
                delete ptr;
            }};
        return std::make_pair(
            [data](const char* name, profile_event_category) {
                data->push(name);
            },
            [data](const char* name, profile_event_category) {
                data->pop(name);
            });
    }
}


}  // namespace log
}  // namespace gko
