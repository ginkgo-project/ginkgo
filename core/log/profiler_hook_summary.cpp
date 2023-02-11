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
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <vector>


#include "core/log/profiler_hook.hpp"


namespace gko {
namespace log {
namespace {


using clock = std::chrono::steady_clock;
using duration = clock::duration;
using time_point = clock::time_point;


template <typename Summary>
bool check_pop_status(Summary& s, const char* name, bool allow_pop_root)
{
    if (s.broken) {
        return false;
    }
    if (s.stack.empty()) {
#if GKO_VERBOSE_LEVEL >= 1
        std::cerr
            << "WARNING: Popping from an empty stack in summary "
               "gko::log::ProfilerHook.\nThis probably means the "
               "logger was created inside a Ginkgo operation but destroyed "
               "outside.\nTo fix this, move the logger creation to "
               "the outermost scope where Ginkgo is used!\nThe profiler "
               "output will most likely be incorrect.\nThe last operation was "
               "pop("
            << name << ")\n";
#endif
        s.broken = true;
        return false;
    }
    if (s.stack.size() == 1 && !allow_pop_root) {
#if GKO_VERBOSE_LEVEL >= 1
        std::cerr
            << "WARNING: Popping the root element during execution in summary "
               "gko::log::ProfilerHook.\nThis probably means the "
               "logger was created inside a Ginkgo operation but destroyed "
               "outside.\nTo fix this, move the logger creation to "
               "the outermost scope where Ginkgo is used!\nThe profiler "
               "output will most likely be incorrect.\nThe last operation was "
               "pop("
            << name << ")\n";
#endif
        s.broken = true;
        return false;
    }
    if (s.check_nesting && s.get_top_name() != name) {
#if GKO_VERBOSE_LEVEL >= 1
        std::cerr << "WARNING: Incorrect nesting in summary "
                     "gko::log::ProfilerHook.\nThis points to incorrect use of "
                     "logger events, the performance output will not be "
                     "correct.\nThe mismatching pair was push("
                  << s.get_top_name() << ") and pop(" << name << ")\n";
#endif
        s.broken = true;
        return false;
    }
    return true;
}

template <typename Summary>
void pop_all(Summary& s)
{
    const auto stack_size = s.stack.size();
    if (stack_size > 1 && !s.broken) {
#if GKO_VERBOSE_LEVEL >= 1
        std::cerr
            << "WARNING: Unfinished events remaining in summary "
               "gko::log::ProfilerHook.\nThis probably means the "
               "logger was created outside a Ginkgo operation but removed "
               "and destroyed inside.\nTo fix this, move the logger "
               "creation to the outermost scope where Ginkgo is used!\nThe "
               "profiler output will most likely be incorrect.\n";
#endif
    }
    // just to be sure, since pop can be no-op, use fixed-length loop
    for (std::size_t i = 0; i < stack_size; i++) {
        if (i < stack_size - 1 && !s.broken) {
#if GKO_VERBOSE_LEVEL >= 1
            std::cerr << "Popping unfinished event \"" << s.get_top_name()
                      << "\"\n";
#endif
        }
        s.pop(s.get_top_name().c_str(), true);
    }
}


struct summary {
    int64 overhead_ns{};
    std::vector<std::pair<int64, time_point>> stack;
    std::unordered_map<std::string, int64> name_map;
    std::vector<ProfilerHook::summary_entry> entries;
    bool broken{};
    bool check_nesting{};
    std::mutex mutex{};

    summary() { push("total"); }

    void push(const char* name)
    {
        if (broken) {
            return;
        }
        const auto now = clock::now();
        std::lock_guard<std::mutex> guard{mutex};
        auto it = name_map.find(name);
        if (it == name_map.end()) {
            const auto new_id = static_cast<int64>(entries.size());
            it = name_map.emplace_hint(it, name, new_id);
            entries.emplace_back();
            entries.back().name = name;
        }
        const auto id = it->second;
        stack.emplace_back(id, now);
        overhead_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds, int64>(
                clock::now() - now)
                .count();
    }

    void pop(const char* name, bool allow_pop_root = false)
    {
        const auto now = clock::now();
        std::lock_guard<std::mutex> guard{mutex};
        if (!check_pop_status(*this, name, allow_pop_root)) {
            return;
        }
        const auto id = stack.back().first;
        const auto partial_entry = stack.back();
        stack.pop_back();
        auto& entry = entries[id];
        const auto elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds, int64>(
                now - partial_entry.second)
                .count();
        entry.count++;
        entry.inclusive_ns += elapsed_ns;
        entry.exclusive_ns += elapsed_ns;
        if (!stack.empty()) {
            entries[stack.back().first].exclusive_ns -= elapsed_ns;
        }
        overhead_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds, int64>(
                clock::now() - now)
                .count();
    }

    const std::string& get_top_name() const
    {
        return entries[stack.back().first].name;
    }
};


struct nested_summary {
    struct entry {
        int64 name_id;
        int64 node_id;
        int64 parent_id;
        int64 elapsed_ns{};
        int64 count{};

        entry(int64 name_id, int64 node_id, int64 parent_id)
            : name_id{name_id}, node_id{node_id}, parent_id{parent_id}
        {}
    };

    struct partial_entry {
        int64 name_id;
        int64 node_id;
        time_point start;

        partial_entry(int64 name_id, int64 node_id, time_point start)
            : name_id{name_id}, node_id{node_id}, start{start}
        {}
    };

    struct pair_hash {
        int64 operator()(std::pair<int64, int64> pair) const
        {
            return pair.first ^ (pair.second << 32);
        }
    };

    int64 overhead_ns{};
    std::vector<partial_entry> stack;
    std::unordered_map<std::pair<int64, int64>, int64, pair_hash> node_map;
    std::unordered_map<std::string, int64> name_map;
    std::vector<entry> nodes;
    std::vector<std::string> names;
    bool broken{};
    bool check_nesting{};
    std::mutex mutex{};

    nested_summary() { push("total"); }

    int64 get_or_add_name_id(const char* name)
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

    void push(const char* name)
    {
        if (broken) {
            return;
        }
        const auto now = clock::now();
        std::lock_guard<std::mutex> guard{mutex};
        const auto name_id = get_or_add_name_id(name);
        const auto node_id = get_or_add_node_id(name_id);
        stack.emplace_back(name_id, node_id, clock::now());
        overhead_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds, int64>(
                clock::now() - now)
                .count();
    }

    void pop(const char* name, bool allow_pop_root = false)
    {
        const auto now = clock::now();
        std::lock_guard<std::mutex> guard{mutex};
        if (!check_pop_status(*this, name, allow_pop_root)) {
            return;
        }
        const auto partial_entry = stack.back();
        const auto name_id = partial_entry.name_id;
        stack.pop_back();
        const auto node_id =
            node_map.at(std::make_pair(name_id, get_parent_id()));
        auto& node = nodes[node_id];
        const auto elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds, int64>(
                now - partial_entry.start)
                .count();
        node.count++;
        node.elapsed_ns += elapsed_ns;
        overhead_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds, int64>(
                clock::now() - now)
                .count();
    }

    const std::string& get_top_name() const
    {
        return names[stack.back().name_id];
    }
};


ProfilerHook::nested_summary_entry build_tree(const nested_summary& summary)
{
    ProfilerHook::nested_summary_entry root;
    const auto num_nodes = summary.nodes.size();
    std::vector<int64> node_permutation(num_nodes);
    // compute permutation to group by parent node
    std::iota(node_permutation.begin(), node_permutation.end(), 0);
    std::stable_sort(node_permutation.begin(), node_permutation.end(),
                     [&summary](int64 lhs, int64 rhs) {
                         return summary.nodes[lhs].parent_id <
                                summary.nodes[rhs].parent_id;
                     });
    // now ever node's children can be represented by a range in the permutation
    // this vector thus maps from node IDs to index ranges in node_permutation
    std::vector<std::pair<int64, int64>> child_ranges(
        num_nodes, std::make_pair(num_nodes, 0));
    for (int64 i = 0; i < num_nodes; i++) {
        const auto entry = summary.nodes[node_permutation[i]];
        if (entry.parent_id >= 0) {
            auto& range = child_ranges[entry.parent_id];
            range.first = std::min(i, range.first);
            range.second = std::max(i + 1, range.second);
        }
    }
    // now we recursively visit children of the root, collecting them in the
    // output nested_summary_entry
    auto visit_node = [&](auto visit_node, int64 permuted_id,
                          ProfilerHook::nested_summary_entry& entry) -> void {
        const auto& summary_node = summary.nodes[node_permutation[permuted_id]];
        entry.name = summary.names[summary_node.name_id];
        entry.elapsed_ns = summary_node.elapsed_ns;
        entry.count = summary_node.count;
        const auto child_range = child_ranges[summary_node.node_id];
        for (auto i = child_range.first; i < child_range.second; i++) {
            entry.children.emplace_back();
            visit_node(visit_node, i, entry.children.back());
        }
    };
    visit_node(visit_node, 0, root);
    return root;
}


}  // namespace


std::shared_ptr<ProfilerHook> ProfilerHook::create_summary(
    std::unique_ptr<summary_writer> writer, bool debug_check_nesting)
{
    // we need to wrap the deleter in a shared_ptr to deal with a GCC 5.5 bug
    // related to move-only functors
    std::shared_ptr<summary> data{
        new summary{}, [writer = std::shared_ptr<summary_writer>{
                            std::move(writer)}](summary* ptr) {
            // clean up open ranges
            pop_all(*ptr);
            writer->write(ptr->entries, ptr->overhead_ns);
            delete ptr;
        }};
    data->check_nesting = debug_check_nesting;
    return std::shared_ptr<ProfilerHook>{new ProfilerHook{
        [data](const char* name, profile_event_category) { data->push(name); },
        [data](const char* name, profile_event_category) { data->pop(name); }}};
}


std::shared_ptr<ProfilerHook> ProfilerHook::create_nested_summary(
    std::unique_ptr<nested_summary_writer> writer, bool debug_check_nesting)
{
    // we need to wrap the deleter in a shared_ptr to deal with a GCC 5.5 bug
    // related to move-only functors
    std::shared_ptr<nested_summary> data{
        new nested_summary{}, [writer = std::shared_ptr<nested_summary_writer>{
                                   std::move(writer)}](nested_summary* ptr) {
            // clean up open ranges
            pop_all(*ptr);
            writer->write_nested(build_tree(*ptr), ptr->overhead_ns);
            delete ptr;
        }};
    data->check_nesting = debug_check_nesting;
    return std::shared_ptr<ProfilerHook>{new ProfilerHook{
        [data](const char* name, profile_event_category) { data->push(name); },
        [data](const char* name, profile_event_category) { data->pop(name); }}};
}


}  // namespace log
}  // namespace gko
