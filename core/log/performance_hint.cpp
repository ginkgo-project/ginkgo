// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/performance_hint.hpp>


#include <iomanip>
#include <sstream>


namespace gko {
namespace log {
namespace {


std::string location_name(const uintptr& location)
{
    std::ostringstream oss;
    oss << std::hex << "0x" << location;
    return oss.str();
}


bool should_log_count(int count)
{
    return count == 10 || count == 100 || count == 1000 || count == 10000;
}


template <typename Key, typename Value>
void compact_storage(std::unordered_map<Key, Value>& map, size_type max_size,
                     size_type target_size)
{
    if (map.size() < max_size) {
        return;
    }
    // sort entries by value in descending order
    std::vector<std::pair<Key, Value>> entries;
    for (auto pair : map) {
        entries.emplace_back(pair);
    }
    GKO_ASSERT(map.size() == entries.size());
    map.clear();
    std::sort(entries.begin(), entries.end(),
              [](auto a, auto b) { return a.second > b.second; });
    GKO_ASSERT(target_size < max_size);
    GKO_ASSERT(target_size <= entries.size());
    entries.erase(entries.begin() + target_size, entries.end());
    map.insert(entries.begin(), entries.end());
}


void print_allocation_message(std::ostream& stream, size_type size, int count)
{
    stream << "Observed " << count << " allocate-free pairs of size " << size
           << " that may point to unnecessary allocations.\n";
}


void print_copy_from_message(std::ostream& stream, gko::uintptr ptr, int count)
{
    stream << "Observed " << count << " cross-executor copies from "
           << location_name(ptr)
           << " that may point to unnecessary data transfers.\n";
}


void print_copy_to_message(std::ostream& stream, gko::uintptr ptr, int count)
{
    stream << "Observed " << count << " cross-executor copies to "
           << location_name(ptr)
           << " that may point to unnecessary data transfers.\n";
}


}  // namespace


void PerformanceHint::on_allocation_completed(const Executor* exec,
                                              const size_type& num_bytes,
                                              const uintptr& location) const
{
    if (num_bytes > allocation_size_limit_) {
        allocation_sizes_[location] = num_bytes;
        // erase smallest allocations first
        compact_storage(allocation_sizes_, histogram_max_size_,
                        histogram_max_size_ * 3 / 4);
    }
}


void PerformanceHint::on_free_completed(const Executor* exec,
                                        const uintptr& location) const
{
    const auto it = allocation_sizes_.find(location);
    if (it != allocation_sizes_.end()) {
        const auto size = it->second;
        allocation_sizes_.erase(it);
        const auto count = ++allocation_histogram_[size];
        if (should_log_count(count)) {
            print_allocation_message(log(), size, count);
        }
        // erase rarest allocation sizes first
        compact_storage(allocation_histogram_, histogram_max_size_,
                        histogram_max_size_ * 3 / 4);
    }
}


void PerformanceHint::on_copy_completed(const Executor* from,
                                        const Executor* to,
                                        const uintptr& location_from,
                                        const uintptr& location_to,
                                        const size_type& num_bytes) const
{
    if (num_bytes > copy_size_limit_ && from != to) {
        const auto count1 = ++copy_src_histogram_[location_from];
        const auto count2 = ++copy_dst_histogram_[location_to];
        if (should_log_count(count1)) {
            print_copy_from_message(log(), location_from, count1);
        }
        if (should_log_count(count2)) {
            print_copy_to_message(log(), location_to, count2);
        }
        compact_storage(copy_src_histogram_, histogram_max_size_,
                        histogram_max_size_ * 3 / 4);
        compact_storage(copy_dst_histogram_, histogram_max_size_,
                        histogram_max_size_ * 3 / 4);
    }
}


void PerformanceHint::print_status() const
{
    for (auto entry : allocation_histogram_) {
        if (entry.second >= 10) {
            print_allocation_message(log(), entry.first, entry.second);
        }
    }
    for (auto entry : copy_src_histogram_) {
        if (entry.second >= 10) {
            print_copy_from_message(log(), entry.first, entry.second);
        }
    }
    for (auto entry : copy_dst_histogram_) {
        if (entry.second >= 10) {
            print_copy_to_message(log(), entry.first, entry.second);
        }
    }
}


std::ostream& PerformanceHint::log() const { return *os_ << prefix_; }


constexpr Logger::mask_type PerformanceHint::mask_;
constexpr const char* PerformanceHint::prefix_;


}  // namespace log
}  // namespace gko
