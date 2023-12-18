// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/memory_footprint.hpp>


#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace log {


void MemoryFootprint::on_allocation_completed(const Executor* exec,
                                              const size_type& num_bytes,
                                              const uintptr& location) const
{
    StackCapture::on_allocation_completed(exec, num_bytes, location);

    current_total_ += num_bytes;
    if (num_bytes > im_data_.largest) {
        im_data_.largest = num_bytes;
        im_data_.largest_stack = this->get_stack();
    }
    if (current_total_ > im_data_.largest) {
        im_data_.max_total = current_total_;
        im_data_.max_total_stack = this->get_stack();
    }
    im_data_.timeline.push_back(current_total_);
    allocated_bytes[location] = num_bytes;
}


void MemoryFootprint::on_free_completed(const Executor* exec,
                                        const uintptr& location) const
{
    current_total_ -= allocated_bytes.at(location);
    allocated_bytes.erase(location);

    StackCapture::on_free_completed(exec, location);
}


MemoryFootprint::data MemoryFootprint::get_data() const
{
#if GKO_VERBOSE_LEVEL >= 1
    if (!this->get_stack().empty()) {
        std::cerr << "The Ginkgo stack is not empty, so the results might not "
                  << "be complete. For best results, please do not call "
                  << "MemoryFootprint::get_data within Ginkgo functionality."
                  << std::endl;
    }
#endif
    return im_data_.create_data(this);
}


MemoryFootprint::data MemoryFootprint::intermediate_data::create_data(
    const MemoryFootprint* mt) const
{
    auto complete_stack = [&](const std::vector<int64>& stack) {
        std::vector<std::string> named_stack;
        named_stack.reserve(stack.size());
        for (auto id : stack) {
            named_stack.push_back(mt->get_inverse_name_map().at(id));
        }
        return named_stack;
    };
    return {max_total, largest, timeline, complete_stack(max_total_stack),
            complete_stack(largest_stack)};
}


}  // namespace log
}  // namespace gko
