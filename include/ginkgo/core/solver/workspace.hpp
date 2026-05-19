// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_WORKSPACE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_WORKSPACE_HPP_


#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {


class LinOpFactory;


namespace solver {


namespace detail {


class SolverBaseLinOp;


/**
 * Type-erased container for any gko::array<T>. Used internally by Workspace
 * to back the workspace_array slots without templating the workspace itself
 * on a value type.
 */
class any_array {
public:
    template <typename ValueType>
    array<ValueType>& init(std::shared_ptr<const Executor> exec, size_type size)
    {
        auto container = std::make_unique<concrete_container<ValueType>>(
            std::move(exec), size);
        auto& arr = container->arr;
        data_ = std::move(container);
        return arr;
    }

    bool empty() const { return data_.get() == nullptr; }

    template <typename ValueType>
    bool contains() const
    {
        return dynamic_cast<const concrete_container<ValueType>*>(data_.get());
    }

    template <typename ValueType>
    array<ValueType>& get()
    {
        GKO_ASSERT(this->template contains<ValueType>());
        return dynamic_cast<concrete_container<ValueType>*>(data_.get())->arr;
    }

    template <typename ValueType>
    const array<ValueType>& get() const
    {
        GKO_ASSERT(this->template contains<ValueType>());
        return dynamic_cast<const concrete_container<ValueType>*>(data_.get())
            ->arr;
    }

    void clear() { data_.reset(); }

private:
    struct generic_container {
        virtual ~generic_container() = default;
    };

    template <typename ValueType>
    struct concrete_container : generic_container {
        template <typename... Args>
        concrete_container(Args&&... args) : arr{std::forward<Args>(args)...}
        {}

        array<ValueType> arr;
    };

    std::unique_ptr<generic_container> data_;
};


}  // namespace detail


/**
 * The Workspace is a node in a solver's temporary-storage tree. Each node owns
 * a flat slot container (operators and arrays sized by the solver) plus a map
 * of named child Workspaces for sub-solvers. Every node is bound to an
 * executor at construction; children inherit their parent's executor.
 *
 * Top-level workspaces are constructed via Workspace::create and passed into
 * LinOpFactory::generate(matrix, unique_ptr<Workspace>). The outer solver
 * builds the child tree as it generates inner solvers; external users only
 * ever construct and hand off a root workspace.
 *
 * One workspace per factory shape. Slot count and type are tied to a
 * particular solver class; reusing the same workspace across a Cg and then a
 * Gmres factory works but defeats the point — the second generate() truncates
 * or extends the slot vector, reallocates mismatched slots, and leaves the
 * old child subtree as dead weight. Hold one workspace per factory you want
 * to amortize allocations for.
 *
 * Not thread-safe. A workspace (or any of its descendants) must not be
 * touched by two solvers concurrently. Non-copyable: a copy would either
 * share scratch storage (unsafe) or silently produce an empty workspace
 * (misleading). Move semantics transfer the full tree.
 */
class Workspace {
public:
    explicit Workspace(std::shared_ptr<const Executor> exec)
        : exec_{std::move(exec)}
    {
        GKO_ASSERT(exec_ != nullptr);
    }

    static std::unique_ptr<Workspace> create(
        std::shared_ptr<const Executor> exec);

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&&) = default;
    Workspace& operator=(Workspace&&) = default;

    Workspace* get_or_create_child(const std::string& tag)
    {
        auto it = children_.find(tag);
        if (it != children_.end()) {
            return it->second.get();
        }
        auto child = std::unique_ptr<Workspace>(new Workspace(exec_));
        child->tag_ = tag;
        auto* ptr = child.get();
        children_.emplace(tag, std::move(child));
        return ptr;
    }

    Workspace* get_child(const std::string& tag) const
    {
        auto it = children_.find(tag);
        if (it != children_.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    bool has_child(const std::string& tag) const
    {
        return children_.find(tag) != children_.end();
    }

    /**
     * Dumps the workspace tree shape to `os`. The child tree is populated as
     * solvers generate into the workspace, so describing a freshly-created
     * (unused) workspace shows only the root.
     */
    void describe(std::ostream& os, int indent = 0) const;

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    /**
     * Rebinds the workspace to a different executor. Rejected when slots are
     * already allocated on a different executor — swapping the pointer would
     * leave the existing operator/array data stranded on the previous device.
     */
    void set_executor(std::shared_ptr<const Executor> exec)
    {
        GKO_THROW_IF_INVALID(
            exec_ == exec || this->empty(),
            "Workspace::set_executor rejected: workspace already holds "
            "allocations on a different executor. clear() the workspace or "
            "construct a fresh one for the new executor.");
        exec_ = std::move(exec);
    }

    void set_size(int num_operators, int num_arrays)
    {
        operators_.resize(num_operators);
        arrays_.resize(num_arrays);
    }

    template <typename LinOpType, typename CreateOperation>
    LinOpType* create_or_get_op(int op_id, CreateOperation create,
                                const std::type_info& expected_type,
                                dim<2> size, size_type stride)
    {
        GKO_ASSERT(op_id >= 0 && op_id < operators_.size());
        // does the existing object have the wrong type?
        // vector types may vary e.g. if users derive from Dense
        auto stored_op = operators_[op_id].get();
        LinOpType* op{};
        if (!stored_op || typeid(*stored_op) != expected_type) {
            auto new_op = create();
            op = new_op.get();
            operators_[op_id] = std::move(new_op);
            return op;
        }
        // does the existing object have the wrong dimensions?
        op = dynamic_cast<LinOpType*>(operators_[op_id].get());
        GKO_ASSERT(op);
        if (op->get_size() != size || op->get_stride() != stride) {
            auto new_op = create();
            op = new_op.get();
            operators_[op_id] = std::move(new_op);
        }
        return op;
    }

    const LinOp* get_op(int op_id) const
    {
        GKO_ASSERT(op_id >= 0 && op_id < operators_.size());
        return operators_[op_id].get();
    }

    LinOp* get_mutable_op(int op_id)
    {
        GKO_ASSERT(op_id >= 0 && op_id < operators_.size());
        return operators_[op_id].get();
    }

    template <typename ValueType>
    array<ValueType>& init_or_get_array(int array_id)
    {
        GKO_ASSERT(array_id >= 0 && array_id < arrays_.size());
        auto& array = arrays_[array_id];
        if (array.empty()) {
            auto& result =
                array.template init<ValueType>(this->get_executor(), 0);
            return result;
        }
        // array types should not change!
        GKO_ASSERT(array.template contains<ValueType>());
        return array.template get<ValueType>();
    }

    template <typename ValueType>
    array<ValueType>& create_or_get_array(int array_id, size_type size)
    {
        auto& result = init_or_get_array<ValueType>(array_id);
        if (result.get_size() != size) {
            result.resize_and_reset(size);
        }
        return result;
    }

    bool empty() const
    {
        for (const auto& op : operators_) {
            if (op) {
                return false;
            }
        }
        for (const auto& arr : arrays_) {
            if (!arr.empty()) {
                return false;
            }
        }
        return true;
    }

    void clear()
    {
        for (auto& op : operators_) {
            op.reset();
        }
        for (auto& array : arrays_) {
            array.clear();
        }
    }

private:
    std::shared_ptr<const Executor> exec_;
    std::vector<std::unique_ptr<LinOp>> operators_;
    std::vector<detail::any_array> arrays_;
    std::map<std::string, std::unique_ptr<Workspace>> children_;
    std::string tag_;
};


/**
 * Extracts the workspace from a solver, invalidating the solver.
 * The solver unique_ptr is reset to nullptr after extraction.
 * Only works on top-level solvers that own their workspace.
 *
 * @param solver  the solver to extract from (will be set to nullptr)
 * @return the extracted workspace
 * @throws InvalidStateError if the solver is not workspace-aware or has no
 *         owned workspace (e.g., it is an inner solver)
 */
std::unique_ptr<Workspace> invalidate_and_extract_workspace(
    std::unique_ptr<LinOp>& solver);


}  // namespace solver
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_SOLVER_WORKSPACE_HPP_
