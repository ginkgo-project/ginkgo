// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_WORKSPACE_HPP_
#define GKO_PUBLIC_CORE_BASE_WORKSPACE_HPP_


#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


class LinOpFactory;
class Workspace;


namespace detail {


/**
 * Prepares `node` for its next user, a LinOp of type `user`. Slot meanings
 * are defined per type, so if a LinOp of another type set the node up last,
 * its operators and arrays are cleared first. Its children are left alone;
 * their own users claim them.
 */
void claim_workspace(Workspace& node, const std::type_info& user);


class WorkspaceHolder;


// whether VectorType is distributed, i.e. has a local vector
template <typename VectorType, typename = void>
struct has_local_vector : std::false_type {};

template <typename VectorType>
struct has_local_vector<
    VectorType,
    std::void_t<decltype(std::declval<const VectorType&>().get_local_vector())>>
    : std::true_type {};


// the size of the part of `vec` this process stores
template <typename VectorType>
dim<2> get_local_size(const VectorType& vec)
{
    if constexpr (has_local_vector<VectorType>::value) {
        return vec.get_local_vector()->get_size();
    } else {
        return vec.get_size();
    }
}


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
 * executor at construction; children initially inherit their parent's
 * executor, and an inner solver may rebind its child node via reset(exec)
 * (used by Multigrid to bind each level to its level executor).
 *
 * Top-level workspaces are constructed via Workspace::create and passed into
 * LinOpFactory::generate(matrix, unique_ptr<Workspace>). The outer solver
 * builds the child tree as it generates inner solvers; external users only
 * ever construct and hand off a root workspace.
 *
 * One workspace per factory shape. Slot meanings are tied to a particular
 * LinOp type, so a node used by a LinOp of a different type than before is
 * cleared first: reusing a workspace across, e.g., a Cg and then a Gmres
 * factory is correct, but allocates again. The old child subtree is kept.
 * Hold one workspace per factory you want to amortize allocations for.
 *
 * Not thread-safe: A workspace (or any of its descendants) must not be
 * used by two solvers concurrently.
 *
 * Non-copyable: a copy would either share scratch storage (unsafe) or
 * silently produce an empty workspace (misleading).
 *
 * Move semantics transfer the full tree.
 *
 * Child nodes are shared: a LinOp borrowing a node keeps it alive, so inner
 * LinOps stay usable after their outer solver is destroyed or its workspace
 * extracted. After an extraction they share their nodes with the next LinOp
 * generated from the workspace and must not be applied concurrently with it.
 */
class Workspace : public std::enable_shared_from_this<Workspace> {
public:
    static std::unique_ptr<Workspace> create(
        std::shared_ptr<const Executor> exec);

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&&) = default;
    Workspace& operator=(Workspace&&) = default;

    Workspace* get_or_create_child(const std::string& tag);

    Workspace* get_child(const std::string& tag) const;

    bool has_child(const std::string& tag) const;

    /**
     * Dumps the workspace tree shape to `os`. The child tree is populated as
     * solvers generate into the workspace, so describing a freshly-created
     * (unused) workspace shows only the root.
     */
    void describe(std::ostream& os, int indent = 0) const;

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    /**
     * Rebinds the workspace to a different executor. When the new executor
     * differs from the current one, existing slot data would be stranded on
     * the previous device, so the workspace is cleared before rebinding.
     * When the executor is unchanged, this is a no-op and slot contents are
     * preserved.
     */
    void reset(std::shared_ptr<const Executor> exec);

    void set_size(int num_operators, int num_arrays);

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

    const LinOp* get_const_op(int op_id) const;

    LinOp* get_mutable_op(int op_id);

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

    bool empty() const;

    void clear();

private:
    // use Workspace::create
    explicit Workspace(std::shared_ptr<const Executor> exec);

    friend void detail::claim_workspace(Workspace&, const std::type_info&);
    friend class detail::WorkspaceHolder;

    // replaces operator slot `op_id`
    void replace_op(int op_id, std::unique_ptr<LinOp> op);

    std::shared_ptr<const Executor> exec_;
    std::vector<std::unique_ptr<LinOp>> operators_;
    std::vector<detail::any_array> arrays_;
    std::map<std::string, std::shared_ptr<Workspace>> children_;
    std::string tag_;
    // the type of the LinOp that set up the slots last
    std::type_index user_type_{typeid(void)};
};


/**
 * Extracts the workspace from a LinOp that owns it (a top-level solver or
 * preconditioner), invalidating the LinOp. The unique_ptr is consumed and reset
 * to nullptr after extraction. Pass with std::move at the call site so the
 * consumption is explicit.
 *
 * @param op  the LinOp to extract from (consumed by rvalue reference; reset to
 *            nullptr on return)
 *
 * @return the extracted workspace
 *
 * @throws InvalidStateError if the LinOp does not keep a workspace or does not
 *         own it (e.g., it is an inner solver)
 */
std::unique_ptr<Workspace> invalidate_and_extract_workspace(
    std::unique_ptr<LinOp>&& op);


namespace detail {


/**
 * Base of every LinOp that keeps temporaries in a workspace node. The node is
 * either owned (a root, for a top-level LinOp) or borrowed (a child node of
 * an outer LinOp's workspace). An owned workspace is created on first use.
 *
 * Copies get a fresh, empty workspace, since copies must not share scratch
 * storage; moves transfer it. LinOps generated by a factory pass their
 * LinOpGenerateComponents to the constructor, which takes over the owned
 * workspace or borrows the view they carry.
 */
class WorkspaceHolder {
public:
    virtual ~WorkspaceHolder() = default;

    WorkspaceHolder(const WorkspaceHolder& other);

    WorkspaceHolder(WorkspaceHolder&& other) noexcept;

    WorkspaceHolder& operator=(const WorkspaceHolder& other);

    WorkspaceHolder& operator=(WorkspaceHolder&& other) noexcept;

protected:
    explicit WorkspaceHolder(std::shared_ptr<const Executor> exec);

    WorkspaceHolder(std::shared_ptr<const Executor> exec,
                    LinOpGenerateComponents& components);

    /**
     * Returns the workspace node, creating an owned one on first use.
     *
     * @throws InvalidStateError if the workspace was extracted
     */
    Workspace* get_workspace_node() const;

    /** Claims the node for this LinOp's type and sizes its slots. */
    void set_workspace_size(int num_operators, int num_arrays) const;

    template <typename LinOpType>
    LinOpType* create_workspace_op(int vector_id, gko::dim<2> size) const
    {
        auto node = this->get_workspace_node();
        return node->template create_or_get_op<LinOpType>(
            vector_id,
            [&] { return LinOpType::create(node->get_executor(), size); },
            typeid(LinOpType), size, size[1]);
    }

    template <typename LinOpType>
    LinOpType* create_workspace_op_with_config_of(int vector_id,
                                                  const LinOpType* vec) const
    {
        // distributed vectors of the same global size may be partitioned
        // differently, so the local size is compared as well
        return this->template create_workspace_op_if<LinOpType>(
            vector_id,
            [vec](const LinOpType& op) {
                return typeid(op) != typeid(*vec) ||
                       op.get_size() != vec->get_size() ||
                       op.get_stride() != vec->get_stride() ||
                       get_local_size(op) != get_local_size(*vec);
            },
            [vec] { return LinOpType::create_with_config_of(vec); });
    }

    template <typename LinOpType>
    LinOpType* create_workspace_op_with_type_of(int vector_id,
                                                const LinOpType* vec,
                                                dim<2> size) const
    {
        auto node = this->get_workspace_node();
        return node->template create_or_get_op<LinOpType>(
            vector_id,
            [&] {
                return LinOpType::create_with_type_of(vec, node->get_executor(),
                                                      size, size[1]);
            },
            typeid(*vec), size, size[1]);
    }

    template <typename LinOpType>
    LinOpType* create_workspace_op_with_type_of(int vector_id,
                                                const LinOpType* vec,
                                                dim<2> global_size,
                                                dim<2> local_size) const
    {
        auto node = this->get_workspace_node();
        // the local size changes when the same global size is partitioned
        // differently
        return this->template create_workspace_op_if<LinOpType>(
            vector_id,
            [&](const LinOpType& op) {
                return typeid(op) != typeid(*vec) ||
                       op.get_size() != global_size ||
                       op.get_stride() != local_size[1] ||
                       get_local_size(op) != local_size;
            },
            [&] {
                return LinOpType::create_with_type_of(vec, node->get_executor(),
                                                      global_size, local_size,
                                                      local_size[1]);
            });
    }

    /**
     * Returns operator slot `vector_id` as a LinOpType, recreated with
     * `create()` if it is empty, of another type, or `needs_new(op)` holds.
     */
    template <typename LinOpType, typename NeedsNew, typename Create>
    LinOpType* create_workspace_op_if(int vector_id, NeedsNew needs_new,
                                      Create create) const
    {
        auto node = this->get_workspace_node();
        auto op = dynamic_cast<LinOpType*>(node->get_mutable_op(vector_id));
        if (!op || needs_new(*op)) {
            auto new_op = create();
            op = new_op.get();
            node->replace_op(vector_id, std::move(new_op));
        }
        return op;
    }

    template <typename ValueType>
    matrix::Dense<ValueType>* create_workspace_scalar(int vector_id,
                                                      size_type size) const
    {
        auto node = this->get_workspace_node();
        return node->template create_or_get_op<matrix::Dense<ValueType>>(
            vector_id,
            [&] {
                return matrix::Dense<ValueType>::create(node->get_executor(),
                                                        dim<2>{1, size});
            },
            typeid(matrix::Dense<ValueType>), gko::dim<2>{1, size}, size);
    }

    template <typename ValueType>
    const matrix::Dense<ValueType>* create_workspace_fixed_scalar(
        int vector_id, size_type size, ValueType val) const
    {
        auto node = this->get_workspace_node();
        return node->template create_or_get_op<matrix::Dense<ValueType>>(
            vector_id,
            [&] {
                auto mat = matrix::Dense<ValueType>::create(
                    node->get_executor(), dim<2>{1, size});
                mat->fill(val);
                return mat;
            },
            typeid(matrix::Dense<ValueType>), gko::dim<2>{1, size}, size);
    }

    template <typename ValueType>
    array<ValueType>& create_workspace_array(int array_id, size_type size) const
    {
        return this->get_workspace_node()
            ->template create_or_get_array<ValueType>(array_id, size);
    }

    template <typename ValueType>
    array<ValueType>& create_workspace_array(int array_id) const
    {
        return this->get_workspace_node()
            ->template init_or_get_array<ValueType>(array_id);
    }

private:
    friend std::unique_ptr<Workspace> gko::invalidate_and_extract_workspace(
        std::unique_ptr<LinOp>&& op);
    friend void attach_workspace(LinOp* op, Workspace* node);
    friend void give_workspace(LinOp* op, std::unique_ptr<Workspace> root);

    void own(std::unique_ptr<Workspace> root);

    void borrow(Workspace* node);

    // the owned root, or nullptr when borrowing or after an extraction
    std::unique_ptr<Workspace> extract_workspace();

    std::shared_ptr<const Executor> exec_;
    mutable std::unique_ptr<Workspace> owned_;
    std::shared_ptr<Workspace> borrowed_;
    mutable Workspace* node_ = nullptr;
    bool extracted_ = false;
};


/**
 * Internal: makes `op` borrow the child node `node`, if `op` keeps a
 * workspace. Used by LinOps that create inner LinOps with create().
 */
void attach_workspace(LinOp* op, Workspace* node);


/**
 * Internal: makes `op` own the root workspace `root`, if `op` keeps a
 * workspace; otherwise `root` is destroyed.
 */
void give_workspace(LinOp* op, std::unique_ptr<Workspace> root);


}  // namespace detail


}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_WORKSPACE_HPP_
