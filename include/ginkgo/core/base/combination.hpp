// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_COMBINATION_HPP_
#define GKO_PUBLIC_CORE_BASE_COMBINATION_HPP_


#include <vector>


#include <ginkgo/core/base/lin_op.hpp>


namespace gko {


/**
 * The Combination class can be used to construct a linear combination of
 * multiple linear operators `c1 * op1 + c2 * op2 + ... + ck * opk`.
 *
 * Combination ensures that all LinOps passed to its constructor use the same
 * executor, and if not, copies the operators to the executor of the first
 * operator.
 *
 * @tparam ValueType  precision of input and result vectors
 *
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Combination : public EnableLinOp<Combination<ValueType>>,
                    public EnableCreateMethod<Combination<ValueType>>,
                    public Transposable {
    friend class EnablePolymorphicObject<Combination, LinOp>;
    friend class EnableCreateMethod<Combination>;

public:
    using value_type = ValueType;
    using transposed_type = Combination<ValueType>;

    /**
     * Returns a list of coefficients of the combination.
     *
     * @return a list of coefficients
     */
    const std::vector<std::shared_ptr<const LinOp>>& get_coefficients()
        const noexcept
    {
        return coefficients_;
    }

    /**
     * Returns a list of operators of the combination.
     *
     * @return a list of operators
     */
    const std::vector<std::shared_ptr<const LinOp>>& get_operators()
        const noexcept
    {
        return operators_;
    }

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Copy-assigns a Combination. The executor is not modified, and the
     * wrapped LinOps are only being cloned if they are on a different executor.
     */
    Combination& operator=(const Combination&);

    /**
     * Move-assigns a Combination. The executor is not modified, and the
     * wrapped LinOps are only being cloned if they are on a different executor,
     * otherwise they share ownership. The moved-from object is empty (0x0 LinOp
     * without operators) afterwards.
     */
    Combination& operator=(Combination&&);

    /**
     * Copy-constructs a Combination. This inherits the executor of the input
     * Combination and all of its operators with shared ownership.
     */
    Combination(const Combination&);

    /**
     * Move-constructs a Combination. This inherits the executor of the input
     * Combination and all of its operators. The moved-from object is empty (0x0
     * LinOp without operators) afterwards.
     */
    Combination(Combination&&);

protected:
    void add_operators() {}

    template <typename... Rest>
    void add_operators(std::shared_ptr<const LinOp> coef,
                       std::shared_ptr<const LinOp> oper, Rest&&... rest)
    {
        GKO_ASSERT_EQUAL_DIMENSIONS(coef, dim<2>(1, 1));
        GKO_ASSERT_EQUAL_DIMENSIONS(oper, this->get_size());
        auto exec = this->get_executor();
        coefficients_.push_back(std::move(coef));
        operators_.push_back(std::move(oper));
        if (coefficients_.back()->get_executor() != exec) {
            coefficients_.back() = gko::clone(exec, coefficients_.back());
        }
        if (operators_.back()->get_executor() != exec) {
            operators_.back() = gko::clone(exec, operators_.back());
        }
        add_operators(std::forward<Rest>(rest)...);
    }

    /**
     * Creates an empty linear combination (0x0 operator).
     *
     * @param exec  Executor associated to the linear combination
     */
    explicit Combination(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Combination>(exec)
    {}

    /**
     * Creates a linear combination of operators using the specified list of
     * coefficients and operators.
     *
     * @tparam CoefficientIterator  a class representing iterators over the
     *                              coefficients of the linear combination
     * @tparam OperatorIterator  a class representing iterators over the
     *                           operators of the linear combination
     *
     * @param coefficient_begin  iterator pointing to the first coefficient
     * @param coefficient_end  iterator pointing behind the last coefficient
     * @param operator_begin  iterator pointing to the first operator
     * @param operator_end  iterator pointing behind the last operator
     */
    template <
        typename CoefficientIterator, typename OperatorIterator,
        typename = xstd::void_t<
            typename std::iterator_traits<
                CoefficientIterator>::iterator_category,
            typename std::iterator_traits<OperatorIterator>::iterator_category>>
    explicit Combination(CoefficientIterator coefficient_begin,
                         CoefficientIterator coefficient_end,
                         OperatorIterator operator_begin,
                         OperatorIterator operator_end)
        : EnableLinOp<Combination>([&] {
              if (operator_begin == operator_end) {
                  throw OutOfBoundsError(__FILE__, __LINE__, 1, 0);
              }
              return (*operator_begin)->get_executor();
          }())
    {
        GKO_ASSERT_EQ(std::distance(coefficient_begin, coefficient_end),
                      std::distance(operator_begin, operator_end));
        this->set_size((*operator_begin)->get_size());
        auto coefficient_it = coefficient_begin;
        for (auto operator_it = operator_begin; operator_it != operator_end;
             ++operator_it) {
            add_operators(*coefficient_it, *operator_it);
            ++coefficient_it;
        }
    }

    /**
     * Creates a linear combination of operators using the specified list of
     * coefficients and operators.
     *
     * @tparam Rest  types of trailing parameters
     *
     * @param coef  the first coefficient
     * @param oper  the first operator
     * @param rest  other coefficient and operators (interleaved)
     */
    template <typename... Rest>
    explicit Combination(std::shared_ptr<const LinOp> coef,
                         std::shared_ptr<const LinOp> oper, Rest&&... rest)
        : Combination(oper->get_executor())
    {
        this->set_size(oper->get_size());
        add_operators(std::move(coef), std::move(oper),
                      std::forward<Rest>(rest)...);
    }

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    std::vector<std::shared_ptr<const LinOp>> coefficients_;
    std::vector<std::shared_ptr<const LinOp>> operators_;

    // TODO: solve race conditions when multithreading
    mutable struct cache_struct {
        cache_struct() = default;
        ~cache_struct() = default;
        cache_struct(const cache_struct& other) {}
        cache_struct& operator=(const cache_struct& other) { return *this; }

        std::unique_ptr<LinOp> zero;
        std::unique_ptr<LinOp> one;
        std::unique_ptr<LinOp> intermediate_x;
    } cache_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_COMBINATION_HPP_
