#ifndef GKO_CORE_BASE_CONVERTIBLE_HPP_
#define GKO_CORE_BASE_CONVERTIBLE_HPP_


namespace gko {


/**
 * ConvertibleTo interface is used to mark that the implementer can be converted
 * to the object of ResultType.
 *
 * This interface is used to enable conversions between different LinOp types.
 * Let `A` and `B` be two LinOp subclasses. To mark that objects of type `A` can
 * be converted to objects of type `B`, `A` should implement the
 * `ConvertibleTo\<B\>` interface. Then the following code:
 *
 * ```c++
 * auto a = A::create(...);
 * auto b = B::create(...);
 *
 * b->copy_from(a.get());
 *
 * ```
 *
 * will convert object `a` to object `b` as follows:
 *
 * 1.  `B::copy_from()` checks that LinOp `a` can be dynamically casted to
 *     `ConvertibleTo\<B\>` (i.e. a can be converted to type `B`)
 * 2.  `B::copy_from()` internally calls `ConvertibleTo\<B\>::convert_to(B*)`
 *     on `a` to complete the conversion.
 *
 * In case `a` is passed in as a unique_ptr (i.e. using
 * `b->copy_from(std::move(a));`) call to `convert_to` will be replaced by a
 * call to `move_to` and trigger move semantics.
 *
 * @tparam ResultType  the type to which the implementer can be converted to
 */
template <typename ResultType>
class ConvertibleTo {
public:
    using result_type = ResultType;

    virtual ~ConvertibleTo() = default;

    /**
     * Converts the implementer to an object of type result_type.
     *
     * @param result  the object used to emplace the result of the conversion
     */
    virtual void convert_to(result_type *result) const = 0;

    /**
     * Converts the implementer to an object of type result_type, by moving
     * data.
     *
     * This method is used when the implementer is a temporary object, and move
     * semantics can be used.
     *
     * @param result  the object used to emplace the result of the conversion
     *
     * @note ConvertibleTo::move_to can be implemented by simply
     *       ConvertibleTo::convert_to. However, this operation can often be
     *       optimized by exploiting the fact that implementer's data can be
     *       moved to the result.
     */
    virtual void move_to(result_type *result) = 0;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_CONVERTIBLE_HPP_
