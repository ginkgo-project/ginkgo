#ifndef GKO_CORE_BASE_CONVERTIBLE_HPP_
#define GKO_CORE_BASE_CONVERTIBLE_HPP_


namespace gko {


template <typename ResultType>
class ConvertibleTo {
public:
    using result_type = ResultType;

    virtual void convert_to(result_type *result) const = 0;
    virtual void move_to(result_type *result) = 0;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_CONVERTIBLE_HPP_
