#pragma once

template <typename Scalar>
class ImplicitFunction
{
public:
    virtual Scalar evaluate(Scalar x, Scalar y, Scalar z) const = 0;
    virtual Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const = 0;
    virtual ~ImplicitFunction() {}
};