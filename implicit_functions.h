#pragma once

#include <string>
#include <array>
#include <Eigen/Core>
#include "vector_math.h"

template <typename Scalar>
class ImplicitFunction
{
public:
    virtual Scalar evaluate(Scalar x, Scalar y, Scalar z) const = 0;
    virtual Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const = 0;
    virtual ~ImplicitFunction() {}
};

template <typename Scalar>
class GeneralFunction : public ImplicitFunction<Scalar>
{
public:
    GeneralFunction(
        std::function<Scalar(Scalar, Scalar, Scalar, Scalar &, Scalar &, Scalar &)> func_grad) : func_grad_(func_grad) {}

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        Scalar gx, gy, gz;
        return func_grad_(x, y, z, gx, gy, gz);
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        return func_grad_(x, y, z, gx, gy, gz);
    }

private:
    std::function<Scalar(Scalar, Scalar, Scalar, Scalar &, Scalar &, Scalar &)> func_grad_;
};

template <typename Scalar>
class ConstantFunction : public ImplicitFunction<Scalar>
{
public:
    ConstantFunction(Scalar value) : value_(value) {}
    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override { return value_; }
    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        gx = 0;
        gy = 0;
        gz = 0;
        return value_;
    }

private:
    Scalar value_;
};

template <typename Scalar>
class PlaneDistanceFunction : public ImplicitFunction<Scalar>
{
public:
    PlaneDistanceFunction(const std::array<Scalar, 3> &point, const std::array<Scalar, 3> &normal)
        : point_(point), normal_(normal)
    {
        normalize_vector(normal_);
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        return compute_dot(normal_, {x - point_[0], y - point_[1], z - point_[2]});
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        gx = normal_[0];
        gy = normal_[1];
        gz = normal_[2];
        return compute_dot(normal_, {x - point_[0], y - point_[1], z - point_[2]});
    }

private:
    std::array<Scalar, 3> point_;
    std::array<Scalar, 3> normal_;
};

template <typename Scalar>
class CylinderDistanceFunction : public ImplicitFunction<Scalar>
{
public:
    CylinderDistanceFunction(const std::array<Scalar, 3> &axis_point,
                             const std::array<Scalar, 3> &axis_unit_vector, Scalar radius)
        : axis_point_(axis_point), axis_unit_vector_(axis_unit_vector), radius_(radius)
    {
        normalize_vector(axis_unit_vector_);
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        std::array<Scalar, 3> vec{x - axis_point_[0], y - axis_point_[1], z - axis_point_[2]};
        Scalar d = compute_dot(axis_unit_vector_, vec);
        std::array<Scalar, 3> vec_para{d * axis_unit_vector_[0], d * axis_unit_vector_[1], d * axis_unit_vector_[2]};
        std::array<Scalar, 3> vec_perp{vec[0] - vec_para[0], vec[1] - vec_para[1], vec[2] - vec_para[2]};
        return radius_ - compute_norm(vec_perp);
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        std::array<Scalar, 3> vec{x - axis_point_[0], y - axis_point_[1], z - axis_point_[2]};
        Scalar d = compute_dot(axis_unit_vector_, vec);
        std::array<Scalar, 3> vec_para{d * axis_unit_vector_[0], d * axis_unit_vector_[1], d * axis_unit_vector_[2]};
        std::array<Scalar, 3> vec_perp{vec[0] - vec_para[0], vec[1] - vec_para[1], vec[2] - vec_para[2]};
        Scalar vec_perp_norm = compute_norm(vec_perp);
        if (vec_perp_norm == 0)
        {
            gx = 0;
            gy = 0;
            gz = 0;
            return radius_;
        }
        else
        {
            gx = -vec_perp[0] / vec_perp_norm;
            gy = -vec_perp[1] / vec_perp_norm;
            gz = -vec_perp[2] / vec_perp_norm;
            return radius_ - vec_perp_norm;
        }
    }

private:
    std::array<Scalar, 3> axis_point_;
    std::array<Scalar, 3> axis_unit_vector_;
    Scalar radius_;
};

template <typename Scalar>
class CylinderSquaredDistanceFunction : public ImplicitFunction<Scalar>
{
public:
    CylinderSquaredDistanceFunction(const std::array<Scalar, 3> &axis_point,
                                    const std::array<Scalar, 3> &axis_unit_vector, Scalar radius)
        : axis_point_(axis_point), axis_unit_vector_(axis_unit_vector), radius_(radius)
    {
        normalize_vector(axis_unit_vector_);
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        std::array<Scalar, 3> vec{x - axis_point_[0], y - axis_point_[1], z - axis_point_[2]};
        Scalar d = compute_dot(axis_unit_vector_, vec);
        std::array<Scalar, 3> vec_para{d * axis_unit_vector_[0], d * axis_unit_vector_[1], d * axis_unit_vector_[2]};
        std::array<Scalar, 3> vec_perp{vec[0] - vec_para[0], vec[1] - vec_para[1], vec[2] - vec_para[2]};
        return radius_ * radius_ - compute_squared_norm(vec_perp);
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        std::array<Scalar, 3> vec{x - axis_point_[0], y - axis_point_[1], z - axis_point_[2]};
        Scalar d = compute_dot(axis_unit_vector_, vec);
        std::array<Scalar, 3> vec_para{d * axis_unit_vector_[0], d * axis_unit_vector_[1], d * axis_unit_vector_[2]};
        std::array<Scalar, 3> vec_perp{vec[0] - vec_para[0], vec[1] - vec_para[1], vec[2] - vec_para[2]};
        gx = -2 * vec_perp[0];
        gy = -2 * vec_perp[1];
        gz = -2 * vec_perp[2];
        return radius_ * radius_ - compute_squared_norm(vec_perp);
    }

private:
    std::array<Scalar, 3> axis_point_;
    std::array<Scalar, 3> axis_unit_vector_;
    Scalar radius_;
};

template <typename Scalar>
class SphereDistanceFunction : public ImplicitFunction<Scalar>
{
public:
    SphereDistanceFunction(const std::array<Scalar, 3> &center, Scalar radius)
        : center_(center), radius_(radius)
    {
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        return radius_ - compute_Euclidean_distance(center_, {x, y, z});
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        std::array<Scalar, 3> vec{x - center_[0], y - center_[1], z - center_[2]};
        Scalar vec_norm = compute_norm(vec);
        if (vec_norm == 0)
        {
            gx = 0;
            gy = 0;
            gz = 0;
            return radius_;
        }
        else
        {
            gx = -vec[0] / vec_norm;
            gy = -vec[1] / vec_norm;
            gz = -vec[2] / vec_norm;
            return radius_ - vec_norm;
        }
    }

private:
    std::array<Scalar, 3> center_;
    Scalar radius_;
};

template <typename Scalar>
class SphereUnsignedDistanceFunction : public ImplicitFunction<Scalar>
{
public:
    SphereUnsignedDistanceFunction(const std::array<Scalar, 3> &center, Scalar radius)
        : center_(center), radius_(radius)
    {
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        return -abs(radius_ - compute_Euclidean_distance(center_, {x, y, z}));
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        std::array<Scalar, 3> vec{x - center_[0], y - center_[1], z - center_[2]};
        Scalar vec_norm = compute_norm(vec);
        if (vec_norm == 0)
        {
            gx = 0;
            gy = 0;
            gz = 0;
            return -radius_;
        }
        else if (vec_norm < radius_)
        {
            gx = vec[0] / vec_norm;
            gy = vec[1] / vec_norm;
            gz = vec[2] / vec_norm;
            return vec_norm - radius_;
        }
        else
        {
            gx = -vec[0] / vec_norm;
            gy = -vec[1] / vec_norm;
            gz = -vec[2] / vec_norm;
            return radius_ - vec_norm;
        }
    }

private:
    std::array<Scalar, 3> center_;
    Scalar radius_;
};

template <typename Scalar>
class SphereSquaredDistanceFunction : public ImplicitFunction<Scalar>
{
public:
    SphereSquaredDistanceFunction(const std::array<Scalar, 3> &center, Scalar radius)
        : center_(center), radius_(radius)
    {
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        return radius_ * radius_ - compute_squared_distance(center_, {x, y, z});
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        gx = 2 * (center_[0] - x);
        gy = 2 * (center_[1] - y);
        gz = 2 * (center_[2] - z);
        return radius_ * radius_ - (gx * gx + gy * gy + gz * gz) / 4;
    }

private:
    std::array<Scalar, 3> center_;
    Scalar radius_;
};

template <typename Scalar>
class ConeDistanceFunction : public ImplicitFunction<Scalar>
{
public:
    ConeDistanceFunction(const std::array<Scalar, 3> &apex,
                         const std::array<Scalar, 3> &axis_unit_vector, Scalar apex_angle)
        : apex_(apex), axis_unit_vector_(axis_unit_vector), apex_angle_(apex_angle)
    {
        normalize_vector(axis_unit_vector_);
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        return compute_dot(axis_unit_vector_, {x - apex_[0], y - apex_[1], z - apex_[2]}) - cos(apex_angle_) * compute_Euclidean_distance({x, y, z}, apex_);
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        std::array<Scalar, 3> vec{x - apex_[0], y - apex_[1], z - apex_[2]};
        Scalar vec_norm = compute_norm(vec);
        Scalar d = compute_dot(axis_unit_vector_, vec);
        if (vec_norm == 0)
        {
            gx = 0;
            gy = 0;
            gz = 0;
            return 0;
        }
        else
        {
            Scalar factor = cos(apex_angle_) / vec_norm;
            gx = axis_unit_vector_[0] - vec[0] * factor;
            gy = axis_unit_vector_[1] - vec[1] * factor;
            gz = axis_unit_vector_[2] - vec[2] * factor;
            return d - cos(apex_angle_) * vec_norm;
        }
    }

private:
    std::array<Scalar, 3> apex_;
    std::array<Scalar, 3> axis_unit_vector_;
    Scalar apex_angle_;
};

template <typename Scalar>
class TorusDistanceFunction : public ImplicitFunction<Scalar>
{
public:
    TorusDistanceFunction(const std::array<Scalar, 3> &center,
                          const std::array<Scalar, 3> &axis_unit_vector,
                          Scalar major_radius, Scalar minor_radius)
        : center_(center), axis_unit_vector_(axis_unit_vector),
          major_radius_(major_radius), minor_radius_(minor_radius)
    {
        normalize_vector(axis_unit_vector_);
    }

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        std::array<Scalar, 3> vec{x - center_[0], y - center_[1], z - center_[2]};
        Scalar d = compute_dot(axis_unit_vector_, vec);
        std::array<Scalar, 3> vec_para{d * axis_unit_vector_[0], d * axis_unit_vector_[1], d * axis_unit_vector_[2]};
        std::array<Scalar, 3> vec_perp{vec[0] - vec_para[0], vec[1] - vec_para[1], vec[2] - vec_para[2]};
        Scalar vec_perp_norm = compute_norm(vec_perp);
        if (vec_perp_norm == 0)
        { // point lies on torus axis
            return minor_radius_ - sqrt(compute_dot(vec_para, vec_para) + major_radius_ * major_radius_);
        }
        else
        {
            return minor_radius_ - sqrt(major_radius_ * (major_radius_ - 2 * vec_perp_norm) + compute_dot(vec, vec));
        }
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        std::array<Scalar, 3> vec{x - center_[0], y - center_[1], z - center_[2]};
        Scalar d = compute_dot(axis_unit_vector_, vec);
        std::array<Scalar, 3> vec_para{d * axis_unit_vector_[0], d * axis_unit_vector_[1], d * axis_unit_vector_[2]};
        std::array<Scalar, 3> vec_perp{vec[0] - vec_para[0], vec[1] - vec_para[1], vec[2] - vec_para[2]};
        Scalar vec_perp_norm = compute_norm(vec_perp);
        if (vec_perp_norm == 0)
        {
            // point lies on torus axis
            Scalar rt = sqrt(compute_dot(vec_para, vec_para) + major_radius_ * major_radius_);
            gx = -vec_para[0] / rt;
            gy = -vec_para[1] / rt;
            gz = -vec_para[2] / rt;
            return minor_radius_ - rt;
        }
        else
        {
            Scalar rt = sqrt(major_radius_ * (major_radius_ - 2 * vec_perp_norm) + compute_dot(vec, vec));
            gx = (vec[0] - major_radius_ * vec_perp[0] / vec_perp_norm) / rt;
            gy = (vec[1] - major_radius_ * vec_perp[1] / vec_perp_norm) / rt;
            gz = (vec[2] - major_radius_ * vec_perp[2] / vec_perp_norm) / rt;
            return minor_radius_ - rt;
        }
    }

private:
    std::array<Scalar, 3> center_;
    std::array<Scalar, 3> axis_unit_vector_;
    Scalar major_radius_;
    Scalar minor_radius_;
};

bool load_functions(const std::string &filename,
                    const std::vector<std::array<double, 3>> &pts,
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &funcVals);

bool load_functions(const std::string &filename, std::vector<std::unique_ptr<ImplicitFunction<double>>> &functions);
