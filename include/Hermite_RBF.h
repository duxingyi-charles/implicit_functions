#pragma once

#include "ImplicitFunction.h"
#include <Eigen/Core>


template<typename Scalar>
class Hermite_RBF : public ImplicitFunction<Scalar> {
public:
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecX = Eigen::Matrix<Scalar, -1, 1>;

    // constant 0 function
    Hermite_RBF() : coeff_a_(), coeff_b_(0, 0, 0, 0) {}

    Hermite_RBF(const std::vector<Vec3> &control_points, const VecX &coeff_a,
                const Vec4 &coeff_b)
            : coeff_a_(coeff_a), coeff_b_(coeff_b), control_points_(control_points) {}

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override {
        size_t num_pt = control_points_.size();
        int dim = 3;
        Vec3 p(x, y, z);

        VecX kern(num_pt * (dim + 1));
        for (size_t i = 0; i < num_pt; ++i) {
            kern(i) = kernel_function(p, control_points_[i]);
        }
        Vec3 G;
        for (size_t i = 0; i < num_pt; ++i) {
            G = kernel_gradient(p, control_points_[i]);
            for (int j = 0; j < dim; ++j) {
                kern(num_pt + i + j * num_pt) = G(j);
            }
        }
        Scalar loc_part = kern.dot(coeff_a_);

        Vec4 kb(1, p(0), p(1), p(2));
        Scalar poly_part = kb.dot(coeff_b_);

        return loc_part + poly_part;
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override {
        size_t num_pt = control_points_.size();
        int dim = 3;
        Vec3 p(x, y, z);

        VecX kern(num_pt * (dim + 1));
        for (size_t i = 0; i < num_pt; ++i) {
            kern(i) = kernel_function(p, control_points_[i]);
        }

        Vec3 G;
        Vec3 grad;
        grad.setZero();
        // sum(ai * fi)
        for (size_t i = 0; i < num_pt; ++i) {
            G = kernel_gradient(p, control_points_[i]);
            grad += G * coeff_a_[i];
            for (int j = 0; j < dim; ++j) {
                kern(num_pt + i + j * num_pt) = G(j);
            }
        }
        // sum(hi * bi)
        Eigen::Matrix<Scalar, 3, 3> H;
        for (size_t i = 0; i < num_pt; ++i) {
            H = kernel_Hessian(p, control_points_[i]);
            grad += H.col(0) * coeff_a_[num_pt + i];
            grad += H.col(1) * coeff_a_[2 * num_pt + i];
            grad += H.col(2) * coeff_a_[3 * num_pt + i];
        }
        // c
        grad(0) += coeff_b_(1);
        grad(1) += coeff_b_(2);
        grad(2) += coeff_b_(3);

        gx = grad(0);
        gy = grad(1);
        gz = grad(2);
        // compute function value
        Scalar loc_part = kern.dot(coeff_a_);
        Vec4 kb(1, p(0), p(1), p(2));
        Scalar poly_part = kb.dot(coeff_b_);
        return loc_part + poly_part;
    }

private:
    VecX coeff_a_;
    Vec4 coeff_b_;
    std::vector<Vec3> control_points_;

    // |p1-p2|^3
    static Scalar kernel_function(const Vec3 &p1, const Vec3 &p2) {
        return pow((p1 - p2).norm(), 3);
    }

    // 3 |p1-p2| (p1-p2)
    static Vec3 kernel_gradient(const Vec3 &p1, const Vec3 &p2) {
        return 3 * (p1 - p2).norm() * (p1 - p2);
    }

    // 3 [ |p1-p2|I + (p1-p2)*(p1-p2)^T/|p1-p1| ]
    static Eigen::Matrix<Scalar, 3, 3> kernel_Hessian(const Vec3 &p1, const Vec3 &p2) {
        Vec3 diff = p1 - p2;
        Scalar len = diff.norm();
        if (len < 1e-8) {
            return Eigen::Matrix<Scalar, 3, 3>::Zero();
        }

        Eigen::Matrix<Scalar, 3, 3> hess = diff * (diff.transpose() / len);
        hess(0, 0) += len;
        hess(1, 1) += len;
        hess(2, 2) += len;
        hess *= 3;

        return hess;
    }
};
