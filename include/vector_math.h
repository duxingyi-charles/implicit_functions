#pragma once

#include <array>

template <typename Scalar>
inline Scalar compute_Euclidean_distance(const std::array<Scalar, 3> &p, const std::array<Scalar, 3> &q)
{
    return sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
}

template <typename Scalar>
inline Scalar compute_squared_distance(const std::array<Scalar, 3> &p, const std::array<Scalar, 3> &q)
{
    return (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]);
}

template <typename Scalar>
inline Scalar compute_norm(const std::array<Scalar, 3> &v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

template <typename Scalar>
inline Scalar compute_squared_norm(const std::array<Scalar, 3> &v)
{
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

template <typename Scalar>
inline void normalize_vector(std::array<Scalar, 3> &v)
{
    Scalar norm = compute_norm<Scalar>(v);
    if (norm == 0) return;
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
}

template <typename Scalar>
inline Scalar compute_dot(const std::array<Scalar, 3> &a, const std::array<Scalar, 3> &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}