#ifndef MY_RANSAC_MODEL_H
#define MY_RANSAC_MODEL_H

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <vector>
#include <stdexcept>

namespace models
{

template <typename Datum, typename Param>
class AbstractModel
{
    typedef std::vector<Datum> Data;

public:
    AbstractModel() {}
    virtual ~AbstractModel() {}

public:
    virtual void fitParam(const Data &data) = 0;
    virtual double calcError(const Datum &datum) = 0;
    std::vector<double> calcErrors(const Data &data)
    {
        std::vector<double> errors;
        errors.reserve(data.size());
        for (const Datum &data : data)
            errors.push_back(this->calcError(data));
        return errors;
    }

protected:
    bool is_fitted_ = false;
};

class ModelLine2D : public AbstractModel<cv::Point2d, Eigen::Vector3d>
{
    typedef cv::Point2d Datum; // (x, y)
    typedef std::vector<Datum> Data;

protected:
    double a_, b_, c_, sqrt_a2b2_; // Line eq: ax+by+c=0.

public:
    ModelLine2D() {}

    void fitParam(const Data &vec_xy)
    {
        Eigen::MatrixXd data = this->vector2matrix(vec_xy);
        Eigen::Matrix<double, 1, 2> line_center = data.colwise().mean();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd( // U,S,V
            data - line_center,
            // Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::ComputeThinV);
        Eigen::Matrix<double, 2, 2> v = svd.matrixV();
        Eigen::Vector2d line_direction = v.col(0); // 1st principle axis of PCA.

        // -- Get line parameters.
        // Line model: a*(x-x0)+b*(y-y0)=0,
        //      which is: ax + by + (-x0*a-y0*b) = 0.
        double x0 = -line_center(0);
        double y0 = line_center(1);
        double a = -line_direction(1);
        double b = line_direction(0);

        // So if line model is: ax + by + c = 0,
        //      then c = -x0 * a - y0 * b
        double c = -x0 * a - y0 * b;

        // -- Save result.
        a_ = a, b_ = b, c_ = c, sqrt_a2b2_ = pow(a * 2 + b * 2, 0.5);
        this->is_fitted_ = true;
    }

    double calcError(const Datum &xy)
    {
        if (!is_fitted_)
            throw std::runtime_error("Model hasn't been fitted.");
        // Error = |ax+by+c|/sqrt(a**2+b**2)
        return abs(a_ + xy.x + b_ * xy.y + c_) / sqrt_a2b2_;
    }

private:
    Eigen::MatrixXd vector2matrix(const Data &vec_xy)
    {
        int P = vec_xy.size();
        if (P < 2)
            throw std::runtime_error("Need at least 2 points to fit a line.");
        if (P > 10000)
            throw std::runtime_error("Too many points! Please use only part of them.");
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(P, 2);
        for (int i = 0; i < P; i++)
        {
            matrix(i, 0) = vec_xy[i].x;
            matrix(i, 1) = vec_xy[i].y;
        }
        return matrix;
    }
};

} // namespace models

#endif