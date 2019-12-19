#ifndef MY_RANSAC_MODEL_H
#define MY_RANSAC_MODEL_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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
    virtual void train(const Data &data) = 0;
    virtual Param getParam() const = 0;
    virtual void printParam() const = 0;
    virtual double calcError(const Datum &datum) const = 0;
    std::vector<double> calcErrors(const Data &data) const
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

class ModelLine2D : public AbstractModel<cv::Point2d, std::vector<double>>
{
    typedef cv::Point2d Datum;         // (x, y)
    typedef std::vector<double> Param; // (a, b, c). Line eq: ax+by+c=0.
    typedef std::vector<Datum> Data;

protected:
    double a_, b_, c_; // Line eq: ax+by+c=0.

private:
    Datum p1_, p2_; // Two points on the line for drawing a line.
    double sqrt_a2b2_;

public:
    ModelLine2D() {}

    Param getParam() const { return Param{a_, b_, c_}; }
    virtual void printParam() const
    {
        std::cout << "2D Line parameters: "
                  << "a = " << a_ << ", "
                  << "b = " << b_ << ", "
                  << "c = " << c_ << std::endl;
    }

    void train(const Data &points)
    {
        // -- Check input.
        int P = points.size();
        if (P < 2)
            throw std::runtime_error("Need at least 2 points to fit a line.");
        if (P > 10000)
            throw std::runtime_error("Too many points! Please use only part of them.");

        // -- Fit line by finding the 1st principle axis of PCA,
        //      which is the line direction.
        Eigen::MatrixXd data = this->vector2matrix(points);
        Eigen::Matrix<double, 1, 2> line_center = data.colwise().mean();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd( // U,S,V
            data.rowwise() - line_center,
            // Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::ComputeThinV);
        Eigen::Matrix<double, 2, 2> v = svd.matrixV();
        Eigen::Vector2d line_direction = v.col(0); // 1st principle axis of PCA.

        // -- Get line parameters.
        // Line model: a*(x-x0)+b*(y-y0)=0,
        //      which is: ax + by + (-x0*a-y0*b) = 0.
        double x0 = line_center(0);
        double y0 = line_center(1);
        double a = -line_direction(1);
        double b = line_direction(0);

        // So if line model is: ax + by + c = 0,
        //      then c = -x0 * a - y0 * b
        double c = -x0 * a - y0 * b;

        // -- Save result.
        // Line parameters.
        a_ = a, b_ = b, c_ = c;
        this->is_fitted_ = true;
        sqrt_a2b2_ = pow(a * 2 + b * 2, 0.5);

        // Save two points' positions for later drawing image.
        constexpr double L = 1000.0; // Line half length.
        this->p1_ = Datum(x0 - L * line_direction(0),
                          y0 - L * line_direction(1));
        this->p2_ = Datum(x0 + L * line_direction(0),
                          y0 + L * line_direction(1));
    }

    double calcError(const Datum &point) const
    {
        if (!is_fitted_)
            throw std::runtime_error("Model hasn't been fitted.");
        // Error = |ax+by+c|/sqrt(a**2+b**2)
        return abs(a_ + point.x + b_ * point.y + c_) / sqrt_a2b2_;
    }

    void draw(cv::Mat *img_disp,
              const cv::Scalar color = {255, 0, 0},
              const int thickness = 3)
    {
        cv::line(*img_disp,
                 {int(p1_.x), int(p1_.y)},
                 {int(p2_.x), int(p2_.y)},
                 color, thickness);
    }

private:
    Eigen::MatrixXd vector2matrix(const Data &points) const
    {
        int P = points.size();
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(P, 2);
        for (int i = 0; i < P; i++)
        {
            matrix(i, 0) = points[i].x;
            matrix(i, 1) = points[i].y;
        }
        return matrix;
    }
};

} // namespace models

#endif