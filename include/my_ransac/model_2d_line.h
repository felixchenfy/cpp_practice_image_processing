#ifndef MY_RANSAC_MODEL_2D_LINE_H
#define MY_RANSAC_MODEL_2D_LINE_H

#include "my_ransac/abstract_model.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <vector>
#include <stdexcept>

namespace models
{

class Model2dLine : public AbstractModel<cv::Point2d, std::vector<double>>
{
    typedef cv::Point2d Datum;         // (x, y)
    typedef std::vector<double> Param; // (a, b, c). Line eq: ax+by+c=0.
    typedef std::vector<Datum> Data;

public:
    Model2dLine() {}
    void fit(const Data &points);
    Param getParam() const { return Param{a_, b_, c_}; }
    void printParam() const;
    double calcError(const Datum &point) const;
    void draw(cv::Mat *img_disp,
              const cv::Scalar color = {255, 0, 0},
              const int thickness = 3);

private:
    Eigen::MatrixXd vector2matrix(const Data &points) const;
    void fitTwoPoints(const Data &points);
    void fitMultiplePoints(const Data &points);

protected:
    double a_, b_, c_; // Line eq: ax+by+c=0.

private:
    Datum p1_, p2_;    // Two points on the line for drawing a line.
    cv::Point2d dxdy_; // Line direction.
    double sqrt_a2b2_; // sqrt(a**2 + b**2)
};

} // namespace models

#endif