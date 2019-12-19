#include "my_ransac/models.h"

namespace models
{

void ModelLine2D::printParam() const
{
    std::cout << "2D Line parameters: "
              << "a = " << a_ << ", "
              << "b = " << b_ << ", "
              << "c = " << c_ << std::endl;
}

void ModelLine2D::fit(const Data &points)
{
    // -- Check input.
    int P = points.size();
    if (P < 2)
        throw std::runtime_error("Need at least 2 points to fit a line.");
    if (P > 10000)
        throw std::runtime_error("Too many points! Please use only part of them.");
    if (P == 2)
        this->fitTwoPoints(points); // Directly obtain params from line equation. Fast.
    else
        this->fitMultiplePoints(points); // SVD and find 1st principle axis of PCA.
    this->is_fitted_ = true;
    sqrt_a2b2_ = pow(a_ * 2 + b_ * 2, 0.5);
}

void ModelLine2D::fitTwoPoints(const Data &points)
{
    // Line eq: ax + by + c = 0.
    const Datum &P = points[0], &Q = points[1];
    const double a = Q.y - P.y;
    const double b = P.x - Q.x;
    const double c = -a * (P.x) - b * (P.y);

    // -- Save results.
    a_ = a, b_ = b, c_ = c;
    dxdy_ = cv::Point2d(b, -a);
    p1_ = P, p2_ = Q;
}

void ModelLine2D::fitMultiplePoints(const Data &points)
{
    // -- Fit line by finding the 1st principle axis of PCA,
    //      which is the line direction.
    // Line eq: ax + by + c = 0.

    const Eigen::MatrixXd data = this->vector2matrix(points);

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

    // -- Save results.
    a_ = a, b_ = b, c_ = c;
    dxdy_ = cv::Point2d(line_direction(0), line_direction(1));
    p1_ = {x0, y0}, p2_ = {x0, y0};
}

double ModelLine2D::calcError(const Datum &point) const
{
    if (!is_fitted_)
        throw std::runtime_error("Model hasn't been fitted.");
    // Error = |ax+by+c|/sqrt(a**2+b**2)
    return abs(a_ + point.x + b_ * point.y + c_) / sqrt_a2b2_;
}

void ModelLine2D::draw(cv::Mat *img_disp,
                       const cv::Scalar color,
                       const int thickness)
{

    // Make the point far away for drawing.
    constexpr double L = 1000.0;
    cv::Point2i p1 = {int(p1_.x - L * dxdy_.x),
                      int(p1_.y - L * dxdy_.y)};
    cv::Point2i p2 = {int(p2_.x + L * dxdy_.x),
                      int(p2_.y + L * dxdy_.y)};
    cv::line(*img_disp, p1, p2, color, thickness);
}

Eigen::MatrixXd ModelLine2D::vector2matrix(const Data &points) const
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
} // namespace models
