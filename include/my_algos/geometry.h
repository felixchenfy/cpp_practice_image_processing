#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace geometry
{

/**
 * Non-maximum suppression. // TODO
 */
void nms();

/**
 * 2D line.I
 */
struct Line2d
{
    double d; // Distance to the origin.
    double s; // Slope of the line.
    Line2d(double d = 0.0, double s = 0.0) : d(d), s(s) {}
};

/**
 * Detect lines by Hough Line detection algorithm.
 * @param edge Image of edge. An edge pixel has a non-zero value.
 * @return A vector of lines.
 */
std::vector<Line2d> houghLine(const cv::Mat1b &edge);

} // namespace geometry
#endif