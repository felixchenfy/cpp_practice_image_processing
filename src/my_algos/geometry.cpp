#include "my_algos/cv_basics.h"
#include "my_algos/geometry.h"
#include "my_algos/maths.h"
#include "my_algos/cv_basics.h"

#include <opencv2/core.hpp>
#include <cmath>
#include <iostream>

namespace geometry
{

std::vector<Line2d> houghLine(
    const cv::Mat1b &edge,
    cv::Mat1i *dst_polar,
    int nms_min_pts,
    int nms_radius)
{
    int r = edge.rows, c = edge.cols;
    const int N_ANGLE_GRID = 180;               // [0, 179] degrees,
    const int N_DIST_GRID = maths::norm2(r, c); // Line's max distance to origin.

    // -- Step 1. Hough transform.
    cv::Mat polar = cv::Mat::zeros(N_ANGLE_GRID, N_DIST_GRID, CV_32SC1);
    for (int y = 0; y < r; y++)
        for (int x = 0; x < c; x++)
            if (edge.at<uchar>(y, x) != 0)
                for (int theta = 0; theta <= 179; theta++)
                {
                    double t = theta / 180.0 * M_PI;
                    int dist = lround(cos(t) * y - sin(t) * x);
                    polar.at<uint32_t>(theta, dist) += 1;
                }

    // -- Step 2. Non maximum suppression.
    std::vector<cv::Point2i> peaks = nms<uint32_t>(polar, nms_min_pts, nms_radius);
    std::vector<Line2d> lines;
    for (const cv::Point2i &p : peaks)
        lines.push_back(Line2d(p.x, p.y)); // p.x is distance; p.y is angle.

    // -- Return.
    if (dst_polar != nullptr)
        (*dst_polar) = std::move(polar);
    return lines;
}

} // namespace geometry