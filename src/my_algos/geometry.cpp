#include "my_algos/cv_basics.h"
#include "my_algos/geometry.h"
#include "my_algos/maths.h"
#include "my_algos/cv_basics.h"

#include <opencv2/core.hpp>
#include <cmath>
#include <iostream>

namespace geometry
{

std::vector<Line2d> detectLineByHoughTransform(
    const cv::Mat1b &edge,
    cv::Mat1i *dst_polar,
    int nms_min_pts,
    int nms_radius)
{
    // Reference:
    // https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
    
    int r = edge.rows, c = edge.cols;
    const int N_ANGLE_GRID = 180;               // [0, 179] degrees,
    const int N_DIST_GRID = maths::norm2(r, c); // Line's max distance to origin.

    // -- Step 1. Hough transform.
    // For each edge point, for each of the possible line passing through this point,
    // convert the line parameter to the polar coordinate, and then add the count by 1.
    cv::Mat polar = cv::Mat::zeros(N_ANGLE_GRID, N_DIST_GRID, CV_32SC1);
    for (int y = 0; y < r; y++)
        for (int x = 0; x < c; x++)
            if (edge.at<uchar>(y, x) != 0)
                for (int theta_i = 0; theta_i <= 179; theta_i++)
                {
                    // Hough transform: from (x, y) to (theta, distance).
                    double theta = theta_i / 180.0 * M_PI;
                    int distance = lround(x * cos(theta) + y * sin(theta));
                    polar.at<uint32_t>(theta_i, distance) += 1;
                }

    // -- Step 2. Non maximum suppression based on min_pts and radius.
    std::vector<cv::Point2i> peaks = nms<uint32_t>(polar, nms_min_pts, nms_radius);
    std::vector<Line2d> lines;
    for (const cv::Point2i &p : peaks)
        lines.push_back(Line2d(p.x, p.y)); // p.x is distance; p.y is angle.

    // -- Return.
    if (dst_polar != nullptr) // Return the image of hough transform.
        (*dst_polar) = std::move(polar);
    return lines;
}

} // namespace geometry