#ifndef MY_CV_GEOMETRY_H
#define MY_CV_GEOMETRY_H

/**
 * Classes and functions:
 *      struct Line2d: Line represented in polar coordinate.
 *      detectLineByHoughTransform: Hough line detection.
 *      nms: non-maximum suppression.
 */

#include "my_cv/cv_commons.h"
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace geometry
{

/**
 * 2D line. Represented in polar coordinate.
 */
struct Line2d
{
    double distance; // Distance to the origin.
    double angle;    // Angle of the line's perpendicular line which passes the origin.
    Line2d(double distance = 0.0, double angle = 0.0) : distance(distance), angle(angle) {}
    void drawToImage(cv::Mat *img, const cv::Scalar color = {0, 0, 255}, const int thickness = 2) const
    {
        assert(img->channels() == 3);
        double a = angle / 180.0 * M_PI; // Angle.
        double x = distance * cos(a);
        double y = distance * sin(a);
        int L = 10 * (img->rows + img->cols); // Draw a line longer than image size.
        double dx = cos(a - M_PI_2) * L, dy = sin(a - M_PI_2) * L;
        cv::line(*img, {int(x + dx), int(y + dy)}, {int(x - dx), int(y - dy)}, color);
    }
    void print() const
    {
        std::cout << "Line parameters: angle = " << angle << " degrees, "
                  << "distance = " << distance << " pixels." << std::endl;
    }
};

/**
 * Detect lines by Hough Line detection algorithm.
 * @param edge Image of edge. A pixel is edge if it has a non-zero value.
 * @param dst_polar (Optional) Image of Hough Transform's result. 
 *      Row number is 180, representing 0~179 degrees; 
 *      Column number is the diagonal length of the input image. 
 * @param nms_min_pts Min points on a line. 
 * @param nms_radius Radius of non-maximum suppression.
 * @return Parameters of each detected line.
 */
std::vector<Line2d>
detectLineByHoughTransform(
    const cv::Mat1b &edge,
    cv::Mat1i *dst_polar = nullptr,
    const int nms_min_pts = 30,
    const int nms_radius = 10);

/**
 * Non-maximum suppression on heatmap.
 * @param heaptmap An image of with pixel_type.
 * @param radius Radius of NMS.
 * @return (x, y) position of each peak point in heatmap.
 *  The points are sort from high score to low score.
 */
template <typename pixel_type>
std::vector<std::pair<pixel_type, cv::Point2i>> nms(
    const cv::Mat &heatmap,
    const int min_value,
    const int radius)
{
    assert(heatmap.channels() == 1);

    // -- Detect local max and store the (score, position).
    cv::Mat mask = cv::Mat::ones(heatmap.size(), CV_8UC1);
    std::vector<std::pair<pixel_type, cv::Point2i>> peaks; // vector of (score, position).
    for (int i = 0; i < heatmap.rows; i++)
        for (int j = 0; j < heatmap.cols; j++)
        {
            const pixel_type score = heatmap.at<pixel_type>(i, j);
            if (mask.at<uchar>(i, j) == 0 || score < min_value)
                continue;
            const bool is_max = cv_commons::isLocalMax<pixel_type>(heatmap, i, j, radius);
            if (is_max)
            {
                cv_commons::setNeighborsToZero<uchar>(&mask, i, j, radius);
                peaks.push_back({score, {j, i}});
            }
        }

    // -- Sort peaks based on their scores.
    std::sort(peaks.begin(), peaks.end(),
              [](auto const &p1, auto const &p2) { return p1.first > p2.first; });

    // -- Get the positions of peaks.
    // std::vector<cv::Point2i> peaks_position;
    // for (auto const &peak : peaks)
    //     peaks_position.push_back(peak.second);

    // -- Return.
    return peaks;
}

} // namespace geometry
#endif