
#include "my_cv/keypoints.h"

#include "my_cv/cv_commons.hpp"
#include "my_cv/filters.h"
#include "my_cv/geometry.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <iostream>

namespace keypoints
{
/**
 * This is Harris' score function for detecting corners.
 * The score is high if the pixel is a corner point.
 * 
 * The pixel is a corner only when its gradient of intensity
 *  is large in both two orthogonal directions.
 * Translated into math, it means that 
 *  that 2x2 matrix has 2 large eigenvalues. 
 * So we can design the score function like this:
 *      score = a1 * a2 - k * (a1 + a2)**2 
 *      where a1,a2 are the eigenvalues.
 * 
 * Since the matrix havs this properties:
 *      trace(m) == a1 + a2 
 *      determinant(m) == a1 * a2
 * The score function is equivalent to:
 *      score = determinant(m) - k * trance(m)**2
 */
inline double _Harris_score_func_for_corner(
    double trace, double determinant, double k = 0.04)
{
    return determinant - k * pow(trace, 2.);
}

/**
 * We can modify the Harris score function to detect edge.
 * The edge pixel has a large gradient in one direction and small in the other,
 *  so the eigenvalues are a large one and a small one.
 * We can define the score as:
 *  score = (a1 - a2)**2
 */
inline double _Harris_score_func_for_edge(
    double trace, double determinant)
{
    return pow(trace, 2.) - 4 * determinant;
}

std::vector<std::pair<double, cv::Point2i>> detectHarrisCorners(
    const cv::Mat &gray,
    cv::Mat1b *dst_img_edge_score,
    cv::Mat3b *dst_img_disp,
    const int max_points,
    const int nms_radius,
    const unsigned char min_score,
    const double scale_score,
    const int harris_window_size)
{
    assert(harris_window_size % 2 == 1);

    cv::Mat1d Ix = filters::sobelX(gray); // Gradient x.
    cv::Mat1d Iy = filters::sobelY(gray); // Gradient y.
    cv::Mat1d Ix2 = Ix.mul(Ix);
    cv::Mat1d Iy2 = Iy.mul(Iy);
    cv::Mat1d Ixy = Ix.mul(Iy);
    cv::Mat1d Ix2_window_mean = filters::gaussion(Ix2, harris_window_size);
    cv::Mat1d Iy2_window_mean = filters::gaussion(Iy2, harris_window_size);
    cv::Mat1d Ixy_window_mean = filters::gaussion(Ixy, harris_window_size);
    cv::Mat1d Ixy2_window_mean = Ixy_window_mean.mul(Ixy_window_mean);
    cv::Mat1d mat_trace = Ix2_window_mean + Iy2_window_mean;
    cv::Mat1d mat_determinant = Ix2_window_mean.mul(Iy2_window_mean) - Ixy2_window_mean;
    cv::Mat1d res_score_img = cv::Mat::zeros(gray.size(), CV_64FC1);

    // -- Detect corners.
    //   (If have you want to detect edge instead of corner,
    //   you may try: _Harris_score_func_for_edge.)
    for (int i = 0; i < gray.rows; i++)
        for (int j = 0; j < gray.cols; j++)
            res_score_img.at<double>(i, j) =
                scale_score * _Harris_score_func_for_corner(
                                  mat_trace.at<double>(i, j),
                                  mat_determinant.at<double>(i, j));

    // -- Non-maximum suppressioin to find pixels with local max score.
    std::vector<std::pair<double, cv::Point2i>>
        corners_score_and_pos = geometry::nms<double>(
            res_score_img, min_score, nms_radius);

    if (corners_score_and_pos.size() > max_points)
        corners_score_and_pos.resize(max_points);

    // -- Get the position of the peaks.
    // std::vector<cv::Point2i> peaks_position;
    // for (auto const &peak : peaks)
    //     peaks_position.push_back(peak.second);

    // -- Return.
    if (dst_img_edge_score != nullptr)
    {
        *dst_img_edge_score = cv_commons::float2uint8(res_score_img);
    }
    if (dst_img_disp != nullptr)
    {
        // Draw it!
        constexpr int RADIUS = 2;
        constexpr int LINE_TYPE = 8;
        const cv::Scalar COLOR_RED{0, 0, 255};
        *dst_img_disp = cv_commons::gray2color(gray);
        for (const auto p : corners_score_and_pos)
        {
            circle(*dst_img_disp, p.second,
                   RADIUS, COLOR_RED, CV_FILLED, LINE_TYPE);
        }
    }
    return corners_score_and_pos;
}
} // namespace keypoints
