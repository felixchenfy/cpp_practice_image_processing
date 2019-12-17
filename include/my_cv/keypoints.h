#ifndef MY_CV_KEYPOINT_H
#define MY_CV_KEYPOINT_H

#include "my_cv/cv_commons.h"
#include "my_cv/filters.h"
#include "my_cv/geometry.h"
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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

std::vector<std::pair<double, cv::Point2i>> detectHarris(
    const cv::Mat &gray,
    cv::Mat *dst_img_edge_score = nullptr,
    cv::Mat *dst_img_disp = nullptr,
    const int max_points = 100,
    const int nms_radius = 10,
    const double min_score = 100.0,
    const double scale_score = 0.00000001,
    const int window_size = 5)
{
    assert(window_size % 2 == 1);

    cv::Mat1d Ix = filters::sobelX(gray); // Gradient x.
    cv::Mat1d Iy = filters::sobelY(gray); // Gradient y.
    cv::Mat1d Ix2 = Ix.mul(Ix);
    cv::Mat1d Iy2 = Iy.mul(Iy);
    cv::Mat1d Ixy = Ix.mul(Iy);
    cv::Mat1d Ix2_window_mean = filters::gaussion(Ix2, window_size);
    cv::Mat1d Iy2_window_mean = filters::gaussion(Iy2, window_size);
    cv::Mat1d Ixy_window_mean = filters::gaussion(Ixy, window_size);
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
        peaks = geometry::nms<double>(
            res_score_img, min_score, nms_radius);

    if (peaks.size() > max_points)
        peaks.resize(max_points);

    // -- Get the position of the peaks.
    // std::vector<cv::Point2i> peaks_position;
    // for (auto const &peak : peaks)
    //     peaks_position.push_back(peak.second);

    // -- Return.
    if (dst_img_edge_score != nullptr)
    {
        *dst_img_edge_score = res_score_img;
    }
    if (dst_img_disp != nullptr)
    {
        // Draw it!
    }
    return peaks;
}
} // namespace keypoints
#endif
