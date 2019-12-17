#ifndef MY_CV_KEYPOINT_H
#define MY_CV_KEYPOINT_H

#include "my_cv/cv_basics.h"
#include "my_cv/filters.h"
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Point2i> detectHarris(
    const cv::Mat &gray,
    cv::Mat *img_disp = nullptr,
    const int ksize = 5)
{
    assert(ksize % 2 == 1);
    cv::Mat1d Ix = filters::sobelX(gray); // Gradient x.
    cv::Mat1d Iy = filters::sobelY(gray); // Gradient y.
    cv::Mat1d Ix2 = Ix.mul(Ix);
    cv::Mat1d Iy2 = Iy.mul(Iy);
    cv::Mat1d Ixy = Ix.mul(Iy);
    cv::Mat1d Ix2_window_mean = filters::gaussion(Ix2, ksize);
    cv::Mat1d Iy2_window_mean = filters::gaussion(Iy2, ksize);
    cv::Mat1d Ixy_window_mean = filters::gaussion(Ixy, ksize);
    cv::Mat1d Ixy2_window_mean = Ixy_window_mean.mul(Ixy_window_mean);
    cv::Mat1d mat_trace = Ix2_window_mean + Iy2_window_mean;
    cv::Mat1d determinant = Ix2_window_mean * Iy2_window_mean - Ixy2_window_mean;
    // for i in range(r):
    //     for j in range(c):
    //         img_of_Harris_response[i][j] = cost_function(
    //             trace=mat_tr[i, j],
    //             determinant=mat_det[i, j])
}
#endif