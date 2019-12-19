#ifndef MY_CV_KEYPOINT_H
#define MY_CV_KEYPOINT_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace keypoints
{

std::vector<std::pair<double, cv::Point2i>>
detectHarrisCorners(
    const cv::Mat &gray,
    cv::Mat1b *dst_img_corner_score = nullptr,
    cv::Mat3b *dst_img_disp = nullptr,
    const int max_points = 300, // Resize dst points vector to be shorter than this.
    const int nms_radius = 10,
    const unsigned char min_score = 100, // The score should be scaled to 0~255 by scale_score.
    const double scale_score = 0.00000001,
    const int harris_window_size = 5);

} // namespace keypoints
#endif
