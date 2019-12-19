#include "my_cv/keypoints.h"
#include "my_cv/filters.h"
#include "my_cv/cv_commons.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

void test_harris()
{

    // -- Read image
    const std::string filename = "data/color_chessboard.jpg";
    // const std::string filename = "data/simple_shapes2.png";
    cv::Mat src_gray;
    cv::cvtColor(cv_commons::readImage(filename), src_gray, cv::COLOR_BGR2GRAY);

    // -- Harris corner detection.
    cv::Mat1b img_corner_score;
    cv::Mat3b img_with_corners;
    const int max_points = 300;
    const int nms_radius = 10;
    const unsigned char min_score = 100;

    std::vector<std::pair<double, cv::Point2i>>
        corners_score_and_position =
            keypoints::detectHarrisCorners(
                src_gray, &img_corner_score, &img_with_corners,
                max_points, nms_radius, min_score);

    // -- Show image.
    const std::string WINDOW_NAME = "Original / Harris score / Corners(red)";
    cv::Mat img_disp = cv_commons::display_images(
        {src_gray, img_corner_score, img_with_corners},
        WINDOW_NAME);
    cv::imwrite("output/test_harris.png", img_disp);
}

int main(int argc, char const *argv[])
{
    test_harris();
}
