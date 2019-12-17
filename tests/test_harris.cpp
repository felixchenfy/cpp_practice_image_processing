
#include "my_cv/filters.h"
#include "my_cv/cv_commons.h"
#include "my_cv/keypoints.h"

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

    // -- Canny(Mine).
    cv::Mat img_edge_score;
    std::vector<std::pair<double, cv::Point2i>>
        keypoints = keypoints::detectHarris(src_gray, &img_edge_score);

    // -- Convert image for display.
    cv::Mat1b img_edge_score_disp = cv_commons::float2uint8(img_edge_score);

    // -- Show image.
    const std::string WINDOW_NAME = "Original / Edge response score";
    cv_commons::display_images(
        {src_gray, img_edge_score_disp},
        WINDOW_NAME);
}

int main(int argc, char const *argv[])
{
    test_harris();
}
