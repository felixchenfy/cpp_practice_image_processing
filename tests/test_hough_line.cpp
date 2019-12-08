
#include "my_algos/geometry.h"
#include "my_algos/filters.h"
#include "my_algos/cv_basics.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

cv::Mat readImage(const std::string &filename = "")
{
    cv::Mat src = imread(cv::samples::findFile(filename), cv::IMREAD_COLOR); // Load an image
    if (src.empty())
    {
        std::cout << "Could not open or find the image: \n"
                  << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return src;
}

void test_hough_line()
{

    // -- Read image
    const std::string filename = "data/color_chessboard.jpg";
    // const string filename = "data/simple_shapes2.png";
    cv::Mat src_gray;
    cv::cvtColor(readImage(filename), src_gray, cv::COLOR_BGR2GRAY);

    // -- Canny(Mine).
    const float lb = 70.0, ub = 210.0;
    const int kernel_size = 3;
    cv::Mat1b edge = filters::canny(src_gray, lb, ub, kernel_size);

    // -- Hough line detection.
    cv::Mat1i polar;
    const int NMS_MIN_PTS = 30, NMS_RADIUS = 10;
    std::vector<geometry::Line2d> lines = geometry::houghLine(edge, &polar, NMS_MIN_PTS, NMS_RADIUS);
    std::cout << "Detect " << lines.size() << " lines." << std::endl;
    cv_basics::display_images({polar}, "Hough Transform Result");
    return;
}

int main(int argc, char const *argv[])
{
    test_hough_line();
    return 0;
}
