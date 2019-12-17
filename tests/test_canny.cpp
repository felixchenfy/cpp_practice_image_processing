
#include "my_cv/filters.h"
#include "my_cv/cv_basics.h"

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

void test_canny()
{

    // -- Read image
    const std::string filename = "data/color_chessboard.jpg";
    // const std::string filename = "data/simple_shapes2.png";
    cv::Mat src_gray;
    cv::cvtColor(readImage(filename), src_gray, cv::COLOR_BGR2GRAY);

    // -- Canny(Mine).
    const float lb = 70.0, ub = 210.0;
    const int kernel_size = 3;
    cv::Mat1b edge_canny = filters::canny(src_gray, lb, ub, kernel_size);

    // -- Canny(OpenCV).
    cv::Mat1b edge_canny_opencv;
    cv::Canny(src_gray, edge_canny_opencv, lb, ub, kernel_size);

    // -- For comparison, compute sobel.
    constexpr bool TAKE_ABS = true;
    cv::Mat1b edge_sobel = cv_basics::float2uint8(filters::sobel(src_gray), TAKE_ABS);

    // -- Show image.
    const std::string WINDOW_NAME = "Original / Sobel / Canny(Mine) / Canny(OpenCV)";
    cv_basics::display_images(
        {src_gray, edge_sobel, edge_canny, edge_canny_opencv},
        WINDOW_NAME);
}

int main(int argc, char const *argv[])
{
    test_canny();
}
