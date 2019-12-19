
#include "my_cv/geometry.h"
#include "my_cv/filters.h"
#include "my_cv/cv_commons.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

void test_hough_line()
{

    // -- Read image
    const std::string filename = "data/color_chessboard.jpg";
    // const std::string filename = "data/simple_shapes2.png";
    cv::Mat3b src_color = cv_commons::readImage(filename);
    cv::Mat1b src_gray;
    cv::cvtColor(src_color, src_gray, cv::COLOR_BGR2GRAY);

    // -- Canny(Mine).
    const double lb = 3000.0, ub = lb * 3;
    cv::Mat1b edge = filters::canny(src_gray, lb, ub);

    // -- Hough line detection.
    cv::Mat1i polar;
    const int NMS_MIN_PTS = 50, NMS_RADIUS = 15;
    std::vector<geometry::Line2dPolor> lines = geometry::detectLineByHoughTransform(edge, &polar, NMS_MIN_PTS, NMS_RADIUS);
    std::cout << "Detect " << lines.size() << " lines." << std::endl;

    // -- Draw hough line, and then display.
    cv::Mat img_disp = src_color.clone();
    for (const geometry::Line2dPolor line : lines)
    {
        line.drawToImage(&img_disp);
        line.print();
    }

    cv::Mat img_disp1 = cv_commons::display_images({polar}, "Hough Transform Result", -1);
    cv::imwrite("output/test_hough_line_polar_coord.png", img_disp1);

    cv::Mat img_disp2 = cv_commons::display_images({edge, img_disp}, "Edge & Detected lines");
    cv::imwrite("output/test_hough_line_edge_and_lines.png", img_disp2);
    return;
}

int main(int argc, char const *argv[])
{
    test_hough_line();
    return 0;
}
