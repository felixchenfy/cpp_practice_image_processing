
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char *window_name = "Edge Map";

static void CannyThreshold(int, void *)
{
    blur(src_gray, detected_edges, Size(3, 3));
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);
    imshow(window_name, dst);
}

void readImage(int argc, char **argv)
{
    const bool kReadFromCommandLine = argc > 1;
    const std::string img_file_name_if_read_from_file = "data/rgbd/common/depth/shaver.png";

    if (kReadFromCommandLine)
    {
        CommandLineParser parser(argc, argv, "{@input | fruits.jpg | input image}");
        src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR); // Load an image
    }
    else
    {
        src = imread(samples::findFile(img_file_name_if_read_from_file), IMREAD_COLOR); // Load an image
    }
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n"
                  << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        exit(EXIT_FAILURE);
        return;
    }
}

int main(int argc, char **argv)
{
    readImage(argc, argv);

    dst.create(src.size(), src.type());
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    namedWindow(window_name, WINDOW_AUTOSIZE);
    createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
    CannyThreshold(0, 0);
    waitKey(0);
    return 0;
}