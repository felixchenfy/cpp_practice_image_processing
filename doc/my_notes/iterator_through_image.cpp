
/**
 * Create color image
 */
cv::Mat grHistogram(260, 301, CV_8UC3, Scalar(0, 0, 0));

/**
 * Iterator through color image (Slow way)
 * img.at<cv::Vec3b>(i, j)[0]
 */
cv::Mat img = cv::imread("lenna.png");
for (int i = 0; i < img.rows; i++)
    for (int j = 0; j < img.cols; j++)
        // You can now access the pixel value with cv::Vec3b
        std::cout << img.at<cv::Vec3b>(i, j)[0] << " " << img.at<cv::Vec3b>(i, j)[1] << " " << img.at<cv::Vec3b>(i, j)[2] << std::endl;

/**
 * Iterator through image (Efficient way)
 */
// https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#the-efficient-way
Mat &ScanImageAndReduceC(Mat &I, const uchar *const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);
    int channels = I.channels();
    int nRows = I.rows;
    int nCols = I.cols * channels;
    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    int i, j;
    uchar *p;
    for (i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        for (j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];
        }
    }
    return I;
}

