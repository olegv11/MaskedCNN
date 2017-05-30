#include "Network.hpp"


using namespace MaskedCNN;

int main()
{
    Network net("/home/oleg/fcn32s-heavy-pascal.caffemodel", 30);
    net.setMaskEnabled(true);
    net.setDisplayMask(0, true);
    net.setDisplayMask(6, true);
    cv::VideoCapture cap = cv::VideoCapture("/home/oleg/videoSmooth2.avi");
    if (!cap.isOpened())
    {
        throw std::exception();
    }

    cv::namedWindow( "Camera", cv::WINDOW_AUTOSIZE );
    cv::Mat frame;

    while (true)
    {
        cap.read(frame);
        cv::imshow("Camera", frame);
        auto x = net.forward(frame);
        cv::imshow(x.front().first, x.front().second);
        cv::imshow(x[1].first, x[1].second);
        cv::imshow(x.back().first, x.back().second);
        cv::waitKey(10);

        std::cout << net.forwardTime() << std::endl;
    }

    return 0;
}

