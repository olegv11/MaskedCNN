#include "Network.hpp"


using namespace MaskedCNN;

int main()
{
    Network net("/home/oleg/Deep_learning/fcn/fcn.berkeleyvision.org/voc-fcn32s/fcn32s-heavy-pascal.caffemodel", 320, 240, 30);
    net.setMaskEnabled(true);
    net.setDisplayMask(0, true);
    net.setDisplayMask(3, true);
    net.setDisplayMask(4, true);
    net.setDisplayMask(5, true);
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
        cv::imshow(x[2].first, x[2].second);
        cv::imshow(x[3].first, x[3].second);
        cv::imshow(x.back().first, x.back().second);
        cv::waitKey(10);

        std::cout << net.forwardTime() << std::endl;
    }

    return 0;
}

