#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "benchmark.hpp"

using namespace cv;

int main(int argc, char **argv)
{
    if(argc != 5){
        std::cerr << "wrong number of arguments" << std::endl;
    }
    cv::setNumThreads(1);


    // A = x*B + (1-x)*C
    std::string file_A = argv[1];
    std::string file_B = argv[2];
    std::string file_C = argv[3];
    double alpha       = std::stod(argv[4]);
    double beta        = 1 - alpha;

    Mat src1 = imread( file_B , IMREAD_GRAYSCALE);
    Mat src2 = imread( file_C , IMREAD_GRAYSCALE);

    if( src1.empty() ) { std::cout << "Error loading src1" << std::endl; return EXIT_FAILURE; }
    if( src2.empty() ) { std::cout << "Error loading src2" << std::endl; return EXIT_FAILURE; }

    Mat dst;

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
        []() {},
        [&]() {
            addWeighted( src1, alpha, src2, beta, 0.0, dst);
        }
    );

    std::cout << time << std::endl;
    
    imwrite(file_A, dst);

    return 0;
}
