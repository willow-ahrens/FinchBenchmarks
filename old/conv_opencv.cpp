#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "benchmark.hpp"

using namespace cv;

void mtxWriteDense(std::string filename, cv::Mat& m){
    std::ofstream fout(filename);

    if(!fout) {
        std::cout<<"File Not Opened : " << filename << std::endl;  return;
    }

    fout << "%%MatrixMarket matrix array real general" << std::endl;
    fout << m.rows << " " << m.cols << std::endl;
    
    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout << m.at<double>(i,j) << std::endl;
        }
    }

    fout.close();
}

int main(int argc, char **argv)
{
    cv::setNumThreads(1);
    if(argc != 3){
        std::cerr << "wrong number of arguments " << argc << std::endl;
    }

    std::string A_path = argv[1];
    std::string C_path = argv[2];

    Mat A_in = imread(A_path, IMREAD_GRAYSCALE);
    Mat A(A_in.rows, A_in.cols, CV_64F);
    A_in.convertTo(A, CV_64F);
    Mat C = Mat::ones(A.rows,A.cols, CV_64F);
    Mat F(11,11, CV_64F, Scalar::all(1.0));

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
        []() {},
        [&]() {
            //cv::boxFilter(A, C, -1, Point(11, 11), Point(-1, -1), false, BORDER_CONSTANT);
            cv::filter2D(A, C, -1, F, Point(-1, -1), false, BORDER_CONSTANT);
        }
    );


    std::cout << time << std::endl;
    
    //cv::boxFilter(A, C, -1, Point(11, 11), Point(-1, -1), false, BORDER_CONSTANT);
    cv::filter2D(A, C, -1, F, Point(-1, -1), false, BORDER_CONSTANT);
    mtxWriteDense(C_path, C);
    return 0;
}
