#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
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
    if(argc != 4){
        std::cerr << "wrong number of arguments " << argc << std::endl;
    }

    std::string A_path = argv[1];
    std::string result_path = argv[3];

    Mat A = imread(A_path, IMREAD_GRAYSCALE);
    Mat C(1000,1000, CV_U8, 0);
    Mat F(11,11, CV_U8, 1);

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
        []() {},
        [&]() {
            filter2D(A, C, -1, F)
        }
    );


    std::cout << time << std::endl;
    
    mtxWriteDense(C_file, C);

    return 0;
}
