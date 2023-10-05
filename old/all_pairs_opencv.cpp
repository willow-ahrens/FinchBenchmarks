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

    // A = x*B + (1-x)*C
    std::string folder = argv[1];
    int range_upper = std::stoi(argv[2]);
    std::string result = argv[3];

    std::vector<Mat> imgs(range_upper);
    for (int i = 1; i <= range_upper; i++){
        std::string img_path = folder + std::to_string(i) + ".png";
        // std::cout << "Loading " << img_path << std::endl;
        imgs[i-1] = imread(img_path, IMREAD_GRAYSCALE);
        if( imgs[i-1].empty() ) { std::cout << "Error loading " << img_path << std::endl; return EXIT_FAILURE; }
    }

    Mat dst(range_upper,range_upper, CV_64F, 0.0);

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
        []() {},
        [&]() {
            for (int k =0; k<range_upper; k++){
                for (int l = k+1; l<range_upper; l++){
                    dst.at<double>(k,l) = norm(imgs[k], imgs[l], NORM_L2);
                }
            }
        }
    );

    std::cout << time << std::endl;
    
    mtxWriteDense(result, dst);

    return 0;
}
