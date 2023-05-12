#ifndef _IMAGE_ENCODER_H_
#define _IMAGE_ENCODER_H_

#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>  
#include <onnxruntime_cxx_api.h>
#include <assert.h>
#include <vector>

using namespace cv;   
using namespace std;
using namespace Ort;
using namespace cv::dnn;

Eigen::Tensor<float, 4> image_encoder(Ort::Session& session, Ort::MemoryInfo& memory_info,Eigen::Tensor<float, 4>& img);

#endif