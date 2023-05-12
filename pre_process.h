#ifndef _PRE_PROCESS_H_
#define _PRE_PROCESS_H_

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>
#include <cmath>
#include <algorithm>
#include <tuple>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "model_weight.h"

using namespace std;
using namespace cv;

typedef struct boxes_container{
    vector<vector<int>> crop_boxes;
    vector<int> layer_idxs;
}boxes_container;


cv::Mat build_point_grid(int n_per_side);
std::vector<cv::Mat> build_all_layer_point_grids(int n_per_side, int n_layers, int scale_per_layer);
boxes_container generate_crop_boxes(cv::Size im_size, int n_layers, float overlap_ratio);

cv::Mat apply_coords(cv::Mat coords, cv::Size original_size);
std::tuple<int, int> get_preprocess_shape(int oldh, int oldw, int long_side_length);

Eigen::Tensor<float, 4> preprocess_image(cv::Mat img);

#endif