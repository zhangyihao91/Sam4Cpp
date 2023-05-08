#ifndef _MASK_DECODER_H_
#define _MASK_DECODER_H_

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

typedef struct MaskDecoderOutput{
    Eigen::Tensor<float, 3> low_res_masks;
    Eigen::Tensor<float, 2> iou_prediction;
}MaskDecoderOutput;

void mask_decoder(Eigen::Tensor<float, 4> image_features, Eigen::Tensor<float, 4> image_pe, Eigen::Tensor<float, 3> sparse_embedding, Eigen::Tensor<float, 4> dense_embedding, MaskDecoderOutput md_result);

#endif