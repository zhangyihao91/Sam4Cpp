#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>

#include <iostream>  
#include <assert.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "pre_process.h"
#include "post_process.h"
#include "image_encoder.h"
#include "prompt_encoder.h"
#include "mask_decoder.h"



using namespace cv;   
using namespace std;
using namespace Ort;
using namespace cv::dnn;


int main()
{
   PromptEncoder prompt_encoder;

   int points_per_side = 32;
   int points_per_batch = 64;
   float pred_iou_thresh = 0.88;
   float stability_score_thresh = 0.95;
   float stability_score_offset = 1.0;
   float box_nms_thresh = 0.7;
   int crop_n_layers = 0;
   float crop_nms_thresh = 0.7;
   float crop_overlap_ratio = 512 / 1500;
   int crop_n_points_downscale_factor = 1;
   int min_mask_region_area = 0;
   
   string img_path = "/home/intellif/workspace/sam4c/dog.jpg";
   Mat input_img = imread(img_path, cv::IMREAD_COLOR);
   Eigen::Tensor<float, 4> test = preprocess_image(input_img);


   static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example");

    // Initialize session options
   Ort::SessionOptions session_options;
   session_options.SetIntraOpNumThreads(1);
   session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

   Ort::MemoryInfo memory_info_1 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
   Ort::MemoryInfo memory_info_2 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

   Ort::Session session_1(env, "/home/intellif/workspace/sam4c/encoder-fix.onnx", session_options);
   Ort::Session session_2(env, "/home/intellif/workspace/sam4c/mask-decoder.onnx", session_options);

   Eigen::Tensor<float, 4> image_features(1, 256, 64, 64);

   image_features = image_encoder(session_1, memory_info_1, test);
   
   for (int i = 0; i < 64; i++){
      cout << image_features(0,0,0,i) << endl;
   }
   return 0; 
}



   // vector<cv::Mat> point_grids = build_all_layer_point_grids(points_per_side, crop_n_layers, crop_n_points_downscale_factor); 

   // Mat input_img = imread(img_path, cv::IMREAD_COLOR);
   //     if (input_img.empty()) {
   //      cout << 'Image could not be read' << endl;
   //      return -1;
   //  }

   //  // Get image size
   //  cv::Size orig_size = input_img.size();
   // //  int width = orig_size.width;
   // //  int height = orig_size.height;

    
   // boxes_container cropped_output = generate_crop_boxes(orig_size,  crop_n_layers, crop_overlap_ratio );
   // vector<vector<int>> crop_boxes = cropped_output.crop_boxes;
   // vector<int> layer_idxs = cropped_output.layer_idxs;

   // for (int i = 0; i < layer_idxs.size(); i++){
   //    auto crop_box = crop_boxes[i];
   //    auto layer_idx = layer_idxs[i];

   //    int x0 = crop_box[0];
   //    int y0 = crop_box[1];
   //    int x1 = crop_box[2];
   //    int y1 = crop_box[3];

   //    cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
   //    cv::Mat cropped_im = input_img(roi);

   //    auto cropped_im_size = cropped_im.size();



   //    Eigen::Tensor<float, 4> image_pe = prompt_encoder.get_dense_pe();

   //    int image_h = cropped_im_size.height;
   //    int image_w = cropped_im_size.width;

   //    cv::Mat points_grid = point_grids[layer_idx];

   //    cv::Mat points_for_image(points_grid.size(), CV_32F);

   //    for (int i =0; i < points_grid.rows; i++){
   //       points_for_image.at<float>(i, 0) = points_grid.at<float>(i, 0) * image_w;
   //       points_for_image.at<float>(i, 1) = points_grid.at<float>(i, 1) * image_h;
   //    }

   //    cv::Mat batch_data;
   //    cv::Mat transformed_points;

   //    // int n_rows = points_for_image.rows;
   //    // int n_cols = points_for_image.cols;

   //    int n_rows = 64;
   //    int n_cols = 64;

   //    MaskDecoderOutput test_result;

   //    // for (int i = 0; i < n_rows; i += 64) {
   //    //    for (int j = 0; j < n_cols; j += 2) {
   //          // take out a (64, 2) block of elements
   //    batch_data = points_for_image(cv::Range(0, 64), cv::Range(0, 2));
   //    transformed_points = apply_coords(batch_data, orig_size);
   //    Eigen::TensorMap<Eigen::Tensor<float, 2>> in_points(transformed_points.ptr<float>(), transformed_points.rows, transformed_points.cols);
   //    Eigen::Tensor<float, 2> in_labels(transformed_points.rows, 1);
   //    in_labels.setConstant(1.0);

   //    Eigen::DSizes<Eigen::DenseIndex, 3> t_dim(64, 1, 2);
   //    Eigen::Tensor<double, 3> in_points_fixed = in_points.cast<double>().reshape(t_dim);

   //    PromptEncoder::TwoEmbeddingResult Embed = prompt_encoder.forward(in_points_fixed, in_labels.cast<double>());

   //    Eigen::Tensor<float, 3> sparse_embedding = Embed.sparse_embedding.cast<float>();
   //    Eigen::Tensor<float, 4> dense_embedding = Embed.dense_embedding.cast<float>();

   //    mask_decoder(session_2, memory_info_2, image_features, image_pe, sparse_embedding, dense_embedding, test_result);

   //    Eigen::Tensor<float, 2> iou_prediction = test_result.iou_prediction;
   //    Eigen::Tensor<float, 4> low_res_masks = test_result.low_res_masks;
   // }