#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>

#include <iostream>  
#include <onnxruntime_cxx_api.h>
#include <assert.h>
#include <vector>
#include <fstream>

#include "image_encoder.h"
#include "prompt_encoder.h"

using namespace cv;   
using namespace std;
using namespace Ort;
using namespace cv::dnn;


int main()
{
   PromptEncoder prompt_encoder;

   float base_x = 16;
   float base_y_1 = 10.6875;
   float base_y_2 = 32.0625;

    std::vector<std::vector<std::vector<float>>> tf_points(64, std::vector<std::vector<float>>(1, std::vector<float>(2, 0.0f)));
    for (int i = 0; i < 32; i++)
    {
        tf_points[i][0][0] = base_x + i*32;
        tf_points[i][0][1] = base_y_1;

    }
    for (int j = 0; j < 32; j++)
    {
        tf_points[j+32][0][0] = base_x + j*32;
        tf_points[j+32][0][1] = base_y_2;

    }

    Eigen::Tensor<double, 3> in_points(64, 1, 2);
   for (int i = 0; i < 64; i++){
      for (int j =0; j < 2; j++){
         in_points(i,0,j) = tf_points[i][0][j];
      }
   }

    Eigen::Tensor<double, 2> in_labels(64, 1);
    in_labels.setConstant(1.0);

   auto result = prompt_encoder.forward(in_points, in_labels);
   Eigen::Tensor<double, 3> sparse_embed = result.sparse_embedding;
   Eigen::Tensor<double, 4> dense_embed = result.dense_embedding;
   
   return 0; 
 
}
// --------Check image_pe
   // Eigen::Tensor<double, 3> image_pe = prompt_encoder.get_dense_pe();
   // Eigen::array<Eigen::Index, 4> new_dims = {1, 256, 64, 64};
   // Eigen::Tensor<double, 4> image_pe_new =  image_pe.reshape(new_dims);
   // for (int i =0; i < 64; ++i){
   //    cout << image_pe_new(0, 0, 0, i) <<  ", " <<endl;
   // }
   // return 0;

// --------Check 2 embeddings


// float base_x = 16;
// float base_y_1 = 10.6875;
// float base_y_2 = 32.0625;

//    std::vector<std::vector<std::vector<float>>> tf_points(64, std::vector<std::vector<float>>(1, std::vector<float>(2, 0.0f)));
//    for (int i = 0; i < 32; i++)
//    {
//       tf_points[i][0][0] = base_x + i*32;
//       tf_points[i][0][1] = base_y_1;

//    }
//    for (int j = 0; j < 32; j++)
//    {
//       tf_points[j+32][0][0] = base_x + j*32;
//       tf_points[j+32][0][1] = base_y_2;

//    }

//    Eigen::Tensor<double, 3> in_points(64, 1, 2);
// for (int i = 0; i < 64; i++){
//    for (int j =0; j < 2; j++){
//       in_points(i,0,j) = tf_points[i][0][j];
//    }
// }

//    Eigen::Tensor<double, 2> in_labels(64, 1);
//    in_labels.setConstant(1.0);


   
