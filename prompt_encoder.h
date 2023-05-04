#ifndef _PROMPT_ENCODER_H_
#define _PROMPT_ENCODER_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
using namespace Eigen;
using namespace std;

class PositionEmbeddingRandom {
public:
    PositionEmbeddingRandom(int num_pos_feats = 64, float scale = 1.0);
    Tensor<double, 3>  _pe_encoding(Eigen::Tensor<double, 3> coords);
    Tensor<double, 3>  forward(Eigen::Index h, Eigen::Index w); 
    Tensor<double, 3>  forward_with_coords(Eigen::Tensor<double, 3> coords_input, Eigen::Index h, Eigen::Index w); 

private:
    int num_pos_feats_;
    float scale_;
};

class PromptEncoder 
{
public:
    PromptEncoder();

    PositionEmbeddingRandom pe_layer;
    int embed_dim;
    int input_image_size;
    Eigen::Index image_embed_size;
    int num_point_embed;

    vector<vector<double>> napew;
    vector<vector<double>> nmew;
    vector<vector<double>> pew;

    typedef struct TwoEmbeddingResult{
       Eigen::Tensor<double , 3> sparse_embedding;
       Eigen::Tensor<double, 4> dense_embedding;
    }TwoEmbeddingResult;
    
    Eigen::Tensor<double, 4> get_dense_pe();
    Eigen::Tensor<double, 3> _embed_points(Tensor<double, 3> in_points, Tensor<double, 2> in_labels);
    TwoEmbeddingResult forward(Tensor<double, 3> in_points, Tensor<double, 2> in_labels);
    
};


#endif