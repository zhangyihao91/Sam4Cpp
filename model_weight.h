#ifndef _MODEL_WEIGHT_H_
#define _MODEL_WEIGHT_H_

#include <vector>
using namespace std;

extern vector<vector<double>> no_mask_embed_weight;
extern vector<vector<double>> point_embeddings_weight;
extern vector<vector<double>> positional_encoding_gaussian_matrix;
extern vector<vector<double>> not_a_point_embed_weight;
extern vector<float> pixel_mean;
extern vector<float> pixel_std;

#endif