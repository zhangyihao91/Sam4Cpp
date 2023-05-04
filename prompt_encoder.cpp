#include <Eigen/Core>
#include <Eigen/Dense>
#include "prompt_encoder.h"
#include "model_weight.h"

//vector<vector<double>> no_mask_embed_weight;

PositionEmbeddingRandom::PositionEmbeddingRandom(int num_pos_feats, float scale) : num_pos_feats_(num_pos_feats), scale_(scale) {}

Eigen::Tensor<double, 3> PositionEmbeddingRandom::forward(Eigen::Index h, Eigen::Index w) {
    Eigen::Tensor<double, 2> grid(h, w);
    grid.setConstant(1.0);
    Eigen::Tensor<double, 2> y_embed = grid.cumsum(Eigen::Index(0)) - 0.5;
    Eigen::Tensor<double, 2> x_embed = grid.cumsum(Eigen::Index(1)) - 0.5;
    double h_num = static_cast<double>(h);
    double w_num = static_cast<double>(w);
    y_embed = y_embed/h_num;
    x_embed = x_embed/w_num;

    Eigen::Tensor<double, 3> coords(h, w, 2);
    coords.chip(0, 2) = x_embed;
    coords.chip(1, 2) = y_embed;

    return _pe_encoding(coords).shuffle(Eigen::array<Eigen::Index, 3>({ 2, 0, 1 }));
}

Eigen::Tensor<double, 3> PositionEmbeddingRandom::forward_with_coords(Eigen::Tensor<double, 3> coords_input, Eigen::Index h, Eigen::Index w) {
    
    double h_num = static_cast<double>(h);
    double w_num = static_cast<double>(w);
    
    coords_input.chip(0, 2) = coords_input.chip(0, 2)/w_num;
    coords_input.chip(1, 2) = coords_input.chip(1, 2)/h_num;
    return _pe_encoding(coords_input.cast<double>()).shuffle(Eigen::array<Eigen::Index, 3>({ 2, 0, 1 }));
}


Eigen::Tensor<double, 3> PositionEmbeddingRandom::_pe_encoding(Eigen::Tensor<double, 3> coords) {
    Eigen::Index C = num_pos_feats_ * 2;
    Eigen::TensorMap<Tensor<double, 2>> nmew(no_mask_embed_weight[0].data(), no_mask_embed_weight.size(), no_mask_embed_weight[0].size());
    coords = 2.0 * coords - 1.0;
    coords = coords * nmew;
    coords = 2.0 * M_PI * coords;

    Eigen::Tensor<double, 3> sin_coords = coords.unaryExpr([](double x) { return std::sin(x); });
    Eigen::Tensor<double, 3> cos_coords = coords.unaryExpr([](double x) { return std::cos(x); });

    Eigen::Tensor<double, 3> pe(coords.dimension(0), coords.dimension(1), sin_coords.dimension(2) + cos_coords.dimension(2));
    pe.slice(Eigen::array<Eigen::Index, 2>{0, 0}, Eigen::array<Eigen::Index, 2>{coords.dimension(0), coords.dimension(1)}) = sin_coords;
    pe.slice(Eigen::array<Eigen::Index, 2>{0, sin_coords.dimension(2)}, Eigen::array<Eigen::Index, 2>{coords.dimension(0), coords.dimension(1)}) = cos_coords;

    return pe;
}



PromptEncoder::PromptEncoder(){
    this->embed_dim = 256;
    this->input_image_size = 1024;
    this->image_embed_size = 64;
    this->num_point_embed = 64;
    
    this->pe_layer = PositionEmbeddingRandom(this->embed_dim / 2);

    this->napew = not_a_point_embed_weight;
    this->nmew = no_mask_embed_weight;
    this->pew = point_embeddings;
 }

Eigen::Tensor<double, 4> PromptEncoder::get_dense_pe()
{
    this->pe_layer.forward(this->image_embed_size, this->image_embed_size);
}

Eigen::Tensor<double, 3> PromptEncoder::_embed_points(Tensor<double, 3> in_points, Tensor<double, 2> in_labels)
{
    in_points = in_points + 0.5;

    // Pad the points tensor with zeros and labels tensor with -1
    Eigen::Tensor<double, 3> padding_point(in_points.dimension(0), 1, 2);
    Eigen::Tensor<double, 2> padding_label(in_labels.dimension(0), 1);

    padding_point.setConstant(0.0);
    padding_point.setConstant(-1.0);

    Eigen::Tensor<double, 3> new_point(in_points.dimension(0), in_points.dimension(1), in_points.dimension(2) + padding_point.dimension(2));
    Eigen::Tensor<double, 2> new_label(in_labels.dimension(0), in_labels.dimension(1)+ padding_label.dimension(1));

    new_point = in_points.concatenate(padding_point, 2);

    return new_point;
}


PromptEncoder::TwoEmbeddingResult PromptEncoder::forward(Tensor<double, 3> in_points, Tensor<double, 2> in_labels)
{
    PromptEncoder::TwoEmbeddingResult output;

    int bs = 1;
    Eigen::Tensor<double, 3> parse_embeddings(bs, 0, this->embed_dim);
    parse_embeddings.setZero();
    Eigen::Tensor<double, 3> point_embeddings = this->_embed_points(in_points, in_labels);
    output.sparse_embedding = parse_embeddings.concatenate(point_embeddings, 1);


    Eigen::TensorMap<Eigen::Tensor<double, 4>> weight_tensor(no_mask_embed_weight[0].data(), 1, 256, 1, 1);
    weight_tensor = weight_tensor.reshape(Eigen::array<Eigen::Index, 4>({1, 256, 1, 1}));

    // Use the broadcast method to expand the tensor along the second dimension
    Eigen::array<Eigen::Index, 4> bcast({1, 1, 64, 64});
    output.dense_embedding = weight_tensor.broadcast(bcast);

    return output;
}