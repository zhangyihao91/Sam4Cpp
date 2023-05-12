#include <Eigen/Core>
#include <Eigen/Dense>
#include "prompt_encoder.h"
#include "model_weight.h"

//vector<vector<double>> no_mask_embed_weight;

PositionEmbeddingRandom::PositionEmbeddingRandom(int num_pos_feats, float scale) : num_pos_feats_(num_pos_feats), scale_(scale) {}

Eigen::Tensor<double, 4> PositionEmbeddingRandom::forward(Eigen::Index h, Eigen::Index w) {
    Eigen::Tensor<double, 2> grid(h, w);
    grid.setConstant(1.0);
    Eigen::Tensor<double, 2> y_embed = grid.cumsum(Eigen::Index(0)) - 0.5;
    Eigen::Tensor<double, 2> x_embed = grid.cumsum(Eigen::Index(1)) - 0.5;
    double h_num = static_cast<double>(h);
    double w_num = static_cast<double>(w);
    y_embed = y_embed/h_num;
    x_embed = x_embed/w_num;

    Eigen::DSizes<Eigen::DenseIndex, 3> t_dim(64 ,64, 1);
    Eigen::Tensor<double, 3> coords = x_embed.reshape(t_dim).concatenate(y_embed.reshape(t_dim), 2);
    
    Eigen::Tensor<double, 3> result = _pe_encoding(coords);
    Eigen::Tensor<double, 3> image_pe = result.shuffle(Eigen::array<Eigen::Index, 3>({ 2, 0, 1 }));
    Eigen::Tensor<double, 4> image_pe_output;
    Eigen::DSizes<Eigen::DenseIndex, 4> out_dim(1, 256 ,64, 64);
    image_pe_output = image_pe.reshape(out_dim);
    return image_pe_output;

}

Eigen::Tensor<double, 3> PositionEmbeddingRandom::forward_with_coords(Eigen::Tensor<double, 3> coords_input, Eigen::Index h, Eigen::Index w) {
    
    double h_num = static_cast<double>(h);
    double w_num = static_cast<double>(w);
    
    coords_input.chip(0, 2) = coords_input.chip(0, 2)/w_num;
    coords_input.chip(1, 2) = coords_input.chip(1, 2)/h_num;
    
    return _pe_encoding(coords_input);
}


Eigen::Tensor<double, 3> PositionEmbeddingRandom::_pe_encoding(Eigen::Tensor<double, 3> coords){
    Tensor<double, 2> pegm(2, 128);
    for (int i = 0; i < 2; ++i){
        for (int j=0; j<128; ++j){
            pegm(i, j) = positional_encoding_gaussian_matrix[i][j];
        }
    }
    coords = 2.0 * coords - 1.0;

    //Eigen::DSizes<Eigen::DenseIndex, 3> p_dim(2 ,128, 1);
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = { Eigen::IndexPair<int>(2, 0) };
    Eigen::Tensor<double, 3> inter_res = coords.contract(pegm, contract_dims);

    const double pi = std::acos(-1);
    Eigen::Tensor<double, 3> coords_new = 2.0 * pi * inter_res;

    Eigen::Tensor<double, 3> sin_coords = coords_new.unaryExpr([](double x) { return std::sin(x); });
    Eigen::Tensor<double, 3> cos_coords = coords_new.unaryExpr([](double x) { return std::cos(x); });

    Eigen::Tensor<double, 3> pe =sin_coords.concatenate(cos_coords, 2);

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
    this->pew = point_embeddings_weight;
 }

Eigen::Tensor<float, 4> PromptEncoder::get_dense_pe()
{
   Eigen::Tensor<double, 4> double_pe = this->pe_layer.forward(this->image_embed_size, this->image_embed_size);
   return double_pe.cast<float>();
}

Eigen::Tensor<double, 3> PromptEncoder::_embed_points(Tensor<double, 3> in_points, Tensor<double, 2> in_labels)
{
    /// try to rewrite this code easy, remember if result is incorrect, check if the problem is here
    in_points = in_points + 0.5;

    // Pad the points tensor with zeros and labels tensor with -1
    Eigen::Tensor<double, 3> padding_point(in_points.dimension(0), 1, 2);
    Eigen::Tensor<double, 2> padding_label(in_labels.dimension(0), 1);

    padding_point.setConstant(0.0);
    padding_label.setConstant(-1.0);

    Eigen::Tensor<double, 3> new_point(in_points.dimension(0), in_points.dimension(1), in_points.dimension(2) + padding_point.dimension(2));
    Eigen::Tensor<double, 2> new_label(in_labels.dimension(0), in_labels.dimension(1)+ padding_label.dimension(1));

    new_point = in_points.concatenate(padding_point, 1);
    new_label = in_labels.concatenate(padding_label, 1);

    Eigen::Tensor<double, 3> point_embedding = this->pe_layer.forward_with_coords(new_point, this->input_image_size = 1024, this->input_image_size = 1024);

    Eigen::Tensor<double ,2> nape_weight(1, 256);
    for (int i = 0; i< 256; i++){
        nape_weight(0, i) = not_a_point_embed_weight[0][i];
    }


    Eigen::Tensor<double ,2> pe_weight(1, 256);
    for (int i = 0; i< 256; i++){
        pe_weight(0, i) = point_embeddings_weight[0][i];
    }

    Eigen::DSizes<Eigen::DenseIndex, 2> mid_dim(64, 256);

    Eigen::array<Eigen::Index, 2> sizes = {{64, 256}};

    Eigen::Tensor<double, 3> v_pe = point_embedding.slice(Eigen::array<Eigen::Index, 3>({0, 0, 0}), Eigen::array<Eigen::Index, 3>({64, 1, 256}));
    Eigen::Tensor<double, 3> v_nape = point_embedding.slice(Eigen::array<Eigen::Index, 3>({0, 1, 0}), Eigen::array<Eigen::Index, 3>({64, 1, 256}));

    Eigen::Tensor<double, 2> tensor_pe = v_pe.reshape(mid_dim);
    
    Eigen::Tensor<double, 2> tensor_nape = v_nape.reshape(mid_dim);
    tensor_nape.setZero();

    Eigen::Tensor<double, 2> broadcast_nape = nape_weight.broadcast(Eigen::array<Eigen::Index, 2>({64, 1}));
    Eigen::Tensor<double, 2> broadcast_pe = pe_weight.broadcast(Eigen::array<Eigen::Index, 2>({64, 1}));

    Eigen::Tensor<double, 2> mid_var_1 = tensor_pe + broadcast_pe;
    Eigen::Tensor<double, 2> mid_var_2 = tensor_nape + broadcast_nape;

    Eigen::DSizes<Eigen::DenseIndex, 3> pe_dim(64 , 1, 256);
    Eigen::Tensor<double, 3> mid_var_1_1 = mid_var_1.reshape(pe_dim);
    Eigen::Tensor<double, 3> mid_var_2_1 = mid_var_2.reshape(pe_dim);

    Eigen::Tensor<double, 3> output_point_embed = mid_var_1_1.concatenate(mid_var_2_1, 1);

    return output_point_embed;
}


PromptEncoder::TwoEmbeddingResult PromptEncoder::forward(Tensor<double, 3> in_points, Tensor<double, 2> in_labels)
{
    PromptEncoder::TwoEmbeddingResult output;

    //Eigen::Tensor<double, 3> parse_embeddings(64, 0, this->embed_dim);
    //parse_embeddings.setZero();
    Eigen::Tensor<double, 3> point_embeddings = this->_embed_points(in_points, in_labels);
    output.sparse_embedding = point_embeddings;

    Eigen::TensorMap<Eigen::Tensor<double, 4>> weight_tensor(no_mask_embed_weight[0].data(), 1, 256, 1, 1);

    // Use the broadcast method to expand the tensor along the second dimension
    Eigen::array<Eigen::Index, 4> bcast({64, 1, 64, 64});
    output.dense_embedding = weight_tensor.broadcast(bcast);

    return output;
}