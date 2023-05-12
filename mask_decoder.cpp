#include "mask_decoder.h"

void mask_decoder(Ort::Session& session, Ort::MemoryInfo& memory_info, Eigen::Tensor<float, 4>& image_features, Eigen::Tensor<float, 4>& image_pe, Eigen::Tensor<float, 3>& sparse_embedding, Eigen::Tensor<float, 4>& dense_embedding, MaskDecoderOutput md_result)
{
    vector<const char*> input_names = {"image_embeddings", "image_pe", "sparse_prompt_embeddings", "dense_prompt_embeddings"};
    vector<vector<int64_t>> input_shapes = {{1, 256, 64, 64}, {1, 256, 64, 64}, {64, 2, 256}, {64, 256, 64, 64}};

    vector<const char*> output_names = {"1132", "1137"};
    vector<vector<int64_t>> output_shapes = {{64, 3, 256, 256}, {64, 3}};

    std::vector<Ort::Value> inputs;

    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, image_features.data(), std::accumulate(input_shapes[0].begin(), input_shapes[0].end(), 1, std::multiplies<int64_t>()), input_shapes[0].data(), input_shapes[0].size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, image_pe.data(), std::accumulate(input_shapes[1].begin(), input_shapes[1].end(), 1, std::multiplies<int64_t>()), input_shapes[1].data(), input_shapes[1].size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, sparse_embedding.data(), std::accumulate(input_shapes[2].begin(), input_shapes[2].end(), 1, std::multiplies<int64_t>()), input_shapes[2].data(), input_shapes[2].size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, dense_embedding.data(), std::accumulate(input_shapes[3].begin(), input_shapes[3].end(), 1, std::multiplies<int64_t>()), input_shapes[3].data(), input_shapes[3].size()));
    
    vector<Ort::Value> output_tensors;
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), input_names.size(), output_names.data(), output_names.size());

// // Convert ONNX output tensors to Eigen tensors
    Eigen::Tensor<float, 4> output_tensor1(Eigen::TensorMap<Eigen::Tensor<float, 4>>(output_tensors[0].GetTensorMutableData<float>(), output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[0], output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1], output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[2], output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[3]));
    Eigen::Tensor<float, 2> output_tensor2(Eigen::TensorMap<Eigen::Tensor<float, 2>>(output_tensors[1].GetTensorMutableData<float>(), output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()[0], output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()[1]));

    md_result.low_res_masks = output_tensor1;
    md_result.iou_prediction = output_tensor2;

}