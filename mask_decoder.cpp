#include "mask_decoder.h"

void mask_decoder(Eigen::Tensor<float, 4> image_features, Eigen::Tensor<float, 4> image_pe, Eigen::Tensor<float, 3> sparse_embedding, Eigen::Tensor<float, 4> dense_embedding, MaskDecoderOutput md_result)
{

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_example");

    // Initialize session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Create a session using the ONNX model
    Ort::Session session(env, "/home/intellif/workspace/sam4c/mask-decoder.onnx", session_options);  
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    vector<const char*> input_names = {"image_embeddings", "image_pe", "sparse_prompt_embeddings", "dense_prompt_embeddings"};
    vector<vector<int64_t>> input_shapes = {{1, 256, 64, 64}, {1, 256, 64, 64}, {64, 2, 256}, {64, 256, 64, 64}};

    vector<const char*> output_names = {"1132", "1137"};
    vector<vector<int64_t>> output_shapes = {{64, 3, 256, 256}, {64, 3}};


    Eigen::TensorMap<Eigen::Tensor<float, 4>> image_f_map(image_features.data(), 1, 256, 64, 64);
    Eigen::TensorMap<Eigen::Tensor<float, 4>> image_pe_map(image_pe.data(), 1, 256, 64, 64);
    Eigen::TensorMap<Eigen::Tensor<float, 3>> sparse_embedding_map(sparse_embedding.data(), 64, 2, 256);
    Eigen::TensorMap<Eigen::Tensor<float, 4>> dense_embedding_map(dense_embedding.data(), 1, 256, 64, 64);

    std::vector<Ort::Value> inputs;

    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, image_f_map.data(), image_features.size(), input_shapes[0].data(), input_shapes[0].size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, image_pe_map.data(), image_pe.size(), input_shapes[1].data(), input_shapes[1].size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, sparse_embedding_map.data(), sparse_embedding.size(), input_shapes[2].data(), input_shapes[2].size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, dense_embedding_map.data(), dense_embedding.size(), input_shapes[3].data(), input_shapes[3].size()));
    
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), 4, output_names.data(), output_names.size());

// // Convert ONNX output tensors to Eigen tensors
    Eigen::Tensor<float, 4> output_tensor1(Eigen::TensorMap<Eigen::Tensor<float, 4>>(output_tensors[0].GetTensorMutableData<float>(), output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[0], output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1], output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[2], output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[3]));
    Eigen::Tensor<float, 2> output_tensor2(Eigen::TensorMap<Eigen::Tensor<float, 2>>(output_tensors[1].GetTensorMutableData<float>(), output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()[0], output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()[1]));

    md_result.low_res_masks = output_tensor1;
    md_result.iou_prediction = output_tensor2;

}