#include "image_encoder.h" 

Eigen::Tensor<float, 4> image_encoder(Ort::Session& session, Ort::MemoryInfo& memory_info, Eigen::Tensor<float, 4>& img)
    {   // Get model input and output names and shapes
    vector<const char*> input_names = {"x"};
    vector<const char*> output_names = {"image_embeddings"};
    vector<int64_t> input_shape = {1, 3, 1024, 1024};
    vector<int64_t> output_shape = {1, 256, 64, 64};
    int input_size = 1 * 3 * 1024 * 1024;

    // Create input and output tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, img.data(), input_size, input_shape.data(), input_shape.size());

    Eigen::Tensor<float, 4> output_tensor(1, 256, 64, 64);
    Ort::Value output_value = Ort::Value::CreateTensor<float>(memory_info, output_tensor.data(), output_tensor.size(),output_shape.data(), output_shape.size());

    session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), &output_value, 1);

// Get the output tensor dat

    return output_tensor;

}