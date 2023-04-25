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

using namespace cv;   
using namespace std;
using namespace Ort;
using namespace cv::dnn;


int main()
{
    // Initialize the ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_example");

    // Initialize session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Create a session using the ONNX model
    Ort::Session session(env, "/home/intellif/workspace/sam4c/encoder-fix.onnx", session_options);

    // Get model input and output names and shapes
    vector<const char*> input_names = {"x"};
    vector<const char*> output_names = {"image_embeddings"};
    vector<int64_t> input_shape = {1, 3, 1024, 1024};
    vector<int64_t> output_shape = {1, 256, 64, 64};
    int input_size = 1 * 3 * 1024 * 1024;

    //wcout << "output_shape" << output_shape[0] << output_shape[1] << output_shape[2] << output_shape[3] << endl;

    // Load the image using OpenCV
    Mat img = imread("/home/intellif/workspace/sam4c/test.jpg");

    if (img.empty())
        {
            cout << "Could not read the image" << endl;
            return -1;
        }

    // Resize the image to the desired input shape
    Mat img_resized;
    resize(img, img_resized, Size(input_shape[3], input_shape[2]));

    // Convert the image to a float32 tensor with channel-first layout
    float* input_data = new float[input_size];
    for (int c = 0; c < 3; c++)
    {
        for (int h = 0; h < input_shape[2]; h++)
        {
            for (int w = 0; w < input_shape[3]; w++)
            {
                input_data[c * input_shape[2] * input_shape[3] + h * input_shape[3] + w] = static_cast<float>(img_resized.at<Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }

    // Create input and output tensors
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data, input_size, input_shape.data(), input_shape.size());
    // Run the model
    
    auto output = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 1);

// Get the output tensor data
    float* output_tensor_data = output[0].GetTensorMutableData<float>();
    
    std::vector<int64_t> output_dims(output_shape.begin(), output_shape.end());
    std::vector<float> output_data(output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3]);

    std::memcpy(output_data.data(), output_tensor_data, output_data.size() * sizeof(float));
  // Reshape the output data to a shape of 1x256x64x64 and store it in a vector
    std::vector<std::vector<std::vector<float>>> output_data_4d(1, std::vector<std::vector<float>>(256, std::vector<float>(64 * 64)));
    for (int i = 0; i < output_data.size(); ++i) {
        int n = i / (256 * 64 * 64);
        int c = (i / (64 * 64)) % 256;
        int h = (i /64 % 64);
        int w = i % 64;
        output_data_4d[n][c][h * 64 + w] = output_data[i];
        }
    // Print the first element of the output data
    // std::cout << "Output Data [0][0][0]: " << output_data_4d[0][0][0] << std::endl;
    // std::cout << "Output Data [0][0][0]: " << output_data_4d[0][0][1] << std::endl;
    // std::cout << "Output Data [0][0][0]: " << output_data_4d[0][0][2] << std::endl;
    // std::cout << "Output Data [0][0][0]: " << output_data_4d[0][0][3] << std::endl;
    // std::cout << "Output Data [0][0][0]: " << output_data_4d[0][0][4] << std::endl;
    // std::cout << "Output Data [0][0][0]: " << output_data_4d[0][0][5] << std::endl;

    // Clean up
    delete[] input_data;

    return 0;
}
