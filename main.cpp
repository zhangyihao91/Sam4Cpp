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
    Ort::Session session(env, "/home/intellif/workspace/sam4c/encoder-matmul.onnx", session_options);

    // Get model input and output names and shapes
    vector<const char*> input_names = {"x"};
    vector<const char*> output_names = {"image_embeddings"};
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    Ort::TensorTypeAndShapeInfo input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    Ort::TensorTypeAndShapeInfo output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    vector<int64_t> input_shape = input_tensor_info.GetShape();
    vector<int64_t> output_shape = output_tensor_info.GetShape();
    int input_size = 1 * 3 * 1024 * 1024;

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
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, nullptr, output_shape[0], output_shape.data(), output_shape.size());

    // Run the model
    std::vector<Ort::Value> output;
    output = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 1);

// Get the output tensor data
    float* output_data = output[0].GetTensorMutableData<float>();

    // Process the output tensor data

    // Create a cv::Mat with the desired shape and copy the output tensor data to it
    Mat output_mat(1, output_shape[1], CV_32FC(output_shape[2] * output_shape[3]), output_data);

    // Reshape the output matrix to the desired shape
    output_mat = output_mat.reshape(1, { output_shape[1], output_shape[2], output_shape[3] });

    // Clean up
    delete[] input_data;
    cout << output_mat.size() << endl;
    return 0;
}
    