This repo is working on FacebookResearch/Segment-anything.

Currently is transpose the python code into Cpp version only with opencv&onnx lib.

I will update the repo after finishing each module of this model.

If anyone hope to use this repo, only to sudo apt-get install opencv and install onnxruntime C++ lib in your local environment.

Best

2023.04.24 Update the image encoder module, next step need to fix the input shape and output shape of image encoder.onnx file and use real input to check if the c++ code inference correct.

To use this code, please install Opencv4 and onnxruntime-cpu by compile from source code.

and also need to download the encoder-data.bin file and put it into the same path of encoder-matmul.onnx.

TO BE CONTINUE~~
