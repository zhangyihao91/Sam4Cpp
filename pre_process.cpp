#include "pre_process.h"

cv::Mat build_point_grid(int n_per_side) {
    float offset = 1.0 / (2 * n_per_side);
    cv::Mat points_one_side = cv::Mat(1, n_per_side, CV_32FC1);
    for (int i = 0; i < n_per_side; i++) {
        points_one_side.at<float>(0, i) = offset + i * (1.0 - 2 * offset) / (n_per_side - 1);
    }
    cv::Mat points_x = cv::Mat(n_per_side, n_per_side, CV_32FC1);
    cv::Mat points_y = cv::Mat(n_per_side, n_per_side, CV_32FC1);
    for (int i = 0; i < n_per_side; i++) {
        for (int j = 0; j < n_per_side; j++) {
            points_x.at<float>(i, j) = points_one_side.at<float>(0, j);
            points_y.at<float>(i, j) = points_one_side.at<float>(0, i);
        }
    }
    cv::Mat points = cv::Mat(n_per_side * n_per_side, 2, CV_32FC1);
    for (int i = 0; i < n_per_side; i++) {
        for (int j = 0; j < n_per_side; j++) {
            points.at<float>(i * n_per_side + j, 0) = points_x.at<float>(i, j);
            points.at<float>(i * n_per_side + j, 1) = points_y.at<float>(i, j);
        }
    }
    return points;
}


std::vector<cv::Mat> build_all_layer_point_grids(int n_per_side, int n_layers, int scale_per_layer){
    std::vector<cv::Mat> points_by_layer;
    for (int i = 0; i <= n_layers; i++) {
        int n_points = n_per_side / pow(scale_per_layer, i);
        auto point_grid =build_point_grid(n_points);
        points_by_layer.push_back(point_grid);
    }
    return points_by_layer;
};

boxes_container generate_crop_boxes(cv::Size im_size, int n_layers, float overlap_ratio) 
{
    /*
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    */
    vector<vector<int>> crop_boxes;
    vector<int> layer_idxs;
    int im_w = im_size.width;
    int im_h = im_size.height;
    int short_side = min(im_h, im_w);

    // Original image
    crop_boxes.push_back({0, 0, im_w, im_h});
    layer_idxs.push_back(0);

    auto crop_len = [](int orig_len, int n_crops, float overlap) -> int {
        return static_cast<int>(ceil((overlap * (n_crops - 1) + orig_len) / n_crops));
    };

    for (int i_layer = 0; i_layer < n_layers; ++i_layer) {
        int n_crops_per_side = pow(2, i_layer + 1);
        int overlap = static_cast<int>(overlap_ratio * short_side * (2.0 / n_crops_per_side));

        int crop_w = crop_len(im_w, n_crops_per_side, overlap);
        int crop_h = crop_len(im_h, n_crops_per_side, overlap);

        vector<int> crop_box_x0, crop_box_y0;
        for (int i = 0; i < n_crops_per_side; ++i) {
            crop_box_x0.push_back((crop_w - overlap) * i);
            crop_box_y0.push_back((crop_h - overlap) * i);
        }

        // Crops in XYWH format
        for (int x0 : crop_box_x0) {
            for (int y0 : crop_box_y0) {
                vector<int> box = {x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)};
                crop_boxes.push_back(box);
                layer_idxs.push_back(i_layer + 1);
            }
        }
    }

    boxes_container output;
    output.crop_boxes = crop_boxes;
    output.layer_idxs = layer_idxs;

    return output;
}



std::tuple<int, int> get_preprocess_shape(int oldh, int oldw, int long_side_length) {
    double scale = static_cast<double>(long_side_length) / std::max(oldh, oldw);
    int newh = static_cast<int>(oldh * scale + 0.5);
    int neww = static_cast<int>(oldw * scale + 0.5);
    return std::make_tuple(newh, neww);
}


cv::Mat apply_coords(cv::Mat coords, cv::Size original_size) {
    int old_h = original_size.height;
    int old_w = original_size.width;
    int new_h, new_w;
    tie(new_h, new_w) = get_preprocess_shape(old_h, old_w, 1024);

    cv::Mat new_coords(coords.size(), CV_32F);
    for (int i = 0; i < coords.rows; i++){
        new_coords.at<float>(i, 0) = coords.at<float>(i, 0) * new_w / old_w ;
        new_coords.at<float>(i, 1) = coords.at<float>(i, 1) * new_h / old_h;
    }
    return new_coords;
}

Eigen::Tensor<float, 4> preprocess_image(cv::Mat img){
    int target_size = 1024;
    cv::Size orig_size = img.size();
    int orig_h = orig_size.height; //534
    int orig_w = orig_size.width;  //800


    Eigen::Tensor<float ,3> pixel_mean_tensor(1, 1, 3);
    Eigen::Tensor<float, 3> pixel_std_tensor(1, 1, 3);

    for (int i = 0; i < 3; i++){
        pixel_mean_tensor(0,0, i) = pixel_mean[i];
        pixel_std_tensor(0, 0, i) = pixel_std[i];
    }

    int mid_w = orig_h * target_size / orig_w + 1;

    cv::Mat mid_image;
    cv::resize(img, mid_image, cv::Size(target_size, mid_w));

    Eigen::Tensor<float, 3> tensor(mid_w, target_size, 3);

    // loop over the rows and columns of the image, and copy the pixel values to the EigenTensor
    for (int row = 0; row < mid_w; row++) {
        for (int col = 0; col < target_size; col++) {
            cv::Vec3b pixel = mid_image.at<cv::Vec3b>(row, col);
            tensor(row, col, 0) = pixel[2]; // red channel
            tensor(row, col, 1) = pixel[1]; // green channel
            tensor(row, col, 2) = pixel[0]; // blue channel
        }
    }

    Eigen::array<Eigen::Index, 3> bcast({mid_w, target_size, 1});
    Eigen::Tensor<float, 3> casted_pixel_mean = pixel_mean_tensor.broadcast(bcast);
    Eigen::Tensor<float, 3> casted_pixel_std = pixel_std_tensor.broadcast(bcast); 

    tensor = (tensor - casted_pixel_mean)/casted_pixel_std;

    int w = tensor.dimension(1);
    int pad_w = target_size - w;

    int h = tensor.dimension(0);
    int pad_h = target_size - h;

    Eigen::array<Eigen::IndexPair<int>, 3> padding = { Eigen::IndexPair<int>(0, pad_h), Eigen::IndexPair<int>(0, pad_w), Eigen::IndexPair<int>(0, 0) };

    // pad the tensor
    Eigen::Tensor<float, 3> padded_tensor = tensor.pad(padding, 0.0f);
    Eigen::DSizes<Eigen::DenseIndex, 4> t_dim(1, 3, target_size ,target_size);

    Eigen::Tensor<float, 4> transformed_tensor = padded_tensor.reshape(t_dim);
    return transformed_tensor;
}