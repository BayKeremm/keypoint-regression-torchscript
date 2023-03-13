#include <torch/script.h> // One-stop header.

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include<string>
#include <iostream>
#include <memory>

std::string get_image_type(const cv::Mat& img, bool more_info=true) 
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');
   
    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    std::cout << r << "\n";
    return r;
}

void show_image(cv::Mat& img, std::string title)
{
    std::string image_type = get_image_type(img);
    cv::imshow(title + " type:" + image_type, img);
    cv::waitKey(0);
}

auto ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze=false, int unsqueeze_dim = 0)
{
    std::cout << "image shape: " << img.size() << std::endl;
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kFloat32);

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
        std::cout << "tensors new shape: " << tensor_image.sizes() << std::endl;
    }
    
    if (show_output)
    {
        std::cout << tensor_image.slice(2, 0, 1) << std::endl;
    }
    std::cout << "tenor shape: " << tensor_image.sizes() << std::endl;
    return tensor_image;
}
auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 })
{
    std::cout << "############### transpose ############" << std::endl;
    std::cout << "shape before : " << tensor.sizes() << std::endl;
    tensor = tensor.permute(dims);
    std::cout << "shape after : " << tensor.sizes() << std::endl;
    std::cout << "######################################" << std::endl;
    return tensor;
}

auto ToInput(at::Tensor tensor_image)
{
    // Create a vector of inputs.
    return std::vector<torch::jit::IValue>{tensor_image};
}
int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module rektnet;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        rektnet = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "model ok\n";


    // get image
    std::string image_filepath = "../test_kpt.png";
    cv::Mat image = cv::imread(image_filepath);
    cv::resize(image,image,cv::Size(80,80));
    cv::Mat newImage;
    cv::cvtColor( image , image , cv::COLOR_RGB2BGR );//opencv uses bgr
    image.convertTo(newImage, CV_32FC3, 1.0 / 255.0);


    // convert the cvimage into tensor
    auto tensor = ToTensor(newImage);
    tensor = transpose(tensor, { (2),(0),(1) });
    //add batch dim (an inplace operation just like in pytorch)
    tensor.unsqueeze_(0);
    std::cout << "shape after tensor conversion : " << tensor.sizes() << std::endl;


    std::vector<torch::jit::IValue> input_to_net = ToInput(tensor);

    //forward the sucker
    auto output = rektnet.forward(input_to_net).toTuple();
    auto points = output->elements()[1].toTensor(); 
    std::cout <<  output << "\n";
    for(int i=0; i < 7 ; i++){
        auto cord = points[0][i]*80;
        int x = cord[0].item<int>();
        int y = cord[1].item<int>();
        cv::circle(image, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
    }
    
    show_image(image, "input");


    return 0;
}