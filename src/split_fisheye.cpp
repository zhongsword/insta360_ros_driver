#include "split_fisheye.hpp"
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <cmath>
#include <iostream>

SplitFisheyeNode::SplitFisheyeNode()
    : Node("split_fisheye_node")
{
    // Configure QoS (reliable, history depth=1)
    auto qos = rclcpp::QoS(1).reliable();

    // Subscribe to the dual fisheye image topic
    dual_fisheye_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/dual_fisheye/image", qos,
        std::bind(&SplitFisheyeNode::imageCallback, this, std::placeholders::_1));

    // Create publishers for left and right fisheye images
    left_pub_ = create_publisher<sensor_msgs::msg::Image>("/dual_fisheye/image/back", qos);
    right_pub_ = create_publisher<sensor_msgs::msg::Image>("/dual_fisheye/image/forward", qos);

    RCLCPP_INFO(get_logger(), "SplitFisheyeNode initialized. Subscribing to /dual_fisheye/image");
    RCLCPP_INFO(get_logger(), "Publishing left fish-eye to /dual_fisheye/image/back");
    RCLCPP_INFO(get_logger(), "Publishing right fish-eye to /dual_fisheye/image/forward");
}

SplitFisheyeNode::~SplitFisheyeNode()
{
}

void SplitFisheyeNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    try {
        // Convert ROS image to OpenCV Mat (RGB8 format)
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "rgb8");
        cv::Mat dual_img = cv_ptr->image;

        // Check image dimensions
        int img_height = dual_img.rows;
        int img_width = dual_img.cols;

        // Validate image dimensions (should be even width)
        if (img_width % 2 != 0) {
            RCLCPP_ERROR(get_logger(), "Image width %d is not even! Cannot split evenly.", img_width);
            return;
        }

        int mid = img_width / 2;

        cv::Mat left_img, right_img;

        // Split into left and right halves
        {
            // BACK: 顺时针旋转90度 (90° clockwise)
            cv::Mat temp_left_img = dual_img(cv::Rect(0, 0, mid, img_height));
            cv::rotate(temp_left_img, left_img, cv::ROTATE_90_CLOCKWISE); 
        }
        {
            // FORWARD: 逆时针旋转90度 (90° counter-clockwise)
            cv::Mat temp_right_img = dual_img(cv::Rect(mid, 0, mid, img_height));
            cv::rotate(temp_right_img, right_img, cv::ROTATE_90_COUNTERCLOCKWISE);
        }

        // Publish left fish-eye image
        auto left_msg = cv_bridge::CvImage(msg->header, "rgb8", left_img).toImageMsg();
        left_pub_->publish(*left_msg);

        // Publish right fish-eye image
        auto right_msg = cv_bridge::CvImage(msg->header, "rgb8", right_img).toImageMsg();
        right_pub_->publish(*right_msg);

        // Log processing info
        RCLCPP_DEBUG(get_logger(), "Split image: %dx%d -> back: %dx%d, forward: %dx%d",
                     img_width, img_height,
                     left_img.cols, left_img.rows,
                     right_img.cols, right_img.rows);
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error processing image: %s", e.what());
    }
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SplitFisheyeNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}