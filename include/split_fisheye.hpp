#ifndef SPLIT_FISHEYE_HPP
#define SPLIT_FISHEYE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>

class SplitFisheyeNode : public rclcpp::Node
{
public:
    explicit SplitFisheyeNode();
    ~SplitFisheyeNode();

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr dual_fisheye_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_pub_;
};

#endif // SPLIT_FISHEYE_HPP