#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <atomic>
#include <algorithm>

#include <camera/camera.h>
#include <camera/photography_settings.h>
#include <camera/device_discovery.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/imu.hpp"

class TestStreamDelegate : public ins_camera::StreamDelegate {
private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;

    // Cache for SPS and PPS to prepend to frames if missing
    std::vector<uint8_t> sps_nal_;
    std::vector<uint8_t> pps_nal_;
    bool have_sps_pps_ = false;

    // Helper to find NAL units and extract SPS/PPS
    void ExtractSpsPps(const uint8_t* data, size_t size) {
        const uint8_t* end = data + size;
        const uint8_t* p = data;
        
        auto find_start = [](const uint8_t* cur, const uint8_t* end) -> const uint8_t* {
            const uint8_t* p2 = cur;
            while (p2 + 3 < end) {
                if (p2[0] == 0x00 && p2[1] == 0x00 && 
                   ((p2[2] == 0x01) || (p2[2] == 0x00 && p2 + 4 < end && p2[3] == 0x01))) {
                    return p2;
                }
                ++p2;
            }
            return end;
        };

        while (p < end) {
            const uint8_t* sc = find_start(p, end);
            if (sc == end) break;
            
            const uint8_t* nal_start = sc;
            size_t sc_size = (sc + 3 < end && sc[2] == 0x01) ? 3 : 4;
            nal_start += sc_size;
            
            if (nal_start >= end) break;
            const uint8_t* next_sc = find_start(nal_start, end);
            const uint8_t* nal_end = next_sc;
            
            if (nal_end <= nal_start) break;

            uint8_t nal_type = nal_start[0] & 0x1F;
            if (nal_type == 7) { // SPS
                sps_nal_.assign(nal_start, nal_end);
            } else if (nal_type == 8) { // PPS
                pps_nal_.assign(nal_start, nal_end);
            }
            p = nal_end;
        }
        have_sps_pps_ = !sps_nal_.empty() && !pps_nal_.empty();
    }

public:
    TestStreamDelegate(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {
        // Publisher for the compressed H.264 video stream
        compressed_pub_ = node_->create_publisher<sensor_msgs::msg::CompressedImage>(
            "/dual_fisheye/image/compressed", 
            rclcpp::SensorDataQoS()
        );

        // Publisher for IMU data (remains the same)
        imu_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", rclcpp::SensorDataQoS());
        RCLCPP_INFO(node_->get_logger(), "Publisher for compressed images and IMU created.");
    }

    virtual ~TestStreamDelegate() {}

    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {}

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index) override {
        // We only care about the main video stream (index 0)
        if (stream_index == 0 && size > 0 && compressed_pub_) {
            // Extract SPS/PPS from the current frame if present
            ExtractSpsPps(data, size);

            auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();

            // Set the header with accurate timestamp from camera
            msg->header.stamp.sec = timestamp / 1000000;  // Convert microseconds to seconds
            msg->header.stamp.nanosec = (timestamp % 1000000) * 1000;  // Remaining to nanoseconds
            msg->header.frame_id = "camera_frame";

            // Set the format to H.264 (based on current resolution < 5.7k)
            msg->format = "h264";

            // If we have cached SPS/PPS, prepend them to ensure every frame has parameter sets
            std::vector<uint8_t> packet_data;
            static const uint8_t start_code[4] = {0x00, 0x00, 0x00, 0x01};
            
            if (have_sps_pps_) {
                // Prepend SPS
                packet_data.insert(packet_data.end(), start_code, start_code + 4);
                packet_data.insert(packet_data.end(), sps_nal_.begin(), sps_nal_.end());
                // Prepend PPS
                packet_data.insert(packet_data.end(), start_code, start_code + 4);
                packet_data.insert(packet_data.end(), pps_nal_.begin(), pps_nal_.end());
            }
            
            // Append the original frame data
            packet_data.insert(packet_data.end(), data, data + size);

            // Copy the modified data into the message
            msg->data.assign(packet_data.begin(), packet_data.end());

            compressed_pub_->publish(std::move(msg));
        }
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        for (const auto& gyro : data) {
            auto msg = std::make_unique<sensor_msgs::msg::Imu>();
            msg->header.stamp = node_->get_clock()->now();
            msg->header.frame_id = "imu_frame";
            msg->angular_velocity.x = gyro.gx;
            msg->angular_velocity.y = gyro.gy;
            msg->angular_velocity.z = gyro.gz;
            
            msg->linear_acceleration.x = gyro.ax * 9.80665;
            msg->linear_acceleration.y = gyro.ay * 9.80665;
            msg->linear_acceleration.z = gyro.az * 9.80665;

            msg->orientation.x = 0.0;
            msg->orientation.y = 0.0;
            msg->orientation.z = 0.0;
            msg->orientation.w = 1.0; // Neutral orientation
            msg->orientation_covariance[0] = -1.0; // No orientation data available

            for (int i = 0; i < 9; i++)
            {
                msg->angular_velocity_covariance[i] = 0;
                msg->linear_acceleration_covariance[i] = 0;
            }
            imu_pub_->publish(std::move(msg));
        }
    }

    void OnExposureData(const ins_camera::ExposureData& data) override {}
};

class CameraWrapper {
private:
    std::shared_ptr<ins_camera::Camera> cam;
    std::shared_ptr<rclcpp::Node> node_;

public:
    CameraWrapper(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {}

    ~CameraWrapper() {
        if (cam) {
            cam->Close();
        }
    }

    int run_camera() {
        ins_camera::DeviceDiscovery discovery;
        auto list = discovery.GetAvailableDevices();
        if (list.empty()) {
            RCLCPP_ERROR(node_->get_logger(), "No available camera devices found.");
            return -1;
        }

        cam = std::make_shared<ins_camera::Camera>(list[0].info);
        if (!cam->Open()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open camera.");
            return -1;
        }
        RCLCPP_INFO(node_->get_logger(), "Camera opened successfully.");
        discovery.FreeDeviceDescriptors(list);

        std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<TestStreamDelegate>(node_);
        cam->SetStreamDelegate(delegate);

        auto start = time(NULL);
        cam->SyncLocalTimeToCamera(start);        
        ins_camera::LiveStreamParam param;
        param.video_resolution = ins_camera::VideoResolution::RES_1920_960P30; //Change this line to edit the resolution
        //Possible resolutions (results may vary per model) are:
        //RES_3840_1920P30
        //RES_2560_1280P30
        //RES_1152_1152P30 (this will give 2304 x 1152 at 30 FPS)
        //RES_1920_960P30  
        param.lrv_video_resulution = ins_camera::VideoResolution::RES_1440_720P30;
        param.video_bitrate = 1024 * 1024 / 2;
        param.enable_audio = false;
        param.using_lrv = false;

        if (!cam->StartLiveStreaming(param)) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to start live streaming.");
            return -1;
        }
        
        RCLCPP_INFO(node_->get_logger(), "Live streaming started.");
        return 0;
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("insta_publisher");
    
    CameraWrapper camera(node);
    if (camera.run_camera() != 0) {
        rclcpp::shutdown();
        return -1;
    }
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}