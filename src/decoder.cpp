#include <iostream>
#include <thread>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
}

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_CUDA) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

class H264DecoderNode : public rclcpp::Node {
private:
    AVCodec* codec_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVCodecParserContext* parser_ctx_ = nullptr;
    AVPacket* pkt_ = nullptr;
    AVFrame* hw_frame_ = nullptr;
    AVFrame* sw_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    cv::Mat gray_frame_; 
    AVBufferRef *hw_device_ctx_ = nullptr;
    enum AVHWDeviceType hw_type_ = AV_HWDEVICE_TYPE_NONE;

    // Cache the latest SPS / PPS to prepend when the stream lacks parameter sets (avoids mosaic).
    std::vector<uint8_t> sps_nal_;
    std::vector<uint8_t> pps_nal_;
    std::vector<uint8_t> parse_buffer_;
    bool have_sps_pps_ = false;
    bool decoder_synced_ = false;
    bool need_decoder_flush_ = true;
    bool warned_no_sps_pps_ = false;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

    std::thread publisher_thread_;
    std::queue<cv::Mat> frame_publish_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_publisher_thread_{false};
    size_t max_queue_size_ = 10;
    
    int skip_frame_ = 0;
    int frame_counter_ = 0;
    bool i_frame_only_ = true;
    bool initialized_ = false;

    void InitDecoder(const std::string& format) {
        // Try V4L2 M2M hardware acceleration for Raspberry Pi 4B
        hw_type_ = AV_HWDEVICE_TYPE_NONE; // V4L2 M2M uses device directly
        const char* decoder_name = "h264_v4l2m2m";

        codec_ = avcodec_find_decoder_by_name(decoder_name);
        if (!codec_) {
            RCLCPP_WARN(this->get_logger(), "V4L2 M2M decoder not available, falling back to software");
            hw_type_ = AV_HWDEVICE_TYPE_NONE;
            codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
            if (!codec_) {
                RCLCPP_ERROR(this->get_logger(), "No H.264 decoder available");
                return;
            }
        } else {
            RCLCPP_INFO(this->get_logger(), "Using V4L2 M2M hardware H.264 decoder");
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE && strcmp(decoder_name, "h264_omx") != 0) {
            int err = av_hwdevice_ctx_create(&hw_device_ctx_, hw_type_, nullptr, nullptr, 0);
            if (err < 0) {
                RCLCPP_WARN(this->get_logger(), "Failed to create %s hardware device context, falling back to software", decoder_name);
                hw_type_ = AV_HWDEVICE_TYPE_NONE;
                codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
                if (!codec_) {
                    RCLCPP_ERROR(this->get_logger(), "No H.264 decoder available");
                    return;
                }
            }
        }

        parser_ctx_ = av_parser_init(codec_->id);
        if (!parser_ctx_) {
            CleanupDecoder();
            return;
        }

        codec_ctx_ = avcodec_alloc_context3(codec_);
        if (!codec_ctx_) {
            CleanupDecoder();
            return;
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE && hw_device_ctx_) {
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            if (hw_type_ != AV_HWDEVICE_TYPE_NONE) {
                codec_ctx_->get_format = get_hw_format;
            }
        }

        if (avcodec_open2(codec_ctx_, codec_, nullptr) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open codec");
            CleanupDecoder();
            return;
        }

        pkt_ = av_packet_alloc();
        if (!pkt_) {
            CleanupDecoder();
            return;
        }

        hw_frame_ = av_frame_alloc();
        if (!hw_frame_) {
            CleanupDecoder();
            return;
        }

        if (hw_type_ != AV_HWDEVICE_TYPE_NONE) {
            sw_frame_ = av_frame_alloc();
            if (!sw_frame_) {
                CleanupDecoder();
                return;
            }
        }
    }

    void PublisherThreadLoop() {
        while (!stop_publisher_thread_) {
            cv::Mat frame_to_publish;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !frame_publish_queue_.empty() || stop_publisher_thread_;
                });

                if (stop_publisher_thread_ && frame_publish_queue_.empty()) {
                    break;
                }
                if (frame_publish_queue_.empty()) {
                    continue;
                }
                frame_to_publish = frame_publish_queue_.front();
                frame_publish_queue_.pop();
            }

            if (!frame_to_publish.empty() && publisher_) {
                auto img_msg = std::make_unique<sensor_msgs::msg::Image>();
                std_msgs::msg::Header header;
                header.stamp = this->get_clock()->now();
                header.frame_id = "camera_frame";
                cv_bridge::CvImage cv_image(header, sensor_msgs::image_encodings::MONO8, frame_to_publish);
                cv_image.toImageMsg(*img_msg);
                publisher_->publish(std::move(img_msg));
            }
        }
    }

    void DecodeAndDisplayPacket(AVPacket* packet) {
        int ret = avcodec_send_packet(codec_ctx_, packet);
        if (ret < 0) {
            decoder_synced_ = false;
            need_decoder_flush_ = true;
            return;
        }

        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                return;
            } else if (ret < 0) {
                decoder_synced_ = false;
                need_decoder_flush_ = true;
                return;
            }

            AVFrame* frame_to_display = hw_frame_;

            if (hw_frame_->format == AV_PIX_FMT_CUDA || hw_frame_->format == AV_PIX_FMT_VAAPI) {
                if (av_hwframe_transfer_data(sw_frame_, hw_frame_, 0) < 0) {
                    av_frame_unref(hw_frame_);
                    decoder_synced_ = false;
                    need_decoder_flush_ = true;
                    continue;
                }
                frame_to_display = sw_frame_;
            }

            if (!sws_ctx_ && frame_to_display->width > 0 && frame_to_display->height > 0) {
                sws_ctx_ = sws_getContext(
                    frame_to_display->width, frame_to_display->height, (AVPixelFormat)frame_to_display->format,
                    frame_to_display->width, frame_to_display->height, AV_PIX_FMT_GRAY8,
                    SWS_POINT, nullptr, nullptr, nullptr);
                
                if (!sws_ctx_) {
                    av_frame_unref(hw_frame_);
                    if (frame_to_display == sw_frame_) av_frame_unref(sw_frame_);
                    return; 
                }
                gray_frame_.create(frame_to_display->height, frame_to_display->width, CV_8UC1);
            }

            if (sws_ctx_ && !gray_frame_.empty()) {
                uint8_t* dst_data[4] = { gray_frame_.data, nullptr, nullptr, nullptr };
                int dst_linesize[4] = { static_cast<int>(gray_frame_.step[0]), 0, 0, 0 };

                sws_scale(sws_ctx_,
                            (const uint8_t* const*)frame_to_display->data, frame_to_display->linesize,
                            0, frame_to_display->height,
                            dst_data, dst_linesize);

                // Apply frame skipping after decoding
                bool should_publish = true;
                
                if (skip_frame_ > 0 && !i_frame_only_) {
                    // Skip frame logic (only when not in i_frame_only mode)
                    should_publish = (frame_counter_++ % (skip_frame_ + 1) == 0);
                }
                
                if (should_publish) {
                    cv::Mat frame_copy = gray_frame_.clone();
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        if (frame_publish_queue_.size() < max_queue_size_) {
                            frame_publish_queue_.push(std::move(frame_copy));
                        } else {
                            // Drop oldest frame if queue full
                            frame_publish_queue_.pop();
                            frame_publish_queue_.push(std::move(frame_copy));
                        }
                    }
                    queue_cv_.notify_one();
                }
            }
            
            av_frame_unref(hw_frame_);
            if (frame_to_display == sw_frame_) {
                av_frame_unref(sw_frame_);
            }
        }
    }

    void CleanupDecoder() {
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        if (sw_frame_) {
            av_frame_free(&sw_frame_);
            sw_frame_ = nullptr;
        }
        if (hw_frame_) {
            av_frame_free(&hw_frame_);
            hw_frame_ = nullptr;
        }
        if (pkt_) {
            av_packet_free(&pkt_);
            pkt_ = nullptr;
        }
        if (codec_ctx_) {
            avcodec_close(codec_ctx_); 
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
        if (parser_ctx_) {
            av_parser_close(parser_ctx_);
            parser_ctx_ = nullptr;
        }
        if (hw_device_ctx_) {
            av_buffer_unref(&hw_device_ctx_);
            hw_device_ctx_ = nullptr;
        }
        codec_ = nullptr;
    }

    // --- NAL parsing helpers -------------------------------------------------
    // Find all NAL units in Annex-B buffer and cache SPS/PPS if present.
    void ExtractSpsPps(const uint8_t* data, size_t size) {
        const uint8_t* end = data + size;
        const uint8_t* p = data;
        auto find_start = [](const uint8_t* cur, const uint8_t* end) -> const uint8_t* {
            // search for 0x000001 or 0x00000001
            const uint8_t* p2 = cur;
            while (p2 + 3 < end) {
                if (p2[0] == 0x00 && p2[1] == 0x00 && ((p2[2] == 0x01) || (p2[2] == 0x00 && p2 + 4 < end && p2[3] == 0x01))) {
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
                if (sps_nal_.size() != static_cast<size_t>(nal_end - nal_start) ||
                    !std::equal(nal_start, nal_end, sps_nal_.begin(), sps_nal_.end())) {
                    sps_nal_.assign(nal_start, nal_end);
                    decoder_synced_ = false;
                    need_decoder_flush_ = true;
                }
            } else if (nal_type == 8) { // PPS
                if (pps_nal_.size() != static_cast<size_t>(nal_end - nal_start) ||
                    !std::equal(nal_start, nal_end, pps_nal_.begin(), pps_nal_.end())) {
                    pps_nal_.assign(nal_start, nal_end);
                    decoder_synced_ = false;
                    need_decoder_flush_ = true;
                }
            }
            p = nal_end;
        }
        have_sps_pps_ = !sps_nal_.empty() && !pps_nal_.empty();
        if (have_sps_pps_) {
            warned_no_sps_pps_ = false;
        }
    }

    // If the incoming packet lacks SPS/PPS but we have cached ones, prepend them.
    void BuildPacketWithCachedPps(const uint8_t* in_data, size_t in_size, std::vector<uint8_t>& out_buf) {
        static const uint8_t start3[3] = {0x00, 0x00, 0x01};
        out_buf.clear();
        if (have_sps_pps_) {
            out_buf.insert(out_buf.end(), start3, start3 + 3);
            out_buf.insert(out_buf.end(), sps_nal_.begin(), sps_nal_.end());
            out_buf.insert(out_buf.end(), start3, start3 + 3);
            out_buf.insert(out_buf.end(), pps_nal_.begin(), pps_nal_.end());
        }
        out_buf.insert(out_buf.end(), in_data, in_data + in_size);
    }

    void compressed_image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        if (msg->format != "h264") {
            return;
        }

        if (!initialized_) {
            InitDecoder(msg->format);
            initialized_ = true;
        }

        if (!codec_ctx_ || !parser_ctx_ || !pkt_ || !hw_frame_) {
            return;
        }

        // First, scan and cache SPS/PPS if present in this message
        ExtractSpsPps(msg->data.data(), msg->data.size());

        // If we still don't have SPS/PPS, drop until we see them to avoid decoder corruption
        if (!have_sps_pps_) {
            if (!warned_no_sps_pps_) {
                RCLCPP_WARN(this->get_logger(), "No SPS/PPS seen yet; dropping frames until parameter sets arrive (avoids mosaic)");
                warned_no_sps_pps_ = true;
            }
            decoder_synced_ = false;
            need_decoder_flush_ = true;
            return;
        }

        // Prepend cached SPS/PPS to ensure every packet has parameter sets (helps when joining mid-stream)
        BuildPacketWithCachedPps(msg->data.data(), msg->data.size(), parse_buffer_);

        const uint8_t* cur_data = parse_buffer_.data();
        size_t remaining_size = parse_buffer_.size();

        while (remaining_size > 0) {
            int bytes_parsed = av_parser_parse2(parser_ctx_, codec_ctx_,
                                                &pkt_->data, &pkt_->size,
                                                cur_data, static_cast<int>(remaining_size),
                                                AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (bytes_parsed < 0) {
                RCLCPP_WARN(this->get_logger(), "Parser error, skipping packet");
                break; 
            }
            cur_data += bytes_parsed;
            remaining_size -= bytes_parsed;

            if (pkt_->size > 0) {
                bool is_key = parser_ctx_->key_frame == 1;

                if (is_key) {
                    if (need_decoder_flush_) {
                        avcodec_flush_buffers(codec_ctx_);
                        need_decoder_flush_ = false;
                    }
                    decoder_synced_ = true;
                }

                if (i_frame_only_) {
                    if (is_key) {
                        DecodeAndDisplayPacket(pkt_);
                    }
                } else {
                    if (!decoder_synced_) {
                        continue; // wait for keyframe to sync decoder
                    }
                    DecodeAndDisplayPacket(pkt_);
                }
            }
        }
    }

public:
    H264DecoderNode() : Node("h264_decoder_node") {
        this->declare_parameter("compressed_topic", "/dual_fisheye/image/compressed");
        this->declare_parameter("uncompressed_topic", "/dual_fisheye/image");
        this->declare_parameter("skip_frame", 0);
        this->declare_parameter("i_frame_only", true);  // Default to true for high quality at low frame rate

        std::string subscribe_topic = this->get_parameter("compressed_topic").as_string();
        std::string publish_topic = this->get_parameter("uncompressed_topic").as_string();
        skip_frame_ = this->get_parameter("skip_frame").as_int();
        i_frame_only_ = true;  // Force I-frame only mode

        subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            subscribe_topic, rclcpp::SensorDataQoS(),
            std::bind(&H264DecoderNode::compressed_image_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(publish_topic, rclcpp::SensorDataQoS());

        publisher_thread_ = std::thread(&H264DecoderNode::PublisherThreadLoop, this);
        
        initialized_ = false;

        RCLCPP_INFO(this->get_logger(), "H.264 Decoder Node initialized");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", subscribe_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to: %s", publish_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Skip frame: %d, I-frame only: %s", skip_frame_, i_frame_only_ ? "true" : "false");
    }

    ~H264DecoderNode() {
        stop_publisher_thread_ = true;
        queue_cv_.notify_one();
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
        }
        CleanupDecoder();
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<H264DecoderNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
