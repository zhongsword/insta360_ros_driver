#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os
from ament_index_python.packages import get_package_share_directory

class FisheyeCalibrator(Node):
    def __init__(self):
        super().__init__('fisheye_calibrator')

        # Parameters
        self.declare_parameter('chessboard_size', [9, 6])  # Number of inner corners
        self.declare_parameter('square_size', 0.025)  # Size of chessboard square in meters
        self.declare_parameter('calibration_images', 20)  # Number of images to collect
        self.declare_parameter('output_file', 'fisheye_calibration.yaml')

        self.chessboard_size = tuple(self.get_parameter('chessboard_size').value)
        self.square_size = self.get_parameter('square_size').value
        self.calibration_images = self.get_parameter('calibration_images').value
        output_file_param = self.get_parameter('output_file').value
        if '$(find-pkg-share' in output_file_param:
            package_share = get_package_share_directory('insta360_ros_driver')
            self.output_file = os.path.join(package_share, 'config', 'fisheye_calibration.yaml')
        else:
            self.output_file = output_file_param

        # Calibration data
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane
        self.image_size = None

        # Prepare object points (chessboard corners in 3D)
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        self.objp = objp

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/dual_fisheye/image/forward',
            self.image_callback,
            10
        )

        self.get_logger().info(f'Fisheye Calibrator initialized')
        self.get_logger().info(f'Chessboard size: {self.chessboard_size}')
        self.get_logger().info(f'Square size: {self.square_size} m')
        self.get_logger().info(f'Target images: {self.calibration_images}')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        if self.image_size is None:
            self.image_size = (cv_image.shape[1], cv_image.shape[0])

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(cv_image, self.chessboard_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                 cv2.CALIB_CB_FAST_CHECK)

        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(cv_image, corners, (11, 11), (-1, -1), criteria)

            self.object_points.append(self.objp)
            self.image_points.append(corners)

            self.get_logger().info(f'Chessboard detected! Collected {len(self.image_points)}/{self.calibration_images} images')

            # Draw and display corners
            cv2.drawChessboardCorners(cv_image, self.chessboard_size, corners, ret)
            cv2.imshow('Chessboard Detection', cv_image)
            cv2.waitKey(500)

            # Check if we have enough images
            if len(self.image_points) >= self.calibration_images:
                self.perform_calibration()
        else:
            self.get_logger().debug('Chessboard not found in image')

    def perform_calibration(self):
        self.get_logger().info('Performing fisheye calibration...')

        # Convert to numpy arrays
        objpoints = [self.objp.reshape(-1, 1, 3) for _ in range(len(self.image_points))]
        imgpoints = [img.reshape(-1, 1, 2) for img in self.image_points]

        # Fisheye calibration
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = []
        tvecs = []

        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints, imgpoints, self.image_size, K, D,
                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                      cv2.fisheye.CALIB_CHECK_COND +
                      cv2.fisheye.CALIB_FIX_SKEW,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )

            self.get_logger().info(f'Calibration completed with RMS error: {rms}')

            # Save calibration data
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            calibration_data = {
                'image_width': self.image_size[0],
                'image_height': self.image_size[1],
                'camera_matrix': K.tolist(),
                'distortion_coefficients': D.flatten().tolist(),
                'rms_error': float(rms)
            }

            with open(self.output_file, 'w') as f:
                yaml.dump(calibration_data, f, default_flow_style=False)

            self.get_logger().info(f'Calibration saved to {self.output_file}')

            # Print results
            print("\nCalibration Results:")
            print(f"Camera Matrix K:\n{K}")
            print(f"Distortion Coefficients D:\n{D}")
            print(f"RMS Error: {rms}")

        except cv2.error as e:
            self.get_logger().error(f'Calibration failed: {e}')

        # Shutdown
        cv2.destroyAllWindows()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    calibrator = FisheyeCalibrator()
    rclpy.spin(calibrator)

if __name__ == '__main__':
    main()