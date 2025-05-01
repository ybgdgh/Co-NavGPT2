#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
import tf2_ros

import open3d as o3d
import numpy as np

def ros_to_open3d_pointcloud(ros_pc2_msg: PointCloud2) -> o3d.geometry.PointCloud:
    """
    将ROS的PointCloud2转换为Open3D的PointCloud。
    注意：此处仅示例读取(x, y, z)，若需要RGB/强度等信息需自行扩展。
    """
    # 解析字段名
    field_names = [field.name for field in ros_pc2_msg.fields]
    assert 'x' in field_names and 'y' in field_names and 'z' in field_names, \
        "PointCloud2 does not contain [x, y, z] fields."

    # 读取点云数据
    cloud_data = list(
        point_cloud2.read_points(
            ros_pc2_msg, 
            field_names=("x", "y", "z"), 
            skip_nans=True
        )
    )
    if len(cloud_data) == 0:
        return o3d.geometry.PointCloud()
    
    points = np.array(cloud_data, dtype=np.float64)

    # 创建 Open3D 点云对象
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(points)

    return o3d_cloud

def preprocess_pointcloud(
    o3d_cloud: o3d.geometry.PointCloud,
    voxel_size: float = 0.05,
    roi_min: np.ndarray = np.array([-1.0, -1.0, -1.0]),
    roi_max: np.ndarray = np.array([1.0, 1.0, 1.0]),
    remove_outlier: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    """
    对点云进行常见的预处理:
    1. 下采样 (voxel_down_sample)
    2. ROI 裁剪 (AxisAlignedBoundingBox)
    3. 离群点去除 (统计滤波)
    
    :param o3d_cloud: 原始点云
    :param voxel_size: 体素下采样大小
    :param roi_min: ROI 立方体的最小角 [x_min, y_min, z_min]
    :param roi_max: ROI 立方体的最大角 [x_max, y_max, z_max]
    :param remove_outlier: 是否去除离群点
    :param nb_neighbors: 统计滤波中用于计算平均距离的邻域大小
    :param std_ratio: 离群点判断阈值(与均值距离的标准差倍数)
    :return: 处理后的点云
    """
    
    # 1. 下采样
    down_cloud = o3d_cloud.voxel_down_sample(voxel_size=voxel_size)

    # 2. ROI 裁剪
    #   创建一个 AxisAlignedBoundingBox 并进行 crop
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=roi_min, max_bound=roi_max)
    cropped_cloud = down_cloud.crop(aabb)

    # 3. 离群点去除（统计滤波）
    if remove_outlier and len(cropped_cloud.points) > 0:
        cl, ind = cropped_cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        # cl 为滤波后点云, ind 为保留点索引
        processed_cloud = cl
    else:
        processed_cloud = cropped_cloud

    return processed_cloud


class PointCloudCalibrator(Node):
    def __init__(self):
        super().__init__('pointcloud_calibrator')

        # 缓存两个原始点云
        self.pc1_raw = None
        self.pc2_raw = None

        # 若已知两个传感器的大概位姿, 可在此赋值初始变换(4x4)
        # 例如, 假设已知 transform 大概在 X 方向偏 1.0米, 其余不变:
        self.icp_initial_transform = np.array([
            [1., 0., 0., 0.0],
            [0., 1., 0., 1.5],
            [0., 0., 1., 0.0],
            [0., 0., 0., 1.0]
        ])
        self.transformation = None 

        # 订阅话题1
        self.subscription_1 = self.create_subscription(
            PointCloud2, 
            '/robot1/Laser_map', 
            self.pc1_callback, 
            10
        )
        # 订阅话题2
        self.subscription_2 = self.create_subscription(
            PointCloud2, 
            '/robot2/Laser_map', 
            self.pc2_callback, 
            10
        )
        # TF static publish
        self.br = tf2_ros.StaticTransformBroadcaster(self)
        # 使用定时器定期检查是否获得两帧点云, 然后执行配准
        self.timer = self.create_timer(5.0, self.timer_callback)

    def pc1_callback(self, msg):
        self.pc1_raw = ros_to_open3d_pointcloud(msg)
        self.get_logger().info("Received pointcloud_1.")

    def pc2_callback(self, msg):
        self.pc2_raw = ros_to_open3d_pointcloud(msg)
        self.get_logger().info("Received pointcloud_2.")

    def timer_callback(self):
        # 如果都接收到点云, 执行一次配准
        if self.pc1_raw is not None and self.pc2_raw is not None:
            self.get_logger().info("Start ICP registration with preprocessing...")
            
            # ===== 数据预处理 =====
            # 根据实际需求, 设置好下采样 voxel_size, ROI 范围, 离群点滤波参数等
            pc1 = self.pc1_raw.voxel_down_sample(voxel_size=0.05)

            pc2 = self.pc2_raw.voxel_down_sample(voxel_size=0.05)

            # ===== 配准 =====
            # transformation = self.perform_icp_registration(pc1, pc2, self.icp_initial_transform)
            self.transformation = adaptive_threshold_icp(
                pc1, 
                pc2, 
                self.icp_initial_transform,
                threshold_min = 0.01, 
                threshold_max = 5.0, 
                threshold_init = 4.0,
            )
            # 分解出平移和四元数
            translation = self.transformation[0:3, 3]
            rot_mat = self.transformation[0:3, 0:3]
            r = R.from_matrix(rot_mat)
            qx, qy, qz, qw = r.as_quat()
            x, y, z = translation
        
            # 构造 TransformStamped
            t = TransformStamped()
            # 时间戳可以设置为当前时刻
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "camera_init_1"          # 父级坐标系
            t.child_frame_id = "camera_init_2"    # 子坐标系
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw

            # 通过StaticTransformBroadcaster发布
            self.br.sendTransform(t)
            print("publish tf static")
            # 输出结果
            self.get_logger().info(f"ICP Transformation:\n{self.transformation}")
            print("This transformation can be replicated using:")
            print("ros2 run tf2_ros static_transform_publisher " \
                  + str(x) + " " + str(y) +" " + str(0) +" " \
                  + str(qx) + " " + str(qy) + " " + str(qz) + " " + str(qw) + " " + "camera_init_2  camera_init_1" )

            # # 停止定时器(仅示例执行一次)
            # self.timer.cancel()
            # self.destroy_node() 

def adaptive_threshold_icp(
    source, 
    target, 
    init_transform = np.eye(4), 
    threshold_min = 0.01, 
    threshold_max = 1.0, 
    threshold_init = 0.1,
    max_outer_iters = 100,
    max_icp_iters = 50,
    fitness_min_correspondences_ratio = 0.01,
    improvement_ratio = 1e-3
):
    """
    自适应阈值ICP外层循环
    
    参数:
    source, target: Open3D点云 (o3d.geometry.PointCloud)
    init_transform: 初始变换(4x4)
    threshold_min, threshold_max: 阈值上下限
    threshold_init: 初始阈值
    max_outer_iters: 外层迭代最大次数
    max_icp_iters: 内部ICP单次执行最大迭代次数
    fitness_min_correspondences_ratio: 若对对应点比例过低则视为匹配不足
    improvement_ratio: 若误差改善低于该比例则视为收敛
    
    返回:
    best_transform: 最终配准后的4x4变换矩阵
    best_fitness: 最终配准对应点数占比
    best_rmse: 最终配准的均方误差
    """

    # 如果没有法线, 对点到平面ICP需要先估计
    source.estimate_normals()
    target.estimate_normals()

    current_transform = init_transform.copy()
    threshold = threshold_init

    # 初始记下配准质量
    prev_fitness = 0.0
    prev_rmse = 9999.0

    best_transform = current_transform
    best_fitness = 0.0
    best_rmse = prev_rmse

    n_src_pts = len(source.points)
    n_tgt_pts = len(target.points)

    for outer_iter in range(max_outer_iters):
        # 调用Open3D ICP(可用PointToPlane 或 PointToPoint)
        result_icp = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            current_transform,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(), 
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_icp_iters
            )
        )

        current_transform = result_icp.transformation
        fitness = result_icp.fitness   # 对应点数 / min(n_src_pts,n_tgt_pts)
        # print(fitness)
        rmse = result_icp.inlier_rmse
        print(prev_rmse)

        # 计算相对于上一次的改善
        fitness_diff = fitness - prev_fitness
        rmse_diff = prev_rmse - rmse  # rmse 下降越多越好

        # ----- 自适应阈值 调整策略示例 -----
        # 1) 若对应点数过低(几乎无匹配), 说明阈值太小 => 增大阈值
        if fitness < fitness_min_correspondences_ratio:
            threshold *= 1.5

        # 2) 否则若对应点数比以前高了很多, 但 rmse 不够好, 说明匹配到很多错误点 => 减小阈值
        elif rmse > prev_rmse and fitness > prev_fitness:
            threshold *= 0.8

        # 3) 如果rmse有明显下降, 则保持或略微收紧阈值
        else:
            # 收紧一些阈值做更精细配准
            threshold *= 0.9

        # 4) 保证 threshold 在 [threshold_min, threshold_max] 区间
        threshold = np.clip(threshold, threshold_min, threshold_max)

        # ----- 收敛判断 -----
        # 如果 rmse 减少幅度很小, 或 fitness 增长很小, 则认为收敛
        # 这里使用一个简单的比例判断
        if abs(rmse_diff) < improvement_ratio and abs(fitness_diff) < improvement_ratio:
            # 认为已经没有显著改善
            # 可在此直接 break
            print("convergae")
            break

        # 保存本次迭代的结果以便下一轮比较
        prev_fitness = fitness
        prev_rmse = rmse

        # 若本次结果比之前都好, 记录下来
        if fitness > best_fitness or (fitness == best_fitness and rmse < best_rmse):
            best_transform = current_transform.copy()
            best_fitness = fitness
            best_rmse = rmse

    return best_transform

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
