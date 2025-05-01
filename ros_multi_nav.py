#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile, ReliabilityPolicy
qos_profile_reliable = QoSProfile(depth=10)
qos_profile_reliable.reliability = ReliabilityPolicy.RELIABLE

# TF
import tf2_ros
from tf_transformations import quaternion_matrix, translation_matrix, concatenate_matrices

# 注意：ROS2 的 message_filters 需要安装对应版本：ros-foxy-message-filters
import message_filters
from sensor_msgs.msg import PointCloud2



import numpy as np
from argparse import Namespace

# 其他依赖
import threading
import open3d as o3d
import open3d.visualization.gui as gui
from multiprocessing import Process, Queue

# 用户自定义的文件
from arguments import get_args
from agents.ros2_agents import ROS_Agent
from utils.vis_gui import ReconstructionWindow
from utils.explored_map_utils import Global_Map_Proc, detect_frontier
import utils.visualization as vu
from utils import chat_utils
import system_prompt

import zmq


def get_pose(trans, rot, robot_id):
    """
    将 TF 中的 Translation / Rotation 转为 4x4 矩阵
    （可根据你实际需要调整坐标轴）
    """
    # for Jackal
    trans_mat = translation_matrix([trans.x, trans.y, trans.z])
    rot_mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    transform_mat = np.dot(trans_mat, rot_mat)
    
    if robot_id == 0:
        trans_mat = translation_matrix([0.20882221049280814,
                                        -0.00118275956714797,
                                        -0.07903620394463086])
        rot_mat = quaternion_matrix([-0.49352434446717836,
                                     0.4995478883158239,
                                     -0.5045718106204675,
                                     0.5022876831305211])
    elif robot_id == 1:
        trans_mat = translation_matrix([0.20626980772421397,
                                        -0.00528269584842609,
                                        -0.015826777380052434])
        rot_mat = quaternion_matrix([0.5130067729487394,
                                     -0.5014588509249872,
                                     0.4940995610242004,
                                     -0.49115037975492243])
        
    T_lidar_camera = np.dot(trans_mat, rot_mat)
    
    return transform_mat @ T_lidar_camera

def remove_robot_points_cell(point_sum, robot_position, radius = 0.5):
    """
    remove the influence of each other's robot pointcloud
    """
    points = np.asarray(point_sum.points)
    colors = np.asarray(point_sum.colors)

    mask = (((points[:, 0] >= robot_position[0] + radius) | \
            (points[:, 0] <= robot_position[0] - radius) | \
            (points[:, 2] >= robot_position[2] + radius) | \
            (points[:, 2] <= robot_position[2] - radius) )) | (points[:, 1] <= robot_position[1] - 0.44)

    points_filtered = points[mask]
    colors_filtered = colors[mask]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered)
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
    
    

    return pcd

class FspNode(Node):
    def __init__(self, args, send_queue, receive_queue):
        super().__init__('fsp_node')

        self.args = args
        self.point_sum = o3d.geometry.PointCloud()
        self.map_process = Global_Map_Proc(args)
        self.goal_points = []
     
        
        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=1200.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 记录深度图时间戳（或多种消息时间戳），这里用列表维护
        self.global_timestamp = [None]*self.args.num_agents

        # 分别为每个机器人创建订阅器 + 时间同步器
        self.sync_list = []
        for i in range(self.args.num_agents):
            # 根据你的多机器人话题命名来
            rgb_topic = f'/robot{i+1}/camera/color/image_raw'
            depth_topic = f'/robot{i+1}/camera/aligned_depth_to_color/image_raw'
            camera_info_topic = f'/robot{i+1}/camera/color/camera_info'

            # 创建 message_filters.Subscriber
            rgb_sub = message_filters.Subscriber(self, Image, rgb_topic, qos_profile=qos_profile_reliable)
            depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile_reliable)

            # 同步器
            sync = message_filters.ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub],
                queue_size=10,
                slop=0.1
            )
            # 注册回调，这里利用 lambda 把 i 绑定进去
            sync.registerCallback(lambda rgb_msg, depth_msg, idx=i: self.rgbd_callback(idx, rgb_msg, depth_msg))
            self.sync_list.append(sync)

            # 普通订阅：CameraInfo
            self.create_subscription(
                CameraInfo,
                camera_info_topic,
                lambda msg, idx=i: self.camera_info_callback(idx, msg),
                10
            )

        # 动作发布器，每个机器人一个
        self.socket_list = []
        for i in range(self.args.num_agents):
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            if i == 0:
                socket.connect("tcp://192.168.100.1:5557")
            else:
                socket.connect("tcp://192.168.100.4:5557")
            self.socket_list.append(socket)
            
            print("socket_" + str(i) + " connected(?) to port 5557")
        
        
        # 创建 agent
        self.agent = []
        self.obs = []
        # "chair", "bed", "potted plant", "toilet", "tv_screen", "couch", "person", "sink"
        goal_name = "person"
        for i in range(self.args.num_agents):
            self.agent.append(ROS_Agent(args, i, goal_name, send_queue, receive_queue))
            # 存放观测数据
            self.obs.append({})
            
            
            
        # 定时器：相当于原先 while not rospy.is_shutdown() + rospy.Rate(2)
        # 每 0.5 秒执行一次
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info("MultiFspNode for multiple robots initialized.")

    def rgbd_callback(self, i, rgb_msg, depth_msg):
        """
        同步后的回调：将第i个机器人的RGB/Depth存入 self.obs[i]
        """
        h, w = rgb_msg.height, rgb_msg.width
        rs_rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(h, w, 3)
        self.obs[i]['rgb'] = rs_rgb

        depth_dtype = np.uint16  # 16UC1
        rs_depth = np.frombuffer(depth_msg.data, dtype=depth_dtype).reshape(h, w)
        self.obs[i]['depth'] = rs_depth

        self.global_timestamp[i] = depth_msg.header.stamp
        # print("received rgbd images")

    def camera_info_callback(self, i, msg):
        """
        订阅相机内参
        """
        # 这里 msg.K 一般是 [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        # 你先取出需要的 fx, fy, cx, cy
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        camera_matrix = {'cx': cx, 'cy': cy, 'fx': fx, 'fy': fy}
        self.obs[i]['cam_K'] = Namespace(**camera_matrix)

    def timer_callback(self):
        """
        定时执行，相当于原先 while+rate 循环里的主要逻辑
        """
        found_goal = False
        vis_pose_pred = []
        grid_pose = []
        visited_vis = []
        open3d_pose = []
        updated_point_sum = o3d.geometry.PointCloud()

        for i in range(self.args.num_agents):
            # 确保拿到了图像数据
            if 'rgb' not in self.obs[i] or 'depth' not in self.obs[i] or "cam_K" not in self.obs[i]:
                return
            if self.global_timestamp[i] is None:
                return
            # print("received rgbd images")
            
            if self.agent[i].l_step == 0:
                # init robots
                try:
                    # 在 ROS2 中，Time 对象可以直接传给 lookup_transform
                    # 这里等效于 tf_buffer.lookup_transform("map", "camera_link", t_stamp)
                    transform = self.tf_buffer.lookup_transform(
                        f'camera_init_{i+1}',
                        f'body_{i+1}',
                        rclpy.time.Time(),  # 或者self.global_timestamp, rclpy.time.Time() 表示最新
                        timeout=rclpy.duration.Duration(seconds=0.5)  # 超时时间可调
                    )
                    # print("received init tf_" + str(1))
                except tf2_ros.TransformException as e:
                    self.get_logger().warn(f"TF init lookup failed for robot {i}: {e}")
                    return

                # 转换成 4x4 矩阵
                trans = transform.transform.translation
                rot = transform.transform.rotation
                init_position = get_pose(trans, rot, 0)
                
                self.agent[i].init_sim_position = init_position[:3, 3]
                self.agent[i].init_sim_rotation = init_position[:3, :3]
        
            # 获取 TF
            try:
                # 在 ROS2 中，Time 对象可以直接传给 lookup_transform
                # 这里等效于 tf_buffer.lookup_transform("map", "camera_link", t_stamp)
                transform = self.tf_buffer.lookup_transform(
                    f'camera_init_1',
                    f'body_{i+1}',
                    self.global_timestamp[i],  # 或者self.global_timestamp, rclpy.time.Time() 表示最新
                    timeout=rclpy.duration.Duration(seconds=0.0)  # 超时时间可调
                )
                # print("received tf_" + str(i+1))
            except tf2_ros.TransformException as e:
                self.get_logger().warn(f"TF lookup failed for robot {i}: {e}")
                return

            # 转换成 4x4 矩阵
            trans = transform.transform.translation
            rot = transform.transform.rotation
            self.obs[i]['pose'] = get_pose(trans, rot, i)
            # 让 agent 做 mapping / act
            self.agent[i].mapping(self.obs[i])
            
            
            visited_vis.append(self.agent[i].visited_vis)
            vis_pose_pred.append([self.agent[i].current_grid_pose[1]*480.0/self.agent[i].map_size, int((self.agent[i].map_size-self.agent[i].current_grid_pose[0])*480.0/self.agent[i].map_size), np.deg2rad(self.agent[i].relative_angle)])
            grid_pose.append(self.agent[i].current_grid_pose)
            
            open3d_pose.append(self.agent[i].camera_position)
            
            if self.agent[i].found_goal:
                found_goal = True 
            
            
            updated_point_sum += self.agent[i].point_sum
            
        # clean the influence of each others
        for i in range(self.args.num_agents):
            # updated_point_sum = remove_robot_points_cell(updated_point_sum, self.agent[i].camera_position)
            self.agent[i].open3d_pose = open3d_pose
            
        # self.point_sum += updated_point_sum
            
        # Merge the map
        obstacle_map, explored_map, top_view_map = self.map_process.Map_Extraction(updated_point_sum, 0.0)
        
        target_score, target_edge_map, target_point_list = self.map_process.Frontier_Det(threshold_point=20)
        print(len(target_point_list))

        # For GPT
        if (self.agent[0].l_step % self.args.num_local_steps == self.args.num_local_steps - 1 or self.agent[0].l_step == 0) and not found_goal:
            self.goal_points.clear()
            if self.args.nav_mode == "gpt":
                target_score, target_edge_map, target_point_list = self.map_process.Frontier_Det(threshold_point=20)
                        
                if len(target_point_list) > 0 and self.agent[0].l_step >= 30:
                    candidate_map_list = chat_utils.get_all_candidate_maps(target_edge_map, top_view_map, vis_pose_pred)
                    message = chat_utils.message_prepare(system_prompt.system_prompt, candidate_map_list, self.agent[i].goal_name)
            
                    goal_frontiers = chat_utils.chat_with_gpt4v(message)
                    for i in range(self.args.num_agents):
                        self.goal_points.append(target_point_list[int(goal_frontiers["robot_"+ str(i)].split('_')[1])])
                        
                else:
                    for i in range(self.args.num_agents):
                        action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
                        self.goal_points.append([int(action[0]), int(action[1])])
                        
            elif self.args.nav_mode == "nearest":
                for i in range(self.args.num_agents):
                    target_score, target_edge_map, target_point_list = detect_frontier(explored_map, obstacle_map, self.agent[i].current_grid_pose, threshold_point=20)
                    if len(target_point_list) > 0:
                        self.goal_points.append(target_point_list[0])
                    else:
                        action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
                        self.goal_points.append([int(action[0]), int(action[1])])

        goal_map = []
        for i in range(self.args.num_agents):
            self.agent[i].obstacle_map = obstacle_map
            self.agent[i].explored_map = explored_map
            action = self.agent[i].act(self.goal_points[i], grid_pose)
            self.agent[i].found_goal = found_goal
            goal_map.append(self.agent[i].goal_map)
            
            # # 发布动作
            if action == 1:
                self.socket_list[i].send(b"speedctl speed|0.3|0.0|0.0")
            elif action == 2:
                self.socket_list[i].send(b"speedctl speed|0.0|0.0|0.5")
            elif action == 3:
                self.socket_list[i].send(b"speedctl speed|0.0|0.0|-0.5")
                
            # print(self.agent[i].l_step)
            if self.agent[i].l_step < 35:
                self.socket_list[i].send(b"speedctl speed|0.0|0.0|0.5")
                
            
        if self.args.visualize or self.args.print_images:
            vis_image = vu.Visualize(self.args, 
                                    self.agent[0].l_step, 
                                    vis_pose_pred, 
                                    obstacle_map, 
                                    explored_map,
                                    self.agent[0].goal_name, 
                                    visited_vis, 
                                    target_edge_map, 
                                    goal_map, 
                                    top_view_map)
            
        # 仅做示例日志
        self.get_logger().info(f"Action published for robot {i}: {action}")

        
        
def visualization_thread(args, send_queue, receive_queue):
    """
    Open3D GUI 线程
    """
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))

    app_win = ReconstructionWindow(args, mono, send_queue, receive_queue)
    app.run()  # 阻塞式，直到窗口退出


def main():
    # 1. 初始化 ROS2
    rclpy.init()

    # 2. 获取用户自定义参数
    args = get_args()
    args.map_height_cm = 30

    # 3. 建立进程 / 线程间通信队列
    send_queue = Queue()
    receive_queue = Queue()

    # 4. 启动可视化线程
    visualization = threading.Thread(target=visualization_thread, args=(args, send_queue, receive_queue))
    visualization.start()

    node = FspNode(args, send_queue, receive_queue)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    # 6. 等待可视化线程结束
    visualization.join()
    print("you successfully navigated to destination point")


if __name__ == "__main__":
    main()
