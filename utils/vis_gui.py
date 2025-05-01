# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/dense_slam_gui.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import os
import numpy as np
import threading
import time
import cv2
from PIL import Image, ImageDraw, ImageFont



def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable

class ReconstructionWindow:

    def __init__(self, args, font_id, send_queue, receive_queue):
        self.args = args
        self.num_agent = args.num_agents

        self.device = "cuda:{}".format(self.args.gpu_id)

        self.window = gui.Application.instance.create_window(
            'Open3D - Reconstruction', 1280, 800)

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Items in adjustable props
        self.adjustable_prop_grid = gui.VGrid(2, spacing,
                                              gui.Margins(em, 0, em, 0))

        ### Update surface?
        rgb_pc_label = gui.Label('RGB PC?')
        self.rgb_pc_box = gui.Checkbox('')
        self.rgb_pc_box.checked = True
        self.adjustable_prop_grid.add_child(rgb_pc_label)
        self.adjustable_prop_grid.add_child(self.rgb_pc_box)

        ### Show trajectory?
        trajectory_label = gui.Label('Trajectory?')
        self.trajectory_box = gui.Checkbox('')
        self.trajectory_box.checked = True
        self.adjustable_prop_grid.add_child(trajectory_label)
        self.adjustable_prop_grid.add_child(self.trajectory_box)

        ## Application control
        b = gui.ToggleSwitch('Resume/Pause')
        b.set_on_clicked(self._on_switch)

        self.text_edit = gui.TextEdit()
        self.submit_button = gui.Button("Submit")
        self.submit_button.set_on_clicked(self._on_submit)
        self.input_text = None

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()



        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.annotated_image = []
        for i in range(self.num_agent):                
            self.annotated_image.append(gui.ImageWidget())
            tab1.add_child(self.annotated_image[i])
            tab1.add_fixed(vspacing)
        tabs.add_tab('Annotated Image', tab1)
        
        ### Rendered image tab
        tab2 = gui.Vert(0, tab_margins)
        self.via_map = []
        for i in range(self.num_agent):   
            self.via_map.append(gui.ImageWidget())
            tab2.add_child(self.via_map[i])
            tab2.add_fixed(vspacing)
        tabs.add_tab('Semantic Maps', tab2)

        ### Info tab
        tab3 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id
        tab3.add_child(self.output_info)
        tabs.add_tab('Info', tab3)

        self.panel.add_child(gui.Label('Reconstruction settings'))
        self.panel.add_child(self.adjustable_prop_grid)
        self.panel.add_child(self.text_edit)
        self.panel.add_child(self.submit_button)
        self.panel.add_child(b)
        self.panel.add_stretch()
        self.panel.add_child(tabs)


        # Scene widget
        self.widget3d = gui.SceneWidget()

        # FPS panel
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.fps_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.saved_objects = None
        self.saved_full_points = None
        self.saved_full_colors = None

        self.is_done = False

        self.is_started = False
        self.is_running = True
        self.episode_n = 0
        self.candidate_id = 0

        self.idx = 0
        self.traj = []
        self.full_pcd_poinst = []
        self.full_pcd_colors = []

        # Start running
        self.send_queue = send_queue
        self.receive_queue = receive_queue
        threading.Thread(name='UpdateMain', target=self.update_main).start()

    def _on_submit(self):
        input_text = self.text_edit.text_value
        print("Input text:", input_text)
        self.send_queue.put([input_text, self.is_running])

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
                                        rect.y, fps_panel_width,
                                        fps_panel_height)

    # Toggle callback: application's main controller
    def _on_switch(self, is_on):
        
        self.is_running = not self.is_running
        self.send_queue.put([None, self.is_running])

    # On start: point cloud buffer and model initialization.
    def _on_start(self):
        max_points = self.est_point_count_slider.int_value

        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32)))
        pcd_placeholder.point.colors = o3c.Tensor(
            np.zeros((max_points, 3), dtype=np.float32))
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)

        # self.model = o3d.t.pipelines.slam.Model(
        #     self.voxel_size_slider.double_value, 16,
        #     self.est_block_count_slider.int_value, o3c.Tensor(np.eye(4)),
        #     o3c.Device(self.device))
        self.is_started = True

        set_enabled(self.fixed_prop_grid, False)
        set_enabled(self.adjustable_prop_grid, True)

    def _on_close(self):
        self.is_done = True

        # if self.is_started:
        # print('Saving model to {}...'.format(self.args.path_npz))
        # if not os.path.exists(self.args.path_npz):
        #     os.makedirs(self.args.path_npz)
        # np.savez(self.args.path_npz + 'arrays.npz', objects=self.saved_objects, full_points=self.saved_full_points, full_colors=self.saved_full_colors)
        print('Finished.')

            # mesh_fname = '.'.join(self.args.path_npz.split('.')[:-1]) + '.ply'
            # print('Extracting and saving mesh to {}...'.format(mesh_fname))
            # mesh = extract_trianglemesh(self.model.voxel_grid, config,
            #                             mesh_fname)
            # print('Finished.')

            # log_fname = '.'.join(self.args.path_npz.split('.')[:-1]) + '.log'
            # print('Saving trajectory to {}...'.format(log_fname))
            # save_poses(log_fname, self.poses)
            # print('Finished.')

        return True
    
    def matrix2lineset(self, poses, color=[0, 0, 1]):
        '''
        Create a open3d line set from a batch of poses

        poses: (N, 4, 4)
        color: (3,)
        '''
        N = poses.shape[0]
        lineset = o3d.geometry.LineSet()
        if np.all(np.ptp(poses[:, :3, 3], axis=0) == 0):
            return lineset
        
        lineset.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
        lineset.lines = o3d.utility.Vector2iVector(
            np.array([[i, i + 1] for i in range(N - 1)])
        )
        lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
        return lineset
    
    def poses2lineset(self, poses, color=[0, 0, 1]):
        '''
        Create a open3d line set from a batch of poses

        poses: (N, 4, 4)
        color: (3,)
        '''
        N = poses.shape[0]
        lineset = o3d.geometry.LineSet()
        if np.all(np.ptp(poses, axis=0) == 0):
            return lineset
        
        lineset.points = o3d.utility.Vector3dVector(poses)
        lineset.lines = o3d.utility.Vector2iVector(
            np.array([[i, i + 1] for i in range(N - 1)])
        )
        lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
        return lineset


    def init_render(self):
       
        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(90, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [-3, 4, 0], [0, 1, 0])

        # self.widget3d.setup_camera(90, bbox, [camera_matrix[3,0], camera_matrix[3,1], camera_matrix[3,2]])[0, 0, 0]
        # self.widget3d.look_at(camera_matrix[:3,0], camera_matrix[:3,1], camera_matrix[:3,2])
        # points = np.random.rand(100, 3)
        # colors = np.zeros((100, 3))
        # colors[:, 0] = 1  # 红色
        # pcd_t = o3d.t.geometry.PointCloud(
        #             o3c.Tensor(points.astype(np.float32)))
        # pcd_t.point.colors = o3c.Tensor(colors)
        # material = rendering.MaterialRecord()
        # material.shader = "defaultUnlit"
        # self.widget3d.scene.add_geometry('points', pcd_t, material)  # Add material argument

        # Add a coordinate frame
        self.widget3d.scene.show_axes(True)

        
    def update_render(self, 
                        agent_id,
                        depth,
                        color,
                        semantic_image,                             
                        vis_image,
                        frustum,
                        point_sum_points,
                        point_sum_colors,
                        traj,
                        plan_path,
                        Open3d_goal_pose):
        

        # self.input_depth_image.update_image(
        #     input_depth.colorize_depth(
        #         1000.0, self.args.min_depth,
        #         self.args.max_depth).to_legacy())
        self.annotated_image[agent_id].update_image(semantic_image.to_legacy())
        self.via_map[agent_id].update_image(vis_image.to_legacy())

        # add the camera
        self.widget3d.scene.remove_geometry("frustum_"+str(agent_id))
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5.0
        self.widget3d.scene.add_geometry("frustum_"+str(agent_id), frustum, mat)

        # add the trajectory
        self.widget3d.scene.remove_geometry("trajectory_"+str(agent_id))
        if self.trajectory_box.checked:
            traj_lineset = self.matrix2lineset(np.stack(traj), color=[0, 1., 0])
            if traj_lineset.has_lines() and traj_lineset.has_points(): 
                mat = rendering.MaterialRecord()
                mat.shader = "unlitLine"
                mat.line_width = 10.0
                self.widget3d.scene.add_geometry("trajectory_"+str(agent_id), traj_lineset, mat)

        self.widget3d.scene.remove_geometry("path_points_"+str(agent_id))
        if len(plan_path) > 0:
            path_lineset = self.poses2lineset(np.stack(plan_path), color=[1., 0, 0])
            if path_lineset.has_lines() and path_lineset.has_points():
                material = rendering.MaterialRecord()
                material.shader = "unlitLine"
                material.line_width = 5.0
                self.widget3d.scene.add_geometry("path_points_"+str(agent_id), path_lineset, material)

        self.widget3d.scene.remove_geometry("full_pcd_"+str(agent_id))
        if self.rgb_pc_box.checked:
            # self.full_pcd_poinst.append(point_sum_points)
            # self.full_pcd_colors.append(point_sum_colors)
            
            # point_sum_points = np.vstack(self.full_pcd_poinst)
            # point_sum_colors = np.vstack(self.full_pcd_colors)
            
            full_pcd = o3d.t.geometry.PointCloud(
            o3c.Tensor(point_sum_points.astype(np.float32)))
            full_pcd.point.colors = o3c.Tensor(point_sum_colors.astype(np.float32))
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.widget3d.scene.add_geometry("full_pcd_"+str(agent_id), full_pcd, material)  # Add material argument

      
        # self.widget3d.scene.remove_geometry("goal_pose_"+str(agent_id))
        # goal_pcd = o3d.geometry.PointCloud()
        # goal_pcd.points = o3d.utility.Vector3dVector(np.array([Open3d_goal_pose]))
        # # Set a large size for the point
        # goal_material = rendering.MaterialRecord()
        # goal_material.shader = "defaultUnlit"
        # goal_material.point_size = 20.0  # Adjust this value to make the point larger
        # # Set the color of the point to distinguish it (e.g., red)
        # goal_colors = np.array([[1.0, 0.0, 0.0]])  # Red color
        # goal_pcd.colors = o3d.utility.Vector3dVector(goal_colors)
        # # Add the goal pose point cloud to the scene
        # self.widget3d.scene.add_geometry("goal_pose_"+str(agent_id), goal_pcd, goal_material)


    # Major loop
    def update_main(self):
        
        height = self.args.frame_height
        width = self.args.frame_width
        
        cx = (width - 1.) / 2.
        cy = (height - 1.) / 2.
        fx = (width / 2.) / np.tan(np.deg2rad(self.args.hfov / 2.))
        fy = (height / 2.) / np.tan(np.deg2rad(self.args.hfov / 2.))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix,
                               o3d.core.Dtype.Float64)

        device = o3d.core.Device(self.device)

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render())

        fps_interval_len = 1
        self.idx = 0
        # pcd = None

        start = time.time()
        while not self.is_done:
            if not self.receive_queue.empty():
                agent_id, image_rgb, image_depth, annotated_image, vis_image, point_sum_points, point_sum_colors, traj, episode_n, plan_path, Open3d_goal_pose, time_step_info = self.receive_queue.get()
                
                # self.widget3d.scene.clear_geometry()

                image_depth = (image_depth * 1000).astype(np.uint16)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) 

                depth = o3d.t.geometry.Image(image_depth)
                color = o3d.t.geometry.Image(image_rgb)
                semantic = o3d.t.geometry.Image(annotated_image)
                vis_image = o3d.t.geometry.Image(vis_image)

                T_frame_to_model = o3c.Tensor(traj[-1])
                frustum = o3d.geometry.LineSet.create_camera_visualization(
                    color.columns, color.rows, intrinsic.numpy(),
                    np.linalg.inv(T_frame_to_model.cpu().numpy()), 0.2)
                frustum.paint_uniform_color([0.961, 0.475, 0.000])

        
                # Output FPS
                if (self.idx % fps_interval_len == 0):
                    end = time.time()
                    elapsed = end - start
                    start = time.time()
                    self.output_fps.text = 'FPS: {:.3f}'.format(fps_interval_len /
                                                                elapsed)

                # Output info
                info = 'Frame {}/{}\n\n'.format(self.idx, 500)
                info += 'Transformation:\n{}\n\n'.format(
                    np.array2string(T_frame_to_model.numpy(),
                                    precision=3,
                                    max_line_width=40,
                                    suppress_small=True,
                                    formatter={'float_kind': lambda x: f"{x:.2f}"}))
                info += time_step_info

                self.output_info.text = info

                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.update_render(
                        agent_id,
                        depth,
                        color,
                        semantic,                             
                        vis_image,
                        frustum,
                        point_sum_points,
                        point_sum_colors,
                        traj,
                        plan_path,
                        Open3d_goal_pose)
                        )
                
                if agent_id == 0:
                    self.idx += 1
                
        
                        


            time.sleep(0.1)

