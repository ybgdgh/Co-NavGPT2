from collections import deque, defaultdict
from typing import Dict
from itertools import count
import os
import logging
import time
import torch
import numpy as np
import random
from habitat import Env, logger
from utils.shortest_path_follower import ShortestPathFollowerCompat

from agents.vlm_multi_agents import VLM_Agent
from arguments import get_args
from utils import chat_utils
import system_prompt
import utils.visualization as vu

from habitat import make_dataset
from habitat.config.default import get_config

import cv2
import open3d as o3d
from multiprocessing import Process, Queue
import multiprocessing as mp

from utils.explored_map_utils import Global_Map_Proc, detect_frontier

def CoNav_env(args, config, rank, dataset, send_queue, receive_queue):
    
    args.rank = rank
    random.seed(config.SEED+rank)
    np.random.seed(config.SEED+rank)
    torch.manual_seed(config.SEED+rank)
    torch.set_grad_enabled(False)
    
    env = Env(config, dataset)

    num_episodes = len(env.episodes)
    print("num_episodes: ", num_episodes)
    receive_queue.put(num_episodes)
    
    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = config.SIMULATOR.NUM_AGENTS
    agent = []
    for i in range(num_agents):
        follower = ShortestPathFollowerCompat(
            env._sim, 0.1, False, i
        )
        agent.append(VLM_Agent(args, i, follower))
        
    map_process = Global_Map_Proc(args)
    
    start_signal = send_queue.get()

    print(start_signal)
    count_episodes = 0
    goal_points = []
    while count_episodes < num_episodes:
        observations = env.reset()
        actions = []
        map_process.reset()
        
        agent_state = env.sim.get_agent_state(0)
        for i in range(num_agents):
            agent[i].reset(observations[i], agent_state)
            actions.append(0)
            
        count_steps = 0
        point_sum = o3d.geometry.PointCloud()
        while not env.episode_over:
            start = time.time()
            visited_vis = []
            pose_pred = []
            point_sum.clear()
            found_goal = False
            clean_diff = True
            for i in range(num_agents):
                agent_state = env.sim.get_agent_state(i)
                agent[i].mapping(observations[i], agent_state)
                point_sum += agent[i].point_sum
                visited_vis.append(agent[i].visited_vis)
                pose_pred.append([agent[i].current_grid_pose[1], int(agent[i].map_size)-agent[i].current_grid_pose[0], np.deg2rad(agent[i].relative_angle)])
                if agent[i].found_goal:
                    found_goal = True 
                if agent[i].clean_diff:
                    clean_diff = False
                    
            obstacle_map, explored_map, top_view_map = map_process.Map_Extraction(point_sum, agent[0].camera_position[1], clean_diff)
            
            if (agent[0].l_step % args.num_local_steps == args.num_local_steps - 1 or agent[0].l_step == 0) and not found_goal:
                goal_points.clear()
                # if args.nav_mode == "gpt":
                target_score, target_edge_map, target_point_list = map_process.Frontier_Det(threshold_point=8)
                for i in range(num_agents):
                    if agent[i].curr_frontier_count > 2*args.num_local_steps + 1 and len(target_point_list) > 0 and args.fill_mode:
                        map_process.obstacle_map[target_edge_map == int(goal_frontiers["robot_"+ str(i)].split('_')[1])+1]
                        obstacle_map[target_edge_map == int(goal_frontiers["robot_"+ str(i)].split('_')[1])+1]
                        agent[i].curr_frontier_count = 0
                        target_score, target_edge_map, target_point_list = map_process.Frontier_Det(threshold_point=8)
                        
                if len(target_point_list) > 0 and agent[0].l_step > 0:
                    candidate_map_list = chat_utils.get_all_candidate_maps(target_edge_map, top_view_map, pose_pred)
                    message = chat_utils.message_prepare(system_prompt.system_prompt, candidate_map_list, agent[i].goal_name)
            
                    goal_frontiers = chat_utils.chat_with_gpt4v(message)
                    for i in range(num_agents):
                        goal_points.append(target_point_list[int(goal_frontiers["robot_"+ str(i)].split('_')[1])])
                        
                else:
                    for i in range(num_agents):
                        action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
                        goal_points.append([int(action[0]), int(action[1])])
                            
                          
            
            goal_map = []
            for i in range(num_agents):
                agent[i].obstacle_map = obstacle_map
                agent[i].explored_map = explored_map
                actions[i] = agent[i].act(goal_points[i])
                goal_map.append(agent[i].goal_map)
            
            if args.visualize or args.print_images:
                vis_image = vu.Visualize(
                    args, agent[0].l_step, 
                    pose_pred, 
                    obstacle_map, 
                    explored_map, 
                    agent[0].goal_id, 
                    visited_vis, 
                    target_edge_map, 
                    goal_map, 
                    top_view_map,
                    agent[0].episode_n,
                    rank)
            
            # print(actions)
            observations = env.step(actions)
            
            step_end = time.time()
            count_steps += 1

        infos = 0
        if (
            0 in actions and env.get_metrics()["spl"]
        ):
            # print("you successfully navigated to destination point")
            infos = 1 #success
        else:
            # print("your navigation was not successful")
            if count_steps >= config.ENVIRONMENT.MAX_EPISODE_STEPS - 1:
                infos = 2 # exploration
            else:
                infos = 3 # detection
                
        count_episodes += 1

        metrics = env.get_metrics()
        
        receive_queue.put([metrics, infos, count_steps])

def main():

    # ------------------------------------------------------------------
    ##### Setup Logging
    # ------------------------------------------------------------------
    args = get_args()
        
    log_dir = "{}/logs/{}/".format(args.dump_location, args.nav_mode)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_dir + "multi-agent.log",
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    # print(args)
    logging.info(args)
    
    agg_metrics: Dict = defaultdict(float)
    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    ##### Setup Configuration
    # ------------------------------------------------------------------
    mp_ctx = mp.get_context("forkserver")
    receive_queue = mp_ctx.Queue() 
    send_queue = mp_ctx.Queue() 
    
    config_env = get_config(config_paths=["configs/"+ args.task_config])
    args.turn_angle = config_env.SIMULATOR.TURN_ANGLE
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config_env.defrost()
    config_env.SIMULATOR.NUM_AGENTS = args.num_agents
    config_env.SIMULATOR.AGENTS = ["AGENT_"+str(i) for i in range(args.num_agents)]
    config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config_env.freeze()
    # ------------------------------------------------------------------
    
    
    # ------------------------------------------------------------------
    ##### Setup Environment and Agents
    # ------------------------------------------------------------------
    scenes = config_env.DATASET.CONTENT_SCENES
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    if "*" in config_env.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config_env.DATASET)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")
    num_episode = []
    processes = []
    for i in range(args.num_processes):
        proc_config = config_env.clone()
        proc_config.defrost()

        if len(scenes) > 0:
            proc_config.DATASET.CONTENT_SCENES = scenes[
                sum(scene_split_sizes[:i]):
                sum(scene_split_sizes[:i + 1])
            ]
            print("Thread {}: {}".format(i, proc_config.DATASET.CONTENT_SCENES))

        dataset = make_dataset(proc_config.DATASET.TYPE, config=proc_config.DATASET)
        proc_config.SIMULATOR.SCENE = dataset.episodes[0].scene_id
        proc_config.freeze()
     
        proc = mp_ctx.Process(target=CoNav_env, args=(args, proc_config, i, dataset, send_queue, receive_queue))
        processes.append(proc)
        proc.start()
        
        num_episode.append(receive_queue.get())

    num_episodes = sum(num_episode)
    print("total num_episodes: ", num_episodes)
    logging.info(num_episodes)
    
    for i in range(args.num_processes):
        send_queue.put('start!')
        
    count_episodes = 0
    agg_metrics: Dict = defaultdict(float)
    total_fail = []
    total_steps = 0
    start = time.time()
    while count_episodes < num_episodes:
        
        if not receive_queue.empty():
            print("received")
            count_episodes += 1
            metrics, infos, count_steps = receive_queue.get()
            
            total_steps += count_steps
            
            for m, v in metrics.items():
                agg_metrics[m] += v
                
            if infos > 0:
                total_fail.append(infos) 

            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(total_steps ),
                "FPS {},".format(int(total_steps  / (end - start)))
            ]) + '\n'
            
            log += "Failed Case: exploration/detection/success/total:"
            log += " {:.0f}/{:.0f}/{:.0f}({:.0f}),".format(
                total_fail.count(2),
                total_fail.count(3),
                total_fail.count(1),
                len(total_fail)) + '\n'
            
            log += "Metrics: "
            log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)

            print(log)
            logging.info(log)


if __name__ == "__main__":
    main()
    
    
