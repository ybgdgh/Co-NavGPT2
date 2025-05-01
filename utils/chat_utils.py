import numpy as np
import ast
import time
import requests
import json
from arguments import get_args
import base64
import openai
from openai import OpenAI
from io import BytesIO
import ast
import cv2

import utils.visualization as vu

client = OpenAI()

gpt_name = [
            'text-davinci-003',
            'gpt-3.5-turbo-0125',
            'gpt-4o',
            'gpt-4o-mini'
        ]           
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


args = get_args()

def get_all_candidate_maps(target_edge_map, top_view_map, pose):
    # show paths in map
    candidate_map_list = []
    for i in range(int(target_edge_map.max())):
        map_with_frontier = top_view_map.copy()
        path_map = np.zeros(target_edge_map.shape)
        path_map[target_edge_map == i+1] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        path_map = cv2.dilate((path_map).astype('uint8'), kernel)
        map_with_frontier[path_map == 1] = [255, 0 , 0]
        map_with_frontier = np.flipud(map_with_frontier)
        map_with_pose = vu.write_number(map_with_frontier, pose, i)
        buffered = BytesIO()
        map_with_pose.save(buffered, format="JPEG")
        candidate_map_list.append(buffered)
        
        opencv_image = np.array(map_with_pose)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("candidate_{}".format(i), opencv_image)
        cv2.waitKey(1)
        
    return candidate_map_list

def get_all_candidate_obs_maps(target_edge_map, top_view_map, pose):
    # show paths in map
    candidate_map_list = []
    for i in range(int(target_edge_map.max())):
        map_with_frontier = top_view_map.copy()
        mask = np.any(top_view_map != 0, axis=-1)
        map_with_frontier[mask] = [255, 255, 255]
        path_map = np.zeros(target_edge_map.shape)
        path_map[target_edge_map == i+1] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        path_map = cv2.dilate((path_map).astype('uint8'), kernel)
        map_with_frontier[path_map == 1] = [255, 0 , 0]
        map_with_frontier = np.flipud(map_with_frontier)
        map_with_pose = vu.write_number(map_with_frontier, pose, i)
        buffered = BytesIO()
        map_with_pose.save(buffered, format="JPEG")
        candidate_map_list.append(buffered)
        
    return candidate_map_list

def get_all_candidate_full_maps(image_id, target_edge_map, top_view_map, pose):
    # show paths in map
    candidate_map_list = []
    for i in range(int(target_edge_map.max())):
        map_with_frontier = top_view_map.copy()
        path_map = np.zeros(target_edge_map.shape)
        path_map[target_edge_map == i+1] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        path_map = cv2.dilate((path_map).astype('uint8'), kernel)
        map_with_frontier[path_map == 1] = [255, 0 , 0]
        map_with_frontier = np.flipud(map_with_frontier)
        
        frontier_image = image_id[i]
        np_image = np.asarray(frontier_image)
        resized_image1 = cv2.resize(np_image, (480, 480))
        resized_image2 = cv2.resize(np.asarray(map_with_frontier), (480, 480))
        
        combined_image = np.hstack((resized_image2, resized_image1))
        
        map_with_pose = vu.write_number_full(combined_image, pose, i)
        
        cv2.imshow("map_with_pose_"+str(i), transform_rgb_bgr(np.asarray(map_with_pose)))
        cv2.waitKey(1)
        buffered = BytesIO()
        map_with_pose.save(buffered, format="JPEG")
        candidate_map_list.append(buffered)
        
    return candidate_map_list

def message_prepare(prompt, candidate_map_list, navigation_instruct):
    base64_image_list = []
    for image_candidate in candidate_map_list:
        base64_image_list.append(base64.b64encode(image_candidate.getvalue()).decode("utf-8"))


    message = []
    message.append({"role": "system", "content": prompt})

    image_contents = []
    image_contents.append({
        "type": "text",
        "text": "two robots need to find a " + navigation_instruct,
    })
    for base64_image in base64_image_list:
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    message.append({"role": "user", "content": image_contents})
    
    return message


def chat_with_gpt4v(chat_history, gpt_type = args.gpt_type):
    num_frontier = len(chat_history[1]['content'])-1
    retries = 5    
    while retries > 0:  
        try: 
            response = client.chat.completions.create(
                model='gpt-4o', 
                response_format = { "type": "json_object" },
                messages=chat_history,
                temperature=0.1,
                max_tokens=100,
            )

            response_message = response.choices[0].message.content
            print('gpt-4o' + " response: ")
            print(response_message)
            try:
                ground_json = ast.literal_eval(response_message)
                # Make sure ground_json has the right size
                if len(ground_json) == args.num_agents+1:
                    # Check if each "robot_i" frontier is in a valid range
                    is_valid = True
                    for i in range(args.num_agents):
                        # If out of range, set is_valid to False and break
                        if int(ground_json[f"robot_{i}"].split('_')[1]) >= num_frontier:
                            is_valid = False
                            break

                    # If still valid after the loop, we're done, return
                    if is_valid:
                        return ground_json
                
            except (SyntaxError, ValueError) as e:
                print(response_message)
        except openai.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except openai.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        retries -=1
            
    # print(ground_json)
    ground_json = {
                    "robot_0": "frontier_0",
                    "robot_1": "frontier_0"
                    }
    return ground_json
