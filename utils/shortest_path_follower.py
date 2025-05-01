# Modified from https://github.com/facebookresearch/habitat-lab/blob/v0.1.4/habitat/tasks/nav/shortest_path_follower.py
# Use the Habitat v0.1.4 ShortestPathFollower for compatibility with
# the dataset generation oracle.

from typing import Optional, Union
import math 

import habitat_sim
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
)


EPSILON = 1e-6


def action_to_one_hot(action: int) -> np.array:
    one_hot = np.zeros(len(HabitatSimActions), dtype=np.float32)
    one_hot[action] = 1
    return one_hot


class ShortestPathFollowerCompat:
    """Utility class for extracting the action on the shortest path to the
        goal.
    Args:
        sim: HabitatSim instance.
        goal_radius: Distance between the agent and the goal for it to be
            considered successful.
        return_one_hot: If true, returns a one-hot encoding of the action
            (useful for training ML agents). If false, returns the
            SimulatorAction.
    """

    def __init__(
        self, 
        sim: HabitatSim, 
        goal_radius: float, 
        return_one_hot: bool = True, 
        agent_id: int = 0 
    ):
        assert (
            getattr(sim, "geodesic_distance", None) is not None
        ), "{} must have a method called geodesic_distance".format(
            type(sim).__name__
        )

        self._sim = sim
        self._agent_id = agent_id
        self._max_delta = sim.habitat_config.FORWARD_STEP_SIZE - EPSILON
        self._goal_radius = goal_radius
        self._step_size = sim.habitat_config.FORWARD_STEP_SIZE

        self._return_one_hot = return_one_hot

    def _get_return_value(self, action) -> Union[int, np.array]:
        if self._return_one_hot:
            return action_to_one_hot(action)
        else:
            return action

    def get_next_action(
        self, goal_pos, current_grid_pose, grid_angle, next_stg_x, next_stg_y
    ) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path."""
        if (
            self._sim.geodesic_distance(
                self._sim.get_agent_state(self._agent_id).position, goal_pos
            )
            <= self._goal_radius
        ):
            return HabitatSimActions.STOP

        angle_st_goal = math.degrees(math.atan2(next_stg_x - current_grid_pose[0],
                                                next_stg_y - current_grid_pose[1]))
        angle_agent = (360 - grid_angle) % 360.0
        if angle_agent > 180:
            angle_agent -= 360
        # angle_agent = 360 - self.relative_angle
        relative_angle = angle_agent - angle_st_goal
        if relative_angle > 180:
            relative_angle -= 360
        if relative_angle < -180:
            relative_angle += 360
            
        if relative_angle > self._sim.habitat_config.TURN_ANGLE:
            return self._get_return_value(HabitatSimActions.TURN_RIGHT)  # Right
        elif relative_angle < -self._sim.habitat_config.TURN_ANGLE:
            return self._get_return_value(HabitatSimActions.TURN_LEFT) # Left
        else:
            return self._get_return_value(HabitatSimActions.MOVE_FORWARD)
            
    
    def get_closet_navigable_point(self, target_point):
        return self._sim.pathfinder.snap_point(target_point)
    

    def get_path_points(self, goal_pos):
        return self._sim.get_straight_shortest_path_points(
            self._sim.get_agent_state(self._agent_id).position, self.get_closet_navigable_point(goal_pos)
        )
  