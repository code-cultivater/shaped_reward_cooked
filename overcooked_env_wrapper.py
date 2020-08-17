from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, BASE_REW_SHAPING_PARAMS
from overcooked_ai_py.planning.planners import MediumLevelPlanner
import numpy as np

class FeatureOverCooked():
    def __init__(self):
        mdp = OvercookedGridworld.from_layout_name("cramped_room", rew_shaping_params=BASE_REW_SHAPING_PARAMS)
        base_params = {
            'start_orientations': False,
            'wait_allowed': False,
            'counter_goals': mdp.terrain_pos_dict['X'],
            'counter_drop': mdp.terrain_pos_dict['X'][1:2],
            'counter_pickup': [],
            'same_motion_goals': True
        }
        self.feature_state=mdp.featurize_state
        self.mlp = MediumLevelPlanner(mdp, mlp_params=base_params)
        self.env = OvercookedEnv.from_mdp(mdp)
    def reset(self):
        self.env.reset()
        obs=self.feature_state(self.env.state,self.mlp)
        return obs
    def step(self, joint_action, joint_agent_action_info=None):
        decoder_action = [Action.INDEX_TO_ACTION[int(index)] for index in joint_action]
        next_state, timestep_sparse_reward, done, info=self.env.step(decoder_action, joint_agent_action_info)
        next_obs=self.feature_state(self.env.state,self.mlp)
        return next_obs,timestep_sparse_reward,done,info

    def get_env_info(self):
        env_info = {
            "n_actions": Action.NUM_ACTIONS,
            "n_agents": 2,
            "state_shape": 2 * 62,
            "obs_shape": 2 * 62,
            "episode_limit": 200,
        }
        return env_info
    def get_obs(self):
        return [np.concatenate(self.feature_state(self.env.state,self.mlp)),
                np.concatenate(self.feature_state(self.env.state,self.mlp))]
    def get_state(self):
        return np.concatenate(self.feature_state(self.env.state,self.mlp))
    def get_avail_agent_actions(self, agent_id):
        return np.ones(Action.NUM_ACTIONS)
class ShapedRewardFeatureOverCooked():
    def __init__(self):
        mdp = OvercookedGridworld.from_layout_name("cramped_room", rew_shaping_params=BASE_REW_SHAPING_PARAMS)
        base_params = {
            'start_orientations': False,
            'wait_allowed': False,
            'counter_goals': mdp.terrain_pos_dict['X'],
            'counter_drop': mdp.terrain_pos_dict['X'][1:2],
            'counter_pickup': [],
            'same_motion_goals': True
        }
        self.feature_state=mdp.featurize_state
        self.mlp = MediumLevelPlanner(mdp, mlp_params=base_params)
        self.env = OvercookedEnv.from_mdp(mdp)
    def reset(self):
        self.env.reset()
        obs=self.feature_state(self.env.state,self.mlp)
        return obs
    def step(self, joint_action, joint_agent_action_info=None):
        decoder_action = [Action.INDEX_TO_ACTION[int(index)] for index in joint_action]
        next_state, timestep_sparse_reward, done, info=self.env.step(decoder_action, joint_agent_action_info)
        next_obs=self.feature_state(self.env.state,self.mlp)
        return next_obs,timestep_sparse_reward,done,info

    def get_env_info(self):
        env_info = {
            "n_actions": Action.NUM_ACTIONS,
            "n_agents": 2,
            "state_shape": 2 * 62,
            "obs_shape": 2 * 62,
            "episode_limit": 200,
        }
        return env_info
    def get_obs(self):
        return [np.concatenate(self.feature_state(self.env.state,self.mlp)),
                np.concatenate(self.feature_state(self.env.state,self.mlp))]
    def get_state(self):
        return np.concatenate(self.feature_state(self.env.state,self.mlp))
    def get_avail_agent_actions(self, agent_id):
        return np.ones(Action.NUM_ACTIONS)