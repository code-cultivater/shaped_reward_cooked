
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, BASE_REW_SHAPING_PARAMS
from overcooked_ai_py.planning.planners import MediumLevelPlanner

import  numpy as np

mdp = OvercookedGridworld.from_layout_name("cramped_room", rew_shaping_params=BASE_REW_SHAPING_PARAMS)

base_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': mdp.terrain_pos_dict['X'],
    'counter_drop': mdp.terrain_pos_dict['X'][1:2],
    'counter_pickup': [],
    'same_motion_goals': True
}
mlp = MediumLevelPlanner(mdp, mlp_params=base_params)
env = OvercookedEnv.from_mdp(mdp)
feature_class = {
    'fs': mdp.featurize_state,
    #'fslp': mdp.featurize_state_local_perspective,
    #'fsd': mdp.featurize_state_dict
}

avail = np.ones([2, 6])

n_episodes=10
param_set={
'n_agents':2,
    'episode_limit':100
};

class Runner:
    def __init__(self):
        pass;
    def select_actions(self,observations,avail,t_env):
        return (np.random.randint(0,5), np.random.randint(0,5))

for e in range(n_episodes):
    env.reset()
    step = 0
    episode_reward = np.zeros(param_set['n_agents'])

    while (step < param_set['episode_limit']):
        obs = env.state#((1, 2) facing (0, -1) holding None, (3, 1) facing (0, -1) holding None), Objects: [], Order list: None
        print(obs)
        obs = feature_class['fs'](obs, mlp=mlp)
        '''
        (array([ 1.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  1., -2.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  2.,  1.,  0.,
        1.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -1.,
       -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -2.,  2.,  0.,  0.,
        0.,  2.,  1.,  0.,  1.,  0.,  2., -1.,  1.,  2.]), array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -1., -1.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0., -2.,  2.,  0.,  0.,  0.,  2.,  1.,
        0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  1.,
       -2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
        2.,  1.,  0.,  1.,  0.,  1., -2.,  1.,  3.,  1.]))
        '''
        print(obs)
        runner=Runner()
        joint_action = runner.select_actions(observations=obs, avail=0, t_env=0)
        real_action = [Action.INDEX_TO_ACTION[int(index)] for index in joint_action]

        next_state, timestep_sparse_reward, done, info = env.step(real_action)
        rewards = [timestep_sparse_reward] * 2
        episode_reward += rewards
        done = [1 if done else 0] * 2

        # print(info)

        #runner.learn(joint_action=joint_action, reward=rewards, done=done)
        step += 1

        if done[0]:
            break
    obs = env.state
    obs = mdp.featurize_state(obs, mlp=mlp)

    #runner.terminal(observations=obs, avail=avail)