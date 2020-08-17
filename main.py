from runner import Runner
#from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args,\
    get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args
#from ma_maze_env import MeetEnv
from ma_maze_env import  MeetEnv_Undisplay as MeetEnv
from overcooked_env_wrapper import FeatureOverCooked
import os

# python main.py --map=3m --alg=qmix
if __name__ == '__main__':
    for i in range(1):
        args = get_common_args()
        ## no communication
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        # communication
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        #env=FeatureOverCooked()
        env=MeetEnv()
        '''
            env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        '''
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)
        #if args.learn:
        if True:
            runner.run(i)
        else:
            win_rate = runner.evaluate_sparse()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
