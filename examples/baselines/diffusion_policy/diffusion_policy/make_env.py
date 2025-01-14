from typing import Optional
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers import RecordEpisode, FrameStack, CPUGymWrapper
from mani_skill.envs.sapien_env import BaseEnv
from typing import Dict
import torch

class RGBDWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)
    
    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images = []
        cam_data = sensor_data["base_camera"]
        rgb_img = torch.Tensor(cam_data["rgb"])
        rgb_img = rgb_img / 255.0
        depth_img = torch.Tensor(cam_data["depth"])
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        images.append(rgb_img)
        images.append(depth_img)
        # for cam_data in sensor_data.values():
        #     rgb_img = torch.Tensor(cam_data["rgb"])
        #     rgb_img = rgb_img / 255.0
        #     depth_img = torch.Tensor(cam_data["depth"])
        #     depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        #     thres = 0.05
        #     depth_img = torch.min(thres * torch.ones_like(depth_img), depth_img) / thres
        #     # print("rgb:", rgb_img.shape, "depth:", depth_img.shape)
        #     images.append(rgb_img)
        #     images.append(depth_img)

        # print("make_env/", images[0].shape)
        images = torch.concat(images, axis=-1)
        # print("make_env/", images.shape)
        # images = images.flatten(start_dim=1)
        # print("make_env/", images.shape)
        
        ###### GOAL CONDITIONED ######
        # goal = observation['extra']['goal_pos']
        # goal = torch.Tensor(goal)
        # # print("images:", images.shape, "goal:", goal.shape)
        # return {
        #     "image": images,
        #     "goal": goal,
        # }
        ###### GOAL CONDITIONED ######
        
        
        ###### NO GOAL CONDITIONED ######
        return images
        ###### NO GOAL CONDITIONED ######

def make_eval_envs(env_id, num_envs: int, sim_backend: str, env_kwargs: dict, other_kwargs: dict, video_dir: Optional[str] = None, wrappers: list[gym.Wrapper] = []):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "cpu":
        # raise NotImplementedError("CPU vectorized environments are not supported for evaluation. NOTE: This error is written by EAI project, not the original code.")
        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs = dict(), other_kwargs = dict()):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True, source_type="diffusion_policy", source_desc="diffusion_policy evaluation rollout")
                env = gym.wrappers.FrameStack(env, other_kwargs['obs_horizon'])
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk
        vector_cls = gym.vector.SyncVectorEnv if num_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
        env = vector_cls([cpu_make_env(env_id, seed, video_dir if seed == 0 else None, env_kwargs, other_kwargs) for seed in range(num_envs)])
    else:
        env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend, reconfiguration_freq=1, **env_kwargs)
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        env = FrameStack(env, num_stack=other_kwargs['obs_horizon'])
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, save_video=True, source_type="diffusion_policy", source_desc="diffusion_policy evaluation rollout", max_steps_per_video=max_episode_steps)
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env
