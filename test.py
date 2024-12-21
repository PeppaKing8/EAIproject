import gymnasium as gym
import cv2
import os
import matplotlib.pyplot as plt
import mani_skill.envs
from mani_skill.envs.tasks.humanoid.humanoid_drawer import UnitreeG1OpenDrawerEnv
import imageio

env = gym.make(
    "HumanoidOpenDrawer-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="rgbd", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_delta_pos", # there is also "pd_joint_delta_pos", ...
    render_mode=None
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)
frames = []
obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
step = 0
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    rgb_frame = obs['sensor_data']['base_camera']['rgb'][0]
    depth_frame = obs['sensor_data']['base_camera']['depth'][0]
    plt.imshow(rgb_frame)
    plt.savefig(f"fig/rgb_frame_{step}.png")
    plt.clf()
    plt.imshow(depth_frame)
    plt.savefig(f"fig/depth_frame_{step}.png")
    plt.clf()
    
    frames.append(rgb_frame)
    step += 1
    # print(obs)
    print(f"Step {step}, reward {reward}")

env.close()

# Save the frames as a gif
imageio.mimsave('fig/rgb_frames.gif', frames, fps=10)