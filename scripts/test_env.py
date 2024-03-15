
# envs = gym.wrappers.RecordVideo(
# 	envs,
# 	"./videos",
# 	step_trigger=lambda step: step % 10000 == 0, # record the videos every 10000 steps
# 	video_length=100  # for each video record up to 100 steps
# )


import isaacgym
import bitbotsenv
import torch

num_envs = 20

envs = bitbotsenv.make(
	seed=0, 
	task="KinovaCabinet", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for _ in range(2000):
	random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	envs.step(random_actions)

