import gym
from stable_baselines3 import SAC, DDPG, PPO
import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw


# The following function comes from:
# https://stackoverflow.com/questions/60765613/how-to-show-episode-in-rendered-openai-gym-environment
def _label_with_episode_number(curr_frame, model_type, episode_num):
    im = Image.fromarray(curr_frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), "Model: " + model_type + " - Frame: " + str(episode_num+1), fill=text_color)
    return im


ENV_TYPE = "BipedalWalker-v3"
env = gym.make(ENV_TYPE)


# Soft Actor Critic ; Best model
print('Soft Actor Critic ; Best model')
sac_agent = SAC.load("./benchmarks/monitor_sac_agent/best_model.zip")
state = env.reset()
frames = []
for i in range(600):
    action, _states = sac_agent.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    frame = env.render(mode='rgb_array')
    frames.append(_label_with_episode_number(frame, "SAC", episode_num=i))
    if done:
        state = env.reset()
env.close()
imageio.mimwrite(os.path.join('./gifs/', 'sac_agent.gif'), frames, fps=60)

# Double Deterministic Policy Gradient ; Best model
print('Double Deterministic Policy Gradient ; Best model')
ddpg_agent = DDPG.load("./benchmarks/monitor_ddpg_agent/best_model.zip")
state = env.reset()
frames = []
for i in range(600):
    action, _states = ddpg_agent.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    frame = env.render(mode='rgb_array')
    frames.append(_label_with_episode_number(frame, "DDPG", episode_num=i))
    if done:
        state = env.reset()
env.close()
imageio.mimwrite(os.path.join('./gifs/', 'ddpg_agent.gif'), frames, fps=60)


# Proximal Policy Optimization ; Best model
print('Proximal Policy Optimization ; Best model')
ppo_agent = PPO.load("./benchmarks/monitor_ppo_agent/best_model.zip")
state = env.reset()
frames = []
for i in range(600):
    action, _states = ppo_agent.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    frame = env.render(mode='rgb_array')
    frames.append(_label_with_episode_number(frame, "PPO", episode_num=i))
    if done:
        state = env.reset()
env.close()
imageio.mimwrite(os.path.join('./gifs/', 'ppo_agent.gif'), frames, fps=60)
