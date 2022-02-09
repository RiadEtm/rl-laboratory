import gym
import torch as th
import os
from stable_baselines3 import SAC, DDPG, PPO
from stable_baselines3.common.monitor import Monitor
from imports.train_monitoring import SaveOnBestTrainingRewardCallback

CHECK_FREQ = 1500
TOTAL_TIMESTEPS = 800000
BENCHMARKS_DIR = "benchmarks"
ENV_TYPE = "BipedalWalker-v3"

"""
Soft Actor Critic 
- MLP Policy
- Paper parameters
"""

print("RUN : Soft Actor Critic")
# Create log dir
log_dir = os.path.join(BENCHMARKS_DIR, "monitor_sac_agent/")
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make(ENV_TYPE)
env = Monitor(env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=CHECK_FREQ, log_dir=log_dir)

# Define and train the agent
sac_agent = SAC("MlpPolicy", env, verbose=1)
sac_agent.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)


"""
Double Deterministic Policy Gradient 
- Custom MLP Policy with 2 layers of 256 units and ReLU activation
- Paper parameters
"""

print("RUN : Double Deterministic Policy Gradient")
# Create log dir
log_dir = os.path.join(BENCHMARKS_DIR, "monitor_ddpg_agent/")
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make(ENV_TYPE)
env = Monitor(env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=CHECK_FREQ, log_dir=log_dir)

# Define and train the agent
ddpg_agent = DDPG("MlpPolicy", env, policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[256,256]), verbose=1)
ddpg_agent.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)


"""
Proximal Policy Optimization 
- Custom MLP Policy with 2 layers of 256 units and ReLU activation
- Paper parameters
"""

print("RUN : Proximal Policy Optimization")
# Create log dir
log_dir = os.path.join(BENCHMARKS_DIR, "monitor_ppo_agent/")
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make(ENV_TYPE)
env = Monitor(env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=CHECK_FREQ, log_dir=log_dir)

# Define and train the agent
ppo_agent = PPO("MlpPolicy", env, policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])]), verbose=1)
ppo_agent.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)


"""
!!! Work in progress : useful snippets below
"""

"""
rewards, _ = evaluate_policy(sac_agent, env, n_eval_episodes=100, return_episode_rewards=True)
plt.plot(rewards)
plt.show()

results_plotter.plot_results(["./monitor_sac_agent"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")
plt.show()
"""


"""
state = env.reset()
while True:
    action, _states = sac_agent.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    env.render()
    if done:
        state = env.reset()
"""