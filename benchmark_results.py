import pandas as pd
from matplotlib import pyplot as plt

sac_df = pd.read_csv("benchmarks/monitor_sac_agent/monitor.csv", skiprows=2, names=['rewards_sac', 'code', 'timestep'])
sac_df['rewards_sac'].plot(legend=True)

ddpg_df = pd.read_csv("benchmarks/monitor_ddpg_agent/monitor.csv", skiprows=2, names=['rewards_ddpg', 'code', 'timestep'])
ddpg_df['rewards_ddpg'].plot(legend=True)

ppo_df = pd.read_csv("benchmarks/monitor_ppo_agent/monitor.csv", skiprows=2, names=['rewards_ppo', 'code', 'timestep'])
ppo_df['rewards_ppo'].plot(legend=True)

plt.show()
