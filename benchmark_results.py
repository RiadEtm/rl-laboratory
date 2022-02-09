import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pprint

AVERAGE_RATIO = 10

palette = sns.color_palette("mako", 3)

sac_df = pd.read_csv("benchmarks/monitor_sac_agent/monitor.csv", skiprows=2, names=['Rewards', 'code', 'timestep'])
ddpg_df = pd.read_csv("benchmarks/monitor_ddpg_agent/monitor.csv", skiprows=2, names=['Rewards', 'code', 'timestep'])
ppo_df = pd.read_csv("benchmarks/monitor_ppo_agent/monitor.csv", skiprows=2, names=['Rewards', 'code', 'timestep'])

sac_rew = pd.DataFrame(np.array(sac_df['Rewards'].iloc[:-9]).reshape(-1, AVERAGE_RATIO), columns=AVERAGE_RATIO*['sac_rewards'])
ddpg_rew = pd.DataFrame(np.array(ddpg_df['Rewards'].iloc[:-1]).reshape(-1, AVERAGE_RATIO), columns=AVERAGE_RATIO*['ddpg_rewards'])
ppo_rew = pd.DataFrame(np.array(ppo_df['Rewards'].iloc[:-5]).reshape(-1, AVERAGE_RATIO), columns=AVERAGE_RATIO*['ppo_rewards'])

cols = list(sac_rew.columns.values) + list(ddpg_rew.columns.values) + list(ppo_rew.columns.values)

draw_df = pd.concat([sac_rew, ddpg_rew, ppo_rew], axis=1, ignore_index=True)
draw_df.columns = cols

sns.lineplot(data=draw_df, palette=palette, dashes=False)
plt.xlabel('Episodes (averaged by ' + str(AVERAGE_RATIO) + ' * 100)')
plt.ylabel('Rewards')
plt.show()
