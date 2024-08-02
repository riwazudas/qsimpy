from env_creator import qsimpy_env_creator
import copy

class GreedyScheduler:
    def __init__(self, num_qnodes, env):
        self.num_qnodes = num_qnodes
        self.env = env

    def select_qnode(self, obs):
        best_reward = float('-inf')
        best_action = 0

        for action in range(self.num_qnodes):
            temp_env= copy.deepcopy(self.env)
            _, reward, done, _, _ = temp_env.step(action)

            if reward > best_reward:
                best_reward = reward
                best_action = action
            self.env.reset()
            
        return best_action

env = qsimpy_env_creator(
    env_config={
        "obs_filter": "rescale_-1_1",
        "reward_filter": None,
        "dataset": "qdataset/qsimpyds_1000_sub_26.csv",
    }
)

print("action space", env.action_space)
num_qnodes = env.action_space.n
greedy_scheduler = GreedyScheduler(num_qnodes, env)
num_ep = 100

for episode in range(num_ep):
    obs = env.reset()
    finished = False
    ep_reward = 0

    while not finished:
        action = greedy_scheduler.select_qnode(obs)
        obs, reward, finished, _, info = env.step(action)
        ep_reward += reward

        if finished:
            print(f"Episode {episode} finished with reward {ep_reward} and info {info}")
            break

env.close()
