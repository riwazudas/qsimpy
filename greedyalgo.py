from env_creator import qsimpy_env_creator

class GreedyScheduler:
    def __init__(self, num_qnodes, env):
        self.num_qnodes = num_qnodes
        self.env = env

    def select_qnode(self, greedy_index):
        greedy_strategy=sorted(self.env.qnodes, key=lambda x: x.next_available_time)
        return self.env.qnodes.index(greedy_strategy[greedy_index])

env = qsimpy_env_creator(
    env_config={
        "obs_filter": "rescale_-1_1",
        "reward_filter": None,
        "dataset": "qdataset/qsimpyds_1000_sub_26.csv",
    }
)

num_qnodes = env.action_space.n
greedy_scheduler = GreedyScheduler(num_qnodes, env)
num_ep = 100

for episode in range(num_ep):
    obs = env.reset()
    finished = False
    ep_reward = 0

    while not finished:
        action = greedy_scheduler.select_qnode(0)
        obs, reward, finished, _, info = env.step(action)
        ep_reward += reward

        if finished:
            print(f"Episode {episode} finished with reward {ep_reward} and info {info}")
            break

env.close()
