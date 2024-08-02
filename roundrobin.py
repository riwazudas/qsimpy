from env_creator import qsimpy_env_creator


env = qsimpy_env_creator ( 
    env_config = {
        "obs_filter": "rescale_-1_1",
        "reward_filter": None,
        "dataset": "qdataset/qsimpyds_1000_sub_26.csv",
    }
)

class RR:
    def __init__(self,num_qnodes) -> None:
        self.num_qnodes= num_qnodes
        self.current_qnode= 0
    
    def select_qnode(self):
        qnode= self.current_qnode
        self.current_qnode=(self.current_qnode+1)% self.num_qnodes
        return qnode

print("action space",env.action_space)
num_qnode= env.action_space.n
round_robin = RR(num_qnode)
num_ep = 100

for episode in range(num_ep):
    obs = env.reset()
    finished= False
    ep_reward=0

    while not finished: 
        action = round_robin.select_qnode()
        obs, reward, finished, _, info = env.step(action)
        ep_reward += reward

        if finished:
            print(f"Episode{episode} finished with reward {ep_reward} and info {info} ")
            break

env.close()


