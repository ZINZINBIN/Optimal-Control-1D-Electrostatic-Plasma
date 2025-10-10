import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Optional
from torch.distributions import Normal
import os, pickle
import numpy as np
from collections import namedtuple, deque
from src.env.pic import PIC
from src.control.actuator import E_field
from src.control.objective import estimate_KL_divergence, estimate_f

# transition
Transition = namedtuple(
    'Transition',
    ('state','action','next_state','reward','done','prob_a')
)

class ReplayBuffer(object):
    def __init__(self, capacity : int):
        self.memory = deque([], maxlen = capacity)
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def get_trajectory(self):
        traj = [self.memory[idx] for idx in range(len(self.memory))]
        return traj
    
    def clear(self):
        self.memory.clear()
    
    def save_buffer(self, env_name : str, tag : str = "", save_path : Optional[str] = None):
        
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/', exist_ok=True)

        if save_path is None:
            save_path = "checkpoints/buffer_{}_{}".format(env_name, tag)
            
        print("Process : saving buffer to {}".format(save_path))
        
        with open(save_path, "wb") as f:
            pickle.dump(self.memory, f)
        
    def load_buffer(self, save_path : str):
        print("Process : loading buffer from {}".format(save_path))
        
        with open(save_path, 'rb') as f:
            self.memory = pickle.load(f)

class ActorCritic(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int, std : float = 0.25, output_min:float = -1.0, output_max:float=1.0):
        super(ActorCritic, self).__init__()
        
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.n_actions = n_actions
        self.std = std
        
        self.output_min = output_min
        self.output_max = output_max

        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(input_dim)

        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.norm2 = nn.LayerNorm(mlp_dim)

        self.fc3 = nn.Linear(mlp_dim, mlp_dim//2)
        self.norm3 = nn.LayerNorm(mlp_dim)

        self.fc_pi = nn.Linear(mlp_dim // 2, n_actions)
        self.fc_v = nn.Linear(mlp_dim // 2, 1)

        self.log_std = nn.Parameter(torch.ones(1, n_actions) * std)

        self.min_values = [output_min for _ in range(n_actions)]
        self.max_values = [output_max for _ in range(n_actions)]

    def forward(self, x:torch.Tensor):
        x = F.tanh(self.fc1(self.norm1(x)))
        x = F.tanh(self.fc2(self.norm2(x)))
        x = F.tanh(self.fc3(self.norm3(x)))

        mu = self.fc_pi(x)
        std = self.log_std.exp().expand_as(mu)

        dist = Normal(mu, std)
        value = self.fc_v(x)

        return dist, value

    def sample(self, x : torch.Tensor):
        dist, value = self.forward(x)
        xs = dist.rsample()

        # rescale action range
        action = (0.5 + 0.5 * xs) * (torch.Tensor(self.max_values).to(x.device) - torch.Tensor(self.min_values).to(x.device)) + torch.Tensor(self.min_values).to(x.device)

        # action bounded for stable learning + valid design parameter
        action = torch.clamp(action, min = torch.Tensor(self.min_values).to(x.device), max = torch.Tensor(self.max_values).to(x.device))

        log_probs = dist.log_prob(xs)
        entropy = dist.entropy().mean()

        return action, entropy, log_probs, value

# update policy
def update_policy(
    memory : ReplayBuffer, 
    policy_network : ActorCritic, 
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    gamma : float = 0.99, 
    eps_clip : float = 0.1,
    entropy_coeff : float = 0.1,
    device : Optional[str] = "cpu"
    ):

    policy_network.train()

    if device is None:
        device = "cpu"
    
    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction = 'none') # Huber Loss for critic network
    
    transitions = memory.get_trajectory()
    memory.clear()
    batch = Transition(*zip(*transitions))
 
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).float().to(device)
    action_batch = torch.cat(batch.action).float().to(device)
    prob_a_batch = torch.cat(batch.prob_a).float().to(device) # pi_old
        
    # Multi-step version reward: Monte Carlo estimate
    rewards = []
    discounted_reward = 0
    for reward in reversed(batch.reward):
        discounted_reward = reward + (gamma * discounted_reward)
        rewards.insert(0, discounted_reward)
        
    reward_batch = torch.cat(rewards).float().to(device)
    
    policy_optimizer.zero_grad()
    
    _, _, next_log_probs, next_value = policy_network.sample(non_final_next_states)
    action, entropy, log_probs, value = policy_network.sample(state_batch)
    
    td_target = reward_batch.view_as(next_value) + gamma * next_value
    
    delta = td_target - value        
    ratio = torch.exp(log_probs - prob_a_batch.detach())
    surr1 = ratio * delta
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * delta
    loss = -torch.min(surr1, surr2) + criterion(value, td_target) - entropy_coeff * entropy
    loss = loss.mean()
    loss.backward()

    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1)
        
    policy_optimizer.step()
    
    return loss

def train(
    env:PIC,
    actuator:E_field,
    memory : ReplayBuffer, 
    policy_network : ActorCritic, 
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    gamma : float = 0.99, 
    eps_clip : float = 0.1,
    entropy_coeff : float = 0.1,
    device : Optional[str] = "cpu",
    num_episode : int = 10000,  
    Nt:int = 1000,
    verbose : int = 8,
    save_last : Optional[str] = None,
    ):

    if device is None:
        device = "cpu"

    reward_list = []
    loss_list = []

    for i_episode in tqdm(range(num_episode), desc = '# Training PPO controller...'):

        # initialize the simulation and actuator setup
        env.reinit()
        actuator.reinit()

        init_state = env.get_state().copy()

        for idx_t in range(Nt):

            policy_network.eval()

            state = env.get_state().ravel() # state: 2N-array
            state_tensor = torch.from_numpy(state).unsqueeze(0).float() # state_tensor: (1,2N)

            action_tensor, entropy, log_probs, value = policy_network.sample(state_tensor.to(device))
            action = action_tensor.detach().squeeze(0).cpu().numpy()

            # update actuator
            actuator.update_E(coeff_cos = action[:policy_network.n_actions//2], coeff_sin = action[policy_network.n_actions//2:])

            # update state
            env.update_state(E_external=actuator.compute_E())

            # get new state
            next_state = env.get_state().ravel()
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).float() 

            # compute cost
            reward = estimate_KL_divergence(estimate_f(state, env.N_mesh, env.L, -10.0, 10.0, 1.0), estimate_f(init_state, env.N_mesh, env.L, -10.0, 10.0, 1.0)) * (-1)
            reward_tensor = torch.tensor([reward])

            # save trajectory into memory
            done = False if idx_t < Nt - 1 else True
            memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done, log_probs)

            # update policy
            if memory.__len__() >= memory.capacity:

                policy_loss = update_policy(
                    memory,
                    policy_network,
                    policy_optimizer,
                    criterion,
                    gamma,
                    eps_clip,
                    entropy_coeff,
                    device,
                )

                reward_list.append(reward)
                loss_list.append(policy_loss.detach().cpu().numpy())

                # save the weight parameters
                torch.save(policy_network.state_dict(), save_last)

        if i_episode % verbose == 0:
            print("| episode:{} | loss:{:.4f} | cost:{:.4f}".format(i_episode+1, loss_list[-1], reward * (-1)))

    print("# Training PPO controller process complete")
    
    reward = np.array(reward_list)
    loss = np.array(loss_list)

    return reward, loss