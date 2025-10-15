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
from src.control.rl.reward import Reward
from src.control.actuator import E_field

# transition
Transition = namedtuple(
    'Transition',
    ('state','action','next_state','reward','done','prob_a')
)

def hidden_init(layer:torch.nn.Linear):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

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

class ActorCritic(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int, std : float = 0.25, output_min:float = -1.0, output_max:float=1.0, x_norm:float = 50.0, v_norm:float = 10.0):
        super(ActorCritic, self).__init__()

        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.n_actions = n_actions
        self.std = std

        self.output_min = output_min
        self.output_max = output_max

        self.x_norm = x_norm
        self.v_norm = v_norm

        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(input_dim)

        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.norm2 = nn.LayerNorm(mlp_dim)

        self.fc3 = nn.Linear(mlp_dim, mlp_dim)
        self.norm3 = nn.LayerNorm(mlp_dim)

        self.fc_pi = nn.Linear(mlp_dim, n_actions)
        self.fc_v = nn.Linear(mlp_dim, 1)

        self.log_std = nn.Parameter(torch.ones(1, n_actions) * np.log(std))

        self.min_values = torch.Tensor([output_min for _ in range(n_actions)])
        self.max_values = torch.Tensor([output_max for _ in range(n_actions)])
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        
        self.fc_pi.weight.data.uniform_(*hidden_init(self.fc_pi))
        self.fc_v.weight.data.uniform_(*hidden_init(self.fc_v))

    def forward(self, x:torch.Tensor):
        z = torch.cat([x.clone()[:,:self.input_dim//2] / self.x_norm, x.clone()[:,self.input_dim//2:] / self.v_norm], dim=1)
        z = F.tanh(self.fc1(self.norm1(z)))
        z = F.tanh(self.fc2(self.norm2(z)))
        z = F.tanh(self.fc3(self.norm3(z)))

        mu = self.fc_pi(z)
        std = self.log_std.exp().expand_as(mu).to(x.device)
        value = self.fc_v(z)

        return mu, std, value

    def sample(self, x : torch.Tensor, deterministic:bool = False):
        mu, std, value = self.forward(x)
        dist = Normal(mu, std)
        
        if deterministic:
            xs = mu
        else:
            xs = dist.rsample()
            
        y = F.tanh(xs)
        
        # Rescale action space with bounded region
        action = (0.5 + 0.5 * y) * (self.max_values.to(x.device) - self.min_values.to(x.device)) + self.min_values.to(x.device)

        log_probs = dist.log_prob(xs)
        log_probs = log_probs.sum(dim = -1, keepdim=True)
        entropy = dist.entropy().mean()

        return action, entropy, log_probs, value

    def get_action(self, x:np.ndarray):
        x = x.ravel()
        state = torch.from_numpy(x).unsqueeze(0).float()
        
        with torch.no_grad():
            action, _, _, _ = self.sample(state, True)
            
        return action.squeeze(0).cpu().numpy()

# update policy
def update_policy(
    memory : ReplayBuffer, 
    policy_network : ActorCritic, 
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    gamma : float = 0.99, 
    eps_clip : float = 0.1,
    entropy_coeff : float = 0.1,
    value_coeff:float = 0.1,
    device : Optional[str] = "cpu",
    k_epoch: int = 4,
    ):

    policy_network.train()

    if device is None:
        device = "cpu"
    
    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction = "mean") # Huber Loss for critic network
    
    transitions = memory.get_trajectory()
    memory.clear()
    batch = Transition(*zip(*transitions))
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.cat(batch.state).float().to(device)
    prob_a_batch = torch.cat(batch.prob_a).float().to(device)
        
    # Multi-step version reward: Monte Carlo estimate
    rewards = []
    discounted_reward = 0
    for reward in reversed(batch.reward):
        discounted_reward = reward + (gamma * discounted_reward)
        rewards.insert(0, discounted_reward)
        
    # Normalizing the rewards
    reward_batch = torch.cat(rewards).float().to(device)
    reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-6)
    
    loss_list = []
    
    for _ in range(k_epoch):
        
        policy_optimizer.zero_grad()
        _, _, _, next_value = policy_network.sample(non_final_next_states)
        _, entropy, log_probs, value = policy_network.sample(state_batch)
        
        td_target = reward_batch.view_as(next_value) + gamma * next_value
        
        delta = td_target - value     
        ratio = torch.exp(log_probs - prob_a_batch.detach())
        
        surr1 = ratio * delta.detach()
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * delta.detach()
        loss = -torch.min(surr1, surr2) + value_coeff * criterion(value, td_target.detach()) - entropy_coeff * entropy
        loss = loss.mean()
        loss.backward()
        
        for param in policy_network.parameters():
            param.grad.data.clamp_(-1.0, 1.0)
            
        policy_optimizer.step()
        
        loss_list.append(loss.detach().cpu().numpy().item())
    
    loss = np.mean(loss_list)
    
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
    value_coeff:float = 0.1,
    device : Optional[str] = "cpu",
    num_episode : int = 10000, 
    Nt:int = 1000,
    verbose : int = 8,
    save_last : Optional[str] = None,
    save_best : Optional[str] = None,
    alpha:float = 0.1,
    beta:float = 0.1,
    k_epoch:int = 4,
    ):

    if device is None:
        device = "cpu"

    # Reward class
    reward_cls = Reward(env.init_dist.get_init_state(), env.N_mesh, env.L, -25.0, 25.0, env.n0, alpha, beta)
    
    # Trajectory
    loss_traj = []
    reward_traj = []
    
    best_reward = None

    for i_episode in tqdm(range(num_episode), desc = '# Training PPO controller...'):

        # initialize the simulation and actuator setup
        env.reinit()
        actuator.reinit()
        reward_cls.reinit()
        
        reward_list = []
        loss_list = []
      
        for idx_t in range(Nt):

            policy_network.eval()

            state = env.get_state().ravel() # state: 2N-array
            state_tensor = torch.from_numpy(state).unsqueeze(0).float() # state_tensor: (1,2N)

            with torch.no_grad():
                action_tensor, _, log_probs, _ = policy_network.sample(state_tensor.to(device))
                
            action = action_tensor.detach().squeeze(0).cpu().numpy()

            # update actuator
            actuator.update_E(coeff_cos = action[:policy_network.n_actions//2], coeff_sin = action[policy_network.n_actions//2:])

            # update state
            env.update_state(E_external=actuator.compute_E())

            # get new state
            next_state = env.get_state().ravel()
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).float() 

            # compute cost
            reward = reward_cls.compute_reward(state, action)           
            reward_tensor = torch.tensor([reward]).float()

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
                    value_coeff,
                    device,
                    k_epoch
                )

                reward_list.append(reward.item())
                loss_list.append(policy_loss)

                # save the weight parameters
                torch.save(policy_network.state_dict(), save_last)

            if done and idx_t < Nt - 1:
                print("| episode:{} | simulation terminated with progress: {:.1f} percent".format(i_episode+1, 100 * (idx_t + 1)/Nt))
                break

        if i_episode % verbose == 0:
            print("| episode:{} | loss:{:.4f} | reward:{:.4f}".format(i_episode+1, loss_list[-1], reward))
            
        reward_mean = np.mean(reward_list)
        loss_mean = np.mean(loss_list)
        
        reward_traj.append(reward_mean)
        loss_traj.append(loss_mean)
        
        if best_reward is not None:
            if reward_mean > best_reward:
                torch.save(policy_network.state_dict(), save_best)
        else:
            best_reward = reward_mean
            torch.save(policy_network.state_dict(), save_best)

    print("# Training PPO controller process complete")

    reward = np.array(reward_traj)
    loss = np.array(loss_traj)

    return reward, loss