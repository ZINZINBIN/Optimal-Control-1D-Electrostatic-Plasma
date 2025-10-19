import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Optional
from torch.distributions import Normal
import random
import numpy as np
from collections import namedtuple, deque
from src.env.pic import PIC
from src.control.rl.reward import Reward
from src.control.actuator import E_field

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True 

# transition
Transition = namedtuple(
    'Transition',
    ('state','action','next_state','reward','done')
)

class ReplayBuffer(object):
    def __init__(self, capacity : int):
        self.memory = deque([], maxlen = capacity)
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def get(self):
        return self.memory.pop()
    
    def sample(self, batch_size : int):
        return random.sample(self.memory, batch_size)
    
    def clear(self):
        self.memory.clear()
            
def hidden_init(layer:torch.nn.Linear):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions : int, mu_min:float = -10.0, mu_max:float = 10.0, log_std_min:float = -2.0, log_std_max:float = 2.0, output_min:float = -1.0, output_max:float=1.0, x_norm:float = 50.0, v_norm:float = 10.0):
        super(Actor, self).__init__()

        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.n_actions = n_actions

        self.mu_min = mu_min
        self.mu_max = mu_max
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.output_min = output_min
        self.output_max = output_max
        
        self.min_values = torch.Tensor([output_min for _ in range(n_actions)])
        self.max_values = torch.Tensor([output_max for _ in range(n_actions)])

        self.x_norm = x_norm
        self.v_norm = v_norm

        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(input_dim)

        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.norm2 = nn.LayerNorm(mlp_dim)

        self.fc3 = nn.Linear(mlp_dim, mlp_dim)
        self.norm3 = nn.LayerNorm(mlp_dim)

        self.layer_mu = nn.Linear(mlp_dim, n_actions)
        self.layer_log_std = nn.Linear(mlp_dim, n_actions)
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        
        self.layer_mu.weight.data.uniform_(*hidden_init(self.layer_mu))
        self.layer_log_std.weight.data.uniform_(*hidden_init(self.layer_log_std))
        
    def forward(self, x:torch.Tensor):
        
        z = torch.cat([x.clone()[:,:self.input_dim//2] / self.x_norm, x.clone()[:,self.input_dim//2:] / self.v_norm], dim=1)

        z = F.tanh(self.fc1(self.norm1(z)))
        z = F.tanh(self.fc2(self.norm2(z)))
        z = F.tanh(self.fc3(self.norm3(z)))

        mu = self.layer_mu(z)
        mu = torch.clamp(mu, min = self.mu_min, max = self.mu_max)
        
        log_std = self.layer_log_std(z)
        log_std = torch.clamp(log_std, min = self.log_std_min, max = self.log_std_max)
        
        return mu, log_std

    def sample(self, x : torch.Tensor, deterministic:bool = False):
        
        mu, log_std = self.forward(x)
        std = log_std.exp()
        
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
        
        return action, entropy, log_probs

    def get_action(self, x:np.ndarray, device:str):
        x = x.ravel()
        state = torch.from_numpy(x).unsqueeze(0).float().to(device)
        action, _, _ = self.sample(state)
        return action.detach().squeeze(0).cpu().numpy()
    
class QNetwork(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions:int, x_norm:float = 50.0, v_norm:float = 10.0):
        super(QNetwork, self).__init__()

        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.n_actions = n_actions

        self.x_norm = x_norm
        self.v_norm = v_norm

        self.fc1 = nn.Linear(input_dim + n_actions, mlp_dim)
        self.norm1 = nn.LayerNorm(input_dim + n_actions)

        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.norm2 = nn.LayerNorm(mlp_dim)

        self.fc3 = nn.Linear(mlp_dim, mlp_dim)
        self.norm3 = nn.LayerNorm(mlp_dim)

        self.fc_v = nn.Linear(mlp_dim, 1)
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        
    def forward(self, x:torch.Tensor, a:torch.Tensor):

        z = torch.cat([x.clone()[:,:self.input_dim//2] / self.x_norm, x.clone()[:,self.input_dim//2:] / self.v_norm], dim=1)
    
        z = torch.cat([z, a], dim = 1)
        z = F.tanh(self.fc1(self.norm1(z)))
        z = F.tanh(self.fc2(self.norm2(z)))
        z = F.tanh(self.fc3(self.norm3(z)))

        value = self.fc_v(z)
        return value

class Critic(nn.Module):
    def __init__(self, input_dim : int, mlp_dim : int, n_actions:int, x_norm:float = 50.0, v_norm:float = 10.0):
        super().__init__()
        self.Q1 = QNetwork(input_dim, mlp_dim, n_actions, x_norm, v_norm)
        self.Q2 = QNetwork(input_dim, mlp_dim, n_actions, x_norm, v_norm)
        
    def forward(self, state : torch.Tensor, action : torch.Tensor):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2
    
    def initialize(self):
        self.Q1.reset_parameters()
        self.Q2.reset_parameters()

# update policy
def update_policy(
    memory : ReplayBuffer, 
    q_network:Critic,
    p_network:Actor,
    target_q_network:Critic,
    target_entropy : torch.Tensor,
    log_alpha : Optional[torch.Tensor],
    p_optimizer:torch.optim.Optimizer,
    q_optimizer:torch.optim.Optimizer,
    a_optimizer:torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    batch_size:int = 10,
    gamma : float = 0.99, 
    tau : float = 1e-2,
    device : Optional[str] = "cpu"
    ):

    q_network.train()
    p_network.eval()
    target_q_network.eval()

    if device is None:
        device = "cpu"
    
    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction = "mean") # Huber Loss for critic network
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device = device, dtype = torch.bool)
 
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.cat(batch.state).float().to(device)
    action_batch = torch.cat(batch.action).float().to(device)
    reward_batch = torch.cat(batch.reward).float().to(device)
    
    # Normalizing the rewards
    # reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-6)
        
    alpha = log_alpha.exp()
   
    # step 1. update Q-network parameters
    # Q = r + r'* Q'(s_{t+1}, J(a|s_{t+1}))
    # J = Loss[(Q - (r + r'*Q(s_{t+1}, a_{t+1})))^2]
    q1, q2 = q_network(state_batch, action_batch)
    
    with torch.no_grad():
        next_action_batch, next_entropy, _ = p_network.sample(non_final_next_states)
        next_q1, next_q2 = target_q_network(non_final_next_states, next_action_batch)
        
        next_q = torch.zeros((batch_size,1), device = device)
        next_q[non_final_mask] = torch.min(next_q1, next_q2) + alpha.to(device) * next_entropy
        
    bellman_q_values = reward_batch.unsqueeze(1) + gamma * next_q
    bellman_q_values = torch.clamp(bellman_q_values, -1e3, 1e3)

    # Update Q network  
    q_optimizer.zero_grad()
    
    q1_loss = criterion(q1, bellman_q_values)
    q2_loss = criterion(q2, bellman_q_values)
    q_loss = q1_loss + q2_loss
    q_loss.backward()

    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
    
    q_optimizer.step()
    
    # Update policy network
    q_network.eval()
    p_network.train()
    
    p_optimizer.zero_grad()
    
    action_batch_sampled, entropy, _ = p_network.sample(state_batch)
    q1_pi, q2_pi = q_network(state_batch, action_batch_sampled)
    
    q = torch.min(q1_pi, q2_pi).detach()
    
    p_loss = torch.mean(q + entropy * alpha.detach()) * (-1)
    p_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(p_network.parameters(), max_norm=1.0)
        
    p_optimizer.step()
    
    # Update temperature alpha
    entropy_loss = (-1) * torch.mean(log_alpha.to(device) * (target_entropy - entropy).detach())
    a_optimizer.zero_grad()
    entropy_loss.backward()
    a_optimizer.step()
    
    # Update target q network parameters
    for target_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    return q1_loss, q2_loss, p_loss

def train(
    env:PIC,
    actuator:E_field,
    memory : ReplayBuffer, 
    q_network:Critic,
    p_network:Actor,
    target_q_network:Critic,
    target_entropy : torch.Tensor,
    log_alpha : Optional[torch.Tensor],
    p_optimizer:torch.optim.Optimizer,
    q_optimizer:torch.optim.Optimizer,
    a_optimizer:torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    batch_size:int = 10,
    tau : float = 1e-2,
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    num_episode : int = 10000,  
    Nt:int = 1000,
    verbose : int = 8,
    save_last : Optional[str] = None,
    save_best : Optional[str] = None,
    alpha:float = 0.1,
    ):

    if device is None:
        device = "cpu"

    # Reward class
    reward_cls = Reward(env.init_dist.get_init_state(), env.N_mesh, env.L, -25.0, 25.0, env.n0, alpha)
    
    # Trajectory
    loss_traj = []
    reward_traj = []
    
    best_reward = None

    for i_episode in tqdm(range(num_episode), desc = '# Training SAC controller...'):

        # initialize the simulation and actuator setup
        env.reinit()
        actuator.reinit()
        reward_cls.reinit()
        
        reward_list = []
        loss_list = []
      
        for idx_t in range(Nt):

            p_network.eval()

            state = env.get_state().ravel() # state: 2N-array
            state_tensor = torch.from_numpy(state).unsqueeze(0).float() # state_tensor: (1,2N)

            with torch.no_grad():
                action_tensor, _, _ = p_network.sample(state_tensor.to(device))
                
            action = action_tensor.detach().squeeze(0).cpu().numpy()

            # update actuator
            actuator.update_E(coeff_cos = action[:p_network.n_actions//2], coeff_sin = action[p_network.n_actions//2:])

            # update state
            env.update_state(E_external=actuator.compute_E())

            # get new state
            next_state = env.get_state().ravel()
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).float() 

            # compute cost
            reward = reward_cls.compute_reward(state, actuator.compute_E())           
            reward_tensor = torch.tensor([reward])

            # save trajectory into memory
            done = False if idx_t < Nt - 1 else True
            memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done)

            # update policy
            if memory.__len__() >= batch_size and idx_t % (batch_size // 4) == 0:

                _, _, p_loss = update_policy(
                    memory,
                    q_network,
                    p_network,
                    target_q_network,
                    target_entropy,
                    log_alpha,
                    p_optimizer,
                    q_optimizer,
                    a_optimizer,
                    criterion,
                    batch_size,
                    gamma, 
                    tau,
                    device
                )

                reward_list.append(reward.item())
                loss_list.append(p_loss.detach().cpu().numpy().item())

                # save the weight parameters
                torch.save(p_network.state_dict(), save_last)

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
                torch.save(p_network.state_dict(), save_best)
        else:
            best_reward = reward_mean
            torch.save(p_network.state_dict(), save_best)
            
    print("# Training SAC controller process complete")

    reward = np.array(reward_traj)
    loss = np.array(loss_traj)

    return reward, loss