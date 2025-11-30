import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Optional
import numpy as np
import random
from collections import namedtuple, deque
from src.env.pic import PIC
from src.control.rl.reward import Reward
from src.control.actuator import E_field

torch.backends.cudnn.benchmark = True

# transition
Transition = namedtuple(
    'Transition',
    ('state','action','next_state','reward','done')
)

class ReplayBuffer(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)

    def get(self):
        return self.memory.pop()

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()

# Function for initialization
def hidden_init(layer: torch.nn.Linear):
    fan_in = layer.weight.data.size()[1]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

# Soft update for parameters: xf = t*xi + (1-t)*xf
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Actor(nn.Module):
    def __init__(
        self,
        input_dim:int,
        mlp_dim:int,
        n_actions:int,
        output_min:float= -1.0,
        output_max:float= 1.0,
        x_norm:float=50.0,
        v_norm:float=10.0,
    ):

        super(Actor, self).__init__()

        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.n_actions = n_actions

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

        self.fc_out = nn.Linear(mlp_dim, n_actions)

        self.reset_parameters()

    def reset_parameters(self):

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

        self.fc_out.weight.data.uniform_(*hidden_init(self.fc_out))

    def forward(self, x:torch.Tensor)->torch.Tensor:
    
        x_pos = x[:,:self.input_dim//2] / self.x_norm
        x_vel = x[:,self.input_dim//2:] / self.v_norm
        
        z = torch.cat([x_pos, x_vel], dim=1)
        z = F.tanh(self.fc1(self.norm1(z)))
        z = F.tanh(self.fc2(self.norm2(z)))
        z = F.tanh(self.fc3(self.norm3(z)))

        mu = F.tanh(self.fc_out(z))
        
        return mu

    def sample(self, x:torch.Tensor)->torch.Tensor:
        mu = self.forward(x)
     
        # Rescale action space with bounded region
        action = (0.5 + 0.5 * mu) * (self.max_values.to(x.device) - self.min_values.to(x.device)) + self.min_values.to(x.device)

        return action

    def get_action(self, x:np.ndarray, device:str)->np.ndarray:
        x = x.ravel()
        state = torch.from_numpy(x).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            action = self.sample(state)
        
        return action.squeeze(0).cpu().numpy()

class Critic(nn.Module):
    def __init__(
        self,
        input_dim : int, 
        mlp_dim : int, 
        n_actions:int, 
        x_norm:float = 50.0, 
        v_norm:float = 10.0
        ):
        super().__init__()
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

        self.fc_out = nn.Linear(mlp_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

        self.fc_out.weight.data.uniform_(*hidden_init(self.fc_out))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Normalize
        x_pos = x[:,:self.input_dim//2] / self.x_norm
        x_vel = x[:,self.input_dim//2:] / self.v_norm
        
        z = torch.cat([x_pos, x_vel, a], dim=1)
        z = F.tanh(self.fc1(self.norm1(z)))
        z = F.tanh(self.fc2(self.norm2(z)))
        z = F.tanh(self.fc3(self.norm3(z)))
        q = F.tanh(self.fc_out(z))
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(
        self, 
        size: int, 
        mu: float = 0.0, 
        theta: float = 0.15, 
        sigma: float = 0.2
        ):
        
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


# update policy
def update_policy(
    memory: ReplayBuffer,
    q_network: Critic,
    p_network: Actor,
    target_q_network: Critic,
    target_p_network: Actor,
    q_optimizer: torch.optim.Optimizer,
    p_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    batch_size: int = 32,
    gamma: float = 0.99,
    tau: float = 1e-2,
    device: Optional[str] = "cpu",
):

    q_network.train()
    p_network.train()

    target_q_network.eval()
    target_p_network.eval()

    if device is None:
        device = "cpu"

    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction = "mean") # Huber Loss for critic network

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Parsing state, action, reward, done, and next state
    state_batch = torch.cat(batch.state).float().to(device)
    action_batch = torch.cat(batch.action).float().to(device)
    reward_batch = torch.cat(batch.reward).float().to(device)   

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device = device, dtype = torch.bool)

    with torch.no_grad():
        next_action = target_p_network.sample(non_final_next_states)
        target_q = target_q_network(non_final_next_states, next_action)
        target_q = reward_batch.unsqueeze(1) + gamma * target_q

    # Update Q network
    q = q_network(state_batch, action_batch)
    q_loss = criterion(q, target_q)

    q_optimizer.zero_grad()
    q_loss.backward()

    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)

    q_optimizer.step()

    # Update policy network
    action_batch_sampled = p_network.sample(state_batch)
    p_loss = (-1) * q_network(state_batch, action_batch_sampled).mean()
    
    p_optimizer.zero_grad()
    p_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(p_network.parameters(), max_norm=1.0)

    p_optimizer.step()

    # Update target newtorks
    soft_update(target_q_network, q_network, tau)
    soft_update(target_p_network, p_network, tau)
    
    return q_loss, p_loss


def train(
    env: PIC,
    actuator: E_field,
    memory: ReplayBuffer,
    q_network: Critic,
    p_network: Actor,
    target_q_network: Critic,
    target_p_network: Actor,
    q_optimizer: torch.optim.Optimizer,
    p_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    batch_size: int = 10,
    tau: float = 1e-2,
    gamma: float = 0.99,
    device: Optional[str] = "cpu",
    num_episode: int = 10000,
    Nt: int = 1000,
    verbose: int = 8,
    save_last: Optional[str] = None,
    save_best: Optional[str] = None,    
    alpha: float = 0.1,
    noise_scale: float = 0.1,
):

    # minimum buffer size to start training
    min_buffer_size = 1024

    if device is None:
        device = "cpu"

    # Reward class
    reward_cls = Reward(env.init_dist.get_init_state(), env.N_mesh, env.L, -25.0, 25.0, env.n0, alpha)

    # Trajectory
    q_loss_traj = []
    p_loss_traj = []
    reward_traj = []

    best_reward = None

    # Ornstein-Uhlenbeck noise
    ou_noise = OrnsteinUhlenbeckNoise(p_network.n_actions)

    for i_episode in tqdm(range(num_episode), desc = '# Training DDPG controller...'):

        # initialize the simulation and actuator setup
        env.reinit()
        actuator.reinit()
        reward_cls.reinit()
        ou_noise.reset()

        reward_list = []
        q_loss_list = []
        p_loss_list = []

        for idx_t in range(Nt):

            p_network.eval()

            state = env.get_state().ravel() # state: 2N-array
            state_tensor = torch.from_numpy(state).unsqueeze(0).float() # state_tensor: (1,2N)

            with torch.no_grad():
                action_tensor = p_network.sample(state_tensor.to(device))
                action = action_tensor.squeeze(0).cpu().numpy()

            # Add exploration noise
            noise = ou_noise.sample() * noise_scale
            action = np.clip(action + noise, p_network.output_min, p_network.output_max)

            # update actuator
            actuator.update_E(coeff_cos = None, coeff_sin = action)

            # update state
            env.update_state(E_external=actuator.compute_E())

            # get new state
            next_state = env.get_state().ravel()
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).float() 

            # compute cost
            reward = reward_cls.compute_reward(state, None)           
            reward_tensor = torch.tensor([reward])

            # save trajectory into memory
            done = False if idx_t < Nt - 1 else True
            memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done)

            # update policy
            if memory.__len__() >= min_buffer_size and idx_t % 4 == 0:

                q_loss, p_loss = update_policy(
                    memory,
                    q_network,
                    p_network,
                    target_q_network,
                    target_p_network,
                    q_optimizer,
                    p_optimizer,
                    criterion,
                    batch_size,
                    gamma,
                    tau,
                    device,
                )

                reward_list.append(reward.item())
                q_loss_list.append(q_loss.detach().cpu().numpy().item())
                p_loss_list.append(p_loss.detach().cpu().numpy().item())

                # save the weight parameters
                torch.save(p_network.state_dict(), save_last)

            if done and idx_t < Nt - 1:
                print("| episode:{} | simulation terminated with progress: {:.1f} percent".format(i_episode+1, 100 * (idx_t + 1)/Nt))
                break

        if i_episode % verbose == 0:
            print("| episode:{} | p loss:{:.4f} | q loss:{:.4f} | reward:{:.4f}".format(i_episode+1, p_loss_list[-1], q_loss_list[-1], reward))

        reward_mean = np.mean(reward_list)
        q_loss_mean = np.mean(q_loss_list)
        p_loss_mean = np.mean(p_loss_list)

        reward_traj.append(reward_mean)
        q_loss_traj.append(q_loss_mean)
        p_loss_traj.append(p_loss_mean)

        if best_reward is not None:
            if reward_mean > best_reward:
                torch.save(p_network.state_dict(), save_best)
        else:
            best_reward = reward_mean
            torch.save(p_network.state_dict(), save_best)

    print("# Training DDPG controller process complete")

    reward = np.array(reward_traj)
    q_loss = np.array(q_loss_traj)
    p_loss = np.array(p_loss_traj)

    return reward, q_loss, p_loss
