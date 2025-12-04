import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Optional
import numpy as np
import random, gc
from collections import namedtuple, deque
from src.env.pic import PIC
from src.control.rl.reward import Reward
from src.control.rl.encode import ParticleEncoder
from src.control.actuator import E_field
from src.interpret.spectrum import compute_E_k_spectrum

torch.backends.cudnn.benchmark = True

# transition
Transition = namedtuple(
    'Transition',
    ('state','action','next_state','reward','done', 'action_bc')
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
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

# Soft update for parameters: xf = t*xi + (1-t)*xf
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

class Actor(nn.Module):
    def __init__(
        self,
        input_dim:int,
        mlp_dim:int,
        n_actions:int,
        output_min:float= -1.0,
        output_max:float= 1.0,
        x_norm:float=1.0,
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

        self.encode = ParticleEncoder(mlp_dim, mlp_dim)

        self.fc1 = nn.Linear(mlp_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(mlp_dim)

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

        # Normalization
        x_pos = x[:,:self.input_dim//2] / self.x_norm
        x_vel = x[:,self.input_dim//2:] / self.v_norm
        z = torch.cat([x_pos, x_vel], dim=1)
        z = self.encode(z)

        z = F.relu(self.norm1(self.fc1(z)))
        z = F.relu(self.norm2(self.fc2(z)))
        z = F.relu(self.norm3(self.fc3(z)))
        mu = F.tanh(self.fc_out(z))
        
        return mu

    def sample(self, x:torch.Tensor)->torch.Tensor:
        mu = self.forward(x)
      
        # Rescale action space with bounded region
        action = (0.5 + 0.5 * mu) * (self.output_max - self.output_min) + self.output_min

        return action

    def get_action(self, x:np.ndarray, device:str)->np.ndarray:
        x = x.ravel()
        state = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action = self.sample(state)

        return action.squeeze(0).cpu().numpy()

class Critic(nn.Module):
    def __init__(
        self,
        input_dim : int, 
        mlp_dim : int, 
        n_actions:int, 
        x_norm:float = 1.0, 
        v_norm:float = 10.0
        ):
        super().__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.n_actions = n_actions

        self.x_norm = x_norm
        self.v_norm = v_norm

        self.encode = ParticleEncoder(mlp_dim, mlp_dim)

        self.fc1 = nn.Linear(mlp_dim + n_actions, mlp_dim)
        self.norm1 = nn.LayerNorm(mlp_dim)

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

        # Normalization
        x_pos = x[:,:self.input_dim//2] / self.x_norm
        x_vel = x[:,self.input_dim//2:] / self.v_norm
        z = torch.cat([x_pos, x_vel], dim=1)

        z = self.encode(z)
        z = torch.cat([z, a], dim=1)
        z = F.relu(self.norm1(self.fc1(z)))
        z = F.relu(self.norm2(self.fc2(z)))
        z = F.relu(self.norm3(self.fc3(z)))
        q = self.fc_out(z)
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
    q1_network: Critic,
    q2_network: Critic,
    p_network: Actor,
    target_q1_network: Critic,
    target_q2_network: Critic,
    target_p_network: Actor,
    q1_optimizer: torch.optim.Optimizer,
    q2_optimizer: torch.optim.Optimizer,
    p_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    batch_size: int = 32,
    gamma: float = 0.99,
    tau: float = 1e-2,
    device: Optional[str] = "cpu",
):

    q1_network.train()
    q2_network.train()
    p_network.train()

    target_q1_network.eval()
    target_q2_network.eval()
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
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)  
    next_state_batch = torch.cat(batch.next_state).float().to(device)
    
    action_bc_batch = torch.cat(batch.action_bc).float().to(device)

    with torch.no_grad():
        next_action = target_p_network.sample(next_state_batch)
        
        noise = (torch.randn_like(next_action) * 0.1).clamp(-0.1, 0.1)
        next_action = next_action + noise
        next_action = next_action.clamp(p_network.output_min, p_network.output_max)

        target_q1 = target_q1_network(next_state_batch, next_action)
        target_q2 = target_q2_network(next_state_batch, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward_batch.unsqueeze(1) + gamma * (1 - done_batch) * target_q

    # Update Q network
    q1 = q1_network(state_batch, action_batch)
    q1_loss = criterion(q1, target_q)
    q1_optimizer.zero_grad()
    q1_loss.backward()
    torch.nn.utils.clip_grad_norm_(q1_network.parameters(), max_norm=1.0)
    q1_optimizer.step()

    q2 = q2_network(state_batch, action_batch)
    q2_loss = criterion(q2, target_q)
    q2_optimizer.zero_grad()
    q2_loss.backward()
    torch.nn.utils.clip_grad_norm_(q2_network.parameters(), max_norm=1.0)
    q2_optimizer.step()

    # Update policy network
    action_batch_sampled = p_network.sample(state_batch)
    
    bc_loss = torch.mean(((action_bc_batch - action_batch_sampled) ** 2).sum(dim = 1) * reward_batch)
    p_loss = (-1) * q1_network(state_batch, action_batch_sampled).mean() + 1.0 * bc_loss

    p_optimizer.zero_grad()
    p_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(p_network.parameters(), max_norm=1.0)

    p_optimizer.step()

    # Update target newtorks
    soft_update(target_q1_network, q1_network, tau)
    soft_update(target_q2_network, q2_network, tau)
    soft_update(target_p_network, p_network, tau)

    return q1_loss, q2_loss, p_loss

def train(
    env: PIC,
    actuator: E_field,
    memory: ReplayBuffer,
    q1_network: Critic,
    q2_network: Critic,
    p_network: Actor,
    target_q1_network: Critic,
    target_q2_network: Critic,
    target_p_network: Actor,
    q1_optimizer: torch.optim.Optimizer,
    q2_optimizer: torch.optim.Optimizer,
    p_optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    batch_size: int = 10,
    tau: float = 1e-2,
    gamma: float = 0.99,
    device: Optional[str] = "cpu",
    num_episode: int = 10000,
    update_freq: int = 8,
    Nt: int = 1000,
    verbose: int = 8,
    save_last: Optional[str] = None,
    save_best: Optional[str] = None,
    alpha: float = 0.1,
    beta: float = 0.1,
    noise_scale: float = 0.1,
    mu: float = 0.0,
    theta: float = 0.15,
    sigma: float = 0.2,
):

    # minimum buffer size to start training
    min_buffer_size = 10000

    if device is None:
        device = "cpu"

    # Reward class
    reward_cls = Reward(env.init_dist.get_init_state(), env.N_mesh, env.L, -25.0, 25.0, env.n0, alpha, beta)

    # Trajectory
    q1_loss_traj = []
    q2_loss_traj = []
    p_loss_traj = []
    reward_traj = []

    best_reward = None

    # Ornstein-Uhlenbeck noise
    ou_noise = OrnsteinUhlenbeckNoise(p_network.n_actions, mu, theta, sigma)

    # Reward-weighted regression for offline training of policy network
    max_mode = p_network.n_actions // 2

    states = []
    actions = []
    rewards = []

    for idx_t in tqdm(range(Nt), "# Offline DDPG reward-weighted regression..."):

        state = env.get_state()
        states.append(state.ravel())

        _, Eks = compute_E_k_spectrum(1.0, 50.0, 50.0 / 250, 250, state, False)
        Eks = Eks[1:max_mode + 1,:]
        actuator.update_E((-1) * np.real(Eks), (+1) * np.imag(Eks))

        action = np.concatenate([actuator.coeff_cos.ravel(), actuator.coeff_sin.ravel()])

        # update state
        env.update_state(E_external=actuator.compute_E())

        actions.append(action)

        # compute reward
        reward = reward_cls.compute_reward(state, action)   
        reward = torch.tensor([reward])
        rewards.append(reward)

    states_bc = torch.cat([torch.tensor(state, dtype=torch.float32).unsqueeze(0) for state in states], dim = 0).to(device)
    actions_bc = torch.cat([torch.tensor(action, dtype=torch.float32).unsqueeze(0) for action in actions], dim = 0).to(device)
    rewards = torch.cat(rewards).to(device)

    # Behavior cloning
    n_bc_epoch = 50
    for epoch in range(n_bc_epoch):
        actions_pred = p_network.sample(states_bc)
        noise = (torch.randn_like(actions_pred) * 0.1).clamp(-0.1, 0.1)
        actions_pred = actions_pred + noise
        actions_pred = actions_pred.clamp(p_network.output_min, p_network.output_max)

        l2 = ((actions_bc - actions_pred) ** 2).sum(dim = 1)
        weighted_l2 = torch.sum(l2 * rewards).sum()
        
        p_optimizer.zero_grad()
        weighted_l2.backward()
        p_optimizer.step()
        
        if epoch % 10 == 0:
            print("| epoch:{} | bc error:{:.4f} | action dev:{:.4f}".format(epoch+1, l2.mean().detach().cpu().item(), (actions_pred - noise).detach().cpu().std().item()))

    # Online stage
    for i_episode in tqdm(range(num_episode), desc = '# Training DDPG controller...'):

        # initialize the simulation and actuator setup
        env.reinit()
        actuator.reinit()
        reward_cls.reinit()
        ou_noise.reset()

        reward_list = []
        q1_loss_list = []
        q2_loss_list = []
        p_loss_list = []

        for idx_t in range(Nt):

            p_network.eval()

            state = env.get_state()
            state_tensor = torch.tensor(state.ravel(), dtype=torch.float32).unsqueeze(0)

            # Behavior cloning
            _, Eks = compute_E_k_spectrum(1.0, 50.0, 50.0 / 250, 250, state, False)
            Eks = Eks[1:max_mode + 1,:]
            action_bc = np.concatenate([(-1) * np.real(Eks).ravel(), (+1) * np.imag(Eks).ravel()])
            action_bc_tensor = torch.tensor(action_bc, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_tensor = p_network.sample(state_tensor.to(device)).cpu()
                action = action_tensor.detach().squeeze(0).cpu().numpy()

            # Add exploration noise
            noise = ou_noise.sample() * noise_scale
            action = np.clip(action + noise, p_network.output_min, p_network.output_max)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

            # update actuator
            # Method 01. Sine and Cos input
            actuator.update_E(coeff_cos = action[:p_network.n_actions//2], coeff_sin = action[p_network.n_actions//2:])

            # update state
            env.update_state(E_external=actuator.compute_E())

            # get new state
            next_state = env.get_state()
            next_state_tensor = torch.tensor(next_state.ravel(), dtype=torch.float32).unsqueeze(0)

            # compute cost
            reward = reward_cls.compute_reward(state, action)           
            reward_tensor = torch.tensor([reward])

            # save trajectory into memory
            done = False if idx_t < Nt - 1 else True
            memory.push(
                state_tensor, 
                action_tensor, 
                next_state_tensor, 
                reward_tensor, 
                done,
                action_bc_tensor,
            )

            # update policy
            if memory.__len__() >= min_buffer_size and idx_t % update_freq == 0:

                q1_loss, q2_loss, p_loss = update_policy(
                    memory,
                    q1_network,
                    q2_network,
                    p_network,
                    target_q1_network,
                    target_q2_network,
                    target_p_network,
                    q1_optimizer,
                    q2_optimizer,
                    p_optimizer,
                    criterion,
                    batch_size,
                    gamma,
                    tau,
                    device,
                )

                reward_list.append(reward.item())
                q1_loss_list.append(q1_loss.detach().cpu().numpy().item())
                q2_loss_list.append(q2_loss.detach().cpu().numpy().item())
                p_loss_list.append(p_loss.detach().cpu().numpy().item())

                # save the weight parameters
                torch.save(p_network.state_dict(), save_last)

            if done and idx_t < Nt - 1:
                print("| episode:{} | simulation terminated with progress: {:.1f} percent".format(i_episode+1, 100 * (idx_t + 1)/Nt))
                break

        if i_episode % verbose == 0 and len(q1_loss_list) > 0:

            with torch.no_grad():
                actions_pred = p_network.sample(states_bc)

            action_dev = (actions_bc - actions_pred).std().item()

            print("| episode:{} | p loss:{:.4f} | q1 loss:{:.4f} | q2 loss{:.4f} | reward:{:.4f} | action dev:{:.4f}".format(i_episode+1, p_loss_list[-1], q1_loss_list[-1], q2_loss_list[-1], reward, action_dev))

        if len(q1_loss_list) > 0:
            reward_mean = np.mean(reward_list)
            q1_loss_mean = np.mean(q1_loss_list)
            q2_loss_mean = np.mean(q2_loss_list)
            p_loss_mean = np.mean(p_loss_list)

            reward_traj.append(reward_mean)
            q1_loss_traj.append(q1_loss_mean)
            q2_loss_traj.append(q2_loss_mean)
            p_loss_traj.append(p_loss_mean)

            if best_reward is not None:
                if reward_mean > best_reward:
                    torch.save(p_network.state_dict(), save_best)
            else:
                best_reward = reward_mean
                torch.save(p_network.state_dict(), save_best)

        # memory cache delete
        gc.collect()

    print("# Training DDPG controller process complete")

    reward = np.array(reward_traj)
    q1_loss = np.array(q1_loss_traj)
    q2_loss = np.array(q2_loss_traj)
    p_loss = np.array(p_loss_traj)

    return reward, q1_loss, q2_loss, p_loss
