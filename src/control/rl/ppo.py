import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Optional
from torch.distributions import Normal
import numpy as np
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
    ('state','action','next_state','reward','done','prob_a')
)

# Function for initialization
def hidden_init(layer: torch.nn.Linear):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
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
        
        self.encode = ParticleEncoder(mlp_dim, mlp_dim)
        
        self.fc1 = nn.Linear(mlp_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(mlp_dim)

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

        z = self.encode(x)
        z = F.relu(self.norm1(self.fc1(z)))
        z = F.relu(self.norm2(self.fc2(z)))
        z = F.relu(self.norm3(self.fc3(z)))

        mu = F.tanh(self.fc_pi(z))
        std = self.log_std.exp().expand_as(mu).to(x.device)
        value = self.fc_v(z)

        return mu, std, value

    def sample(self, x : torch.Tensor, deterministic:bool = False):
        mu, std, value = self.forward(x)
        dist = Normal(mu, std)
        
        if deterministic:
            y = mu
        else:
            y = dist.rsample()
                    
        # Rescale action space with bounded region
        action = (0.5 + 0.5 * y) * (self.output_max - self.output_min) + self.output_min

        log_probs = dist.log_prob(y)
        log_probs = log_probs.sum(dim = -1, keepdim=True)
        entropy = dist.entropy().mean()

        return action, entropy, log_probs, value

    def get_action(self, x:np.ndarray)->np.ndarray:
        
        x = x.ravel()
        state = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _, _ = self.sample(state, True)
            
        return action.squeeze(0).cpu().numpy()

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
    ):

    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(T)):
        next_value = next_values[t]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns

# update policy
def update_policy(
    memory : ReplayBuffer, 
    policy_network : ActorCritic, 
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    gamma : float = 0.99, 
    lam:float = 0.95,
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

    next_state_batch = torch.cat(batch.next_state).float().to(device)
    state_batch = torch.cat(batch.state).float().to(device)
    prob_a_batch = torch.cat(batch.prob_a).float().to(device)
    reward_batch = torch.cat(batch.reward).float().to(device) 
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device)

    with torch.no_grad():        
        _, _, values = policy_network(state_batch)
        values = values.squeeze(-1)

        _, _, next_values = policy_network(next_state_batch)
        next_values = next_values.squeeze(-1)

    # GAE
    advantages, returns = compute_gae(reward_batch, values, done_batch, next_values, gamma, lam)
    td_target = advantages.unsqueeze(-1)

    # Normalized At
    # td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-6)

    loss_list = []

    for _ in range(k_epoch):

        policy_optimizer.zero_grad()
        _, entropy, log_probs, value = policy_network.sample(state_batch)

        delta = td_target - value     
        ratio = torch.exp(log_probs - prob_a_batch.detach())

        surr1 = ratio * delta
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * delta

        p_loss = -torch.min(surr1, surr2).mean()
        q_loss = value_coeff * criterion(value, returns.view_as(value))
        e_loss = -entropy_coeff * entropy
        loss = p_loss + q_loss + e_loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=0.5)

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
    lam:float = 0.95,
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
    
    # Reward-weighted regression for offline training of policy network
    max_mode = policy_network.n_actions // 2

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

    states = torch.cat([torch.tensor(state, dtype=torch.float32).unsqueeze(0) for state in states], dim = 0).to(device)
    actions = torch.cat([torch.tensor(action, dtype=torch.float32).unsqueeze(0) for action in actions], dim = 0).to(device)
    rewards = torch.cat(rewards).to(device)

    actions_pred, _, _, _ = policy_network.sample(states)

    l2 = torch.sum(((actions - actions_pred) ** 2), dim = 1)
    l2_action = torch.sum(l2) * (-1)
    
    policy_optimizer.zero_grad()
    l2_action.backward()
    policy_optimizer.step()
    
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
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_tensor, _, log_probs, _ = policy_network.sample(state_tensor.to(device))
                
            action = action_tensor.detach().squeeze(0).cpu().numpy()

            # update actuator
            actuator.update_E(coeff_cos = action[:policy_network.n_actions//2], coeff_sin = action[policy_network.n_actions//2:])

            # update state
            env.update_state(E_external=actuator.compute_E())

            # get new state
            next_state = env.get_state().ravel()
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

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
                    lam,
                    eps_clip,
                    entropy_coeff,
                    value_coeff,
                    device,
                    k_epoch
                )

                reward_list.append(reward.item())
                loss_list.append(policy_loss.item())

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
