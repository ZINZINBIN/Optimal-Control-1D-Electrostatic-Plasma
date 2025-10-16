import numpy as np
import os, argparse, torch
from tqdm.auto import tqdm
from scipy.io import savemat
from src.env.pic import PIC
from src.env.dist import BumpOnTail, TwoStream
from src.control.actuator import E_field
from src.control.rl.ppo import ActorCritic, train, ReplayBuffer
from src.control.rl.reward import Reward
from src.plot import (
    plot_two_stream_evolution, 
    plot_bump_on_tail_evolution, 
    plot_log_E, 
    plot_E_k_spectrum,
    plot_E_k_over_time, 
    plot_cost_over_time, 
    plot_E_k_external_over_time
)

def parsing():
    parser = argparse.ArgumentParser(description="Optimization of RL for optimal control in Vlasov-Poisson plasma kinetic system")

    # Simulation setting
    parser.add_argument("--simcase", type = str, default = "two-stream", choices=["two-stream", "bump-on-tail"])
    parser.add_argument("--interpol", type=str, default = "CIC", choices=["CIC", "TSC"])
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--save_plot", type=str, default="./result/")
    parser.add_argument("--save_file", type=str, default="./dataset/")

    # PIC parameters (default)
    parser.add_argument("--num_particle", type = int, default = 10000)  
    parser.add_argument("--num_mesh", type = int, default = 500)        
    parser.add_argument("--t_min", type = float, default = 0)
    parser.add_argument("--t_max", type = float, default = 50)
    parser.add_argument("--dt", type = float, default = 0.05)          

    # Physical length scale and initial density
    parser.add_argument("--L", type = float, default = 50)
    parser.add_argument("--n0", type = float, default = 1.0)

    # Beam velocity (Two-stream) or high energy electron velocity (Bump-on-tail)
    parser.add_argument("--vb", type = float, default = 3.0)

    # Thermal velocity (both cases)
    parser.add_argument("--vth", type = float, default = 1.0)

    # Initial perturbation parameters (both cases)
    parser.add_argument("--A", type = float, default = 0.1)
    parser.add_argument("--n_mode", type = int, default = 3)

    # Distribution parameters (Bump-on-tail)
    parser.add_argument("--a", type = float, default = 0.2)   

    # Controller
    parser.add_argument("--max_mode", type = int, default = 5)
    parser.add_argument("--coeff_max", type=float, default= 1.0)
    parser.add_argument("--coeff_min", type=float, default= -1.0)

    # Network
    parser.add_argument("--mlp_dim", type = int, default = 32)
    parser.add_argument("--r", type =float, default = 0.995)
    parser.add_argument("--std", type = float, default = 0.5)
    parser.add_argument("--capacity", type=int, default=10)
    parser.add_argument("--eps_clip", type=float, default=0.25)
    parser.add_argument("--entropy_coeff", type=float, default=0.10)
    parser.add_argument("--value_coeff", type=float, default=0.25)
    parser.add_argument("--num_episode", type=int, default=1000)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--k_epoch", type=int, default=4)

    # Cost parameters
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--save_last", type=str, default="ppo_last.pt")
    parser.add_argument("--save_best", type=str, default="ppo_best.pt")
    
    # Torch device
    parser.add_argument("--gpu_num", type=int, default=0)
    
    # Setup
    parser.add_argument("--optimize", type=bool, default=False)

    args = vars(parser.parse_args())
    return args

# torch device state
print("=============== Device setup ===============")
print("torch cuda avaliable : ", torch.cuda.is_available())
print("torch current gpu : ", torch.cuda.current_device())
print("torch available gpus : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":

    args = parsing()

    savepath = os.path.join(args["save_plot"], args['simcase'], "ppo-control")
    filepath = os.path.join(args['save_file'], args['simcase'], "ppo-control")
    
    # Information
    print("=============== Information ================")
    print("Simulation : {}".format(args['simcase']))
    print("RL algorithm : PPO")
    print("Input mode number : {}".format(args['max_mode']))
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda"
    else:
        device = 'cpu'

    # Directory check
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if args['simcase'] == "two-stream":
        dist = TwoStream(v0 = args['vb'], sigma = args['vth'], n_samples=args['num_particle'], L = args['L'])

    elif args['simcase'] == "bump-on-tail":
        dist = BumpOnTail(a = args['a'], v0 = args['vb'], sigma = args['vth'], n_samples=args['num_particle'], L = args['L'])

    # PIC simulation code
    sim = PIC(
        N=args["num_particle"],
        N_mesh=args["num_mesh"],
        n0=args["n0"],
        L=args["L"],
        dt=args["dt"],
        tmin=args["t_min"],
        tmax=args["t_max"],
        gamma=args["gamma"],
        A=args["A"],
        n_mode=args["n_mode"],
        interpol=args["interpol"],
        init_dist=dist,
    )

    # Actuator
    actuator = E_field(args['L'], args['num_mesh'], args['max_mode'])

    # Controller
    input_dim = args['num_particle'] * 2
    n_actions = 2 * args['max_mode']
    network = ActorCritic(input_dim, args['mlp_dim'], n_actions, args['std'], output_min = args['coeff_min'], output_max = args['coeff_max'])
    network.to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr = args['lr'])

    # maximum simulation time (integer)
    Nt = int(np.ceil((args['t_max'] - args['t_min']) / args['dt']))

    # Optimize controller
    memory = ReplayBuffer(args['capacity'])
    
    if args['optimize']:
        
        reward, loss = train(
            sim, 
            actuator, 
            memory, 
            network, 
            optimizer, 
            None, 
            args['r'], 
            args['eps_clip'], 
            args['entropy_coeff'],
            args['value_coeff'],
            device, 
            args['num_episode'], 
            Nt, 
            args['verbose'], 
            os.path.join(filepath, args['save_last']),
            os.path.join(filepath, args['save_best']),
            args['alpha'],
            args['beta'],
            args['k_epoch']
        )
        
        # save optimization process
        mdic = {
            'reward':reward,
            'loss':loss
        }

        # save data
        savemat(file_name = os.path.join(filepath, "process.mat"), mdict=mdic, do_compression=True)

    # Trajectory of the system's state
    pos_list = []
    vel_list = []
    E_list = []
    PE_list = []

    # Trajectory of the input control
    coeff_cos = []
    coeff_sin = []
    
    # initialize the simulation
    sim.reinit()
    
    # load best model
    network.load_state_dict(torch.load(os.path.join(filepath, args['save_best'])))
    network.cpu()
    
    # no gradient
    network.eval()
    
    # Compute the cost function
    reward = Reward(sim.init_dist.get_init_state(), args['num_mesh'], args['L'], -25.0, 25.0, args['n0'], 1.0, 1.0)
    
    cost_kl_list = []
    cost_ee_list = []
    cost_ie_list = []

    for idx_t in tqdm(range(Nt), "PIC simulation with E-field control"):
        
        # Update coefficients
        state = sim.get_state()
        coeffs = network.get_action(state)
        actuator.update_E(coeffs[:args['max_mode']], coeffs[args['max_mode']:])

        # Get action
        E_external = actuator.compute_E()

        # save current state
        E = sim.get_energy()
        PE = sim.get_electric_energy()

        pos_list.append(sim.x.copy())
        vel_list.append(sim.v.copy())
        E_list.append(E)
        PE_list.append(PE)

        coeff_cos.append(actuator.coeff_cos.copy())
        coeff_sin.append(actuator.coeff_sin.copy())
        
        # Update motion
        sim.update_state(E_external)
        
        # Compute code
        cost_kl = reward.compute_kl_divergence(sim.get_state())
        cost_ee = reward.compute_electric_energy(sim.get_state())
        cost_ie = reward.compute_input_energy(coeffs)
        
        cost_kl_list.append(cost_kl)
        cost_ee_list.append(cost_ee)
        cost_ie_list.append(cost_ie)

    qs = np.concatenate(pos_list, axis = 1)
    ps = np.concatenate(vel_list, axis = 1)
    snapshot = np.concatenate([qs, ps], axis=0)

    E = np.array(E_list)
    PE = np.array(PE_list)

    coeff_cos = np.concatenate(coeff_cos, axis = 1)
    coeff_sin = np.concatenate(coeff_sin, axis = 1)

    mdic = {
        "snapshot": snapshot,
        "E":E,
        "PE":PE,
        "N": args["num_particle"],
        "N_mesh": args["num_mesh"],
        "n0": args["n0"],
        "L": args["L"],
        "dt": args["dt"],
        "tmin": args["t_min"],
        "tmax": args["t_max"],
        "n_mode": args['n_mode'],
        "A": args['A'],
        "vth":args["vth"],
        "vb": args['vb'],
        "a": args['a'],
        "coeff_cos":coeff_cos,
        "coeff_sin":coeff_sin,
    }

    # save data
    savemat(file_name = os.path.join(filepath, "data.mat"), mdict=mdic, do_compression=True)
    
    # Plot cost function
    cost = {
        r"$J_{KL}$":cost_kl_list,
        r"$J_{ee}$":cost_ee_list,
        r"$J_{ie}$":cost_ie_list
    }
    
    plot_cost_over_time(args['t_max'], Nt, cost, savepath, "cost.pdf")
    
    # Plot the result
    if args['simcase'] == "two-stream":
        plot_two_stream_evolution(snapshot, savepath, "phase_space_evolution.pdf", 0, args['L'], -10.0, 10.0)
        
    elif args['simcase'] == "bump-on-tail":
        plot_bump_on_tail_evolution(snapshot, savepath, "phase_space_evolution.pdf", 0, args['L'], -10.0, 10.0)
    
    # Electric energy
    plot_log_E(args['t_max'], args['L'], args['L'] / args['num_mesh'], args['num_mesh'], snapshot, savepath, "log_E.pdf")
    
    # Electric field spectrum
    plot_E_k_spectrum(args['t_max'], args['L'], args['L'] / args['num_mesh'], args['num_mesh'], snapshot, savepath, "E_k_spectrum.pdf")
    
    # Fourier coefficient over time
    plot_E_k_over_time(args['t_max'], args['L'], args['L'] / args['num_mesh'], args['num_mesh'], args['max_mode'], snapshot, savepath, "Ek_t.pdf")

    # Amplitude of each external E field over time
    plot_E_k_external_over_time(args['t_max'], coeff_cos, coeff_sin, savepath, "Ek_t_external.pdf")