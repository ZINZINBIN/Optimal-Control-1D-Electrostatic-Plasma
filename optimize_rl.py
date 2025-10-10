import numpy as np
import os, argparse, torch
from scipy.io import savemat
from src.env.pic import PIC
from src.env.dist import BumpOnTail, TwoStream
from src.control.actuator import E_field
from src.control.rl.ppo import ActorCritic, train, ReplayBuffer

def parsing():
    parser = argparse.ArgumentParser(description="Optimization of RL for optimal control in Vlasov-Poisson plasma kinetic system")

    # Simulation setting
    parser.add_argument("--simcase", type = str, default = "two-stream", choices=["two-stream", "bump-on-tail"])
    parser.add_argument("--interpol", type=str, default = "CIC", choices=["CIC", "TSC"])
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--save_dir", type=str, default="./dataset/")
    parser.add_argument("--tag", type=str, default="ppo-control")

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
    parser.add_argument("--max_mode", type = int, default = 3)
    parser.add_argument("--coeff_max", type=float, default = 1.0)
    parser.add_argument("--coeff_min", type=float, default=-1.0)

    # Network
    parser.add_argument("--mlp_dim", type = int, defautl = 32)
    parser.add_argument("--std", type = float, default = 0.25)
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--eps_clip", type=float, default=0.25)
    parser.add_argument("--entropy_coeff", type=float, default=0.1)
    parser.add_argument("--num_episode", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_weight", type=str, default="./dataset/ppo.pt")

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()

    tag = args['tag']
    savepath = os.path.join(args["save_dir"], args['simcase'])

    # Directory check
    if not os.path.exists(savepath):
        os.makedirs(savepath)

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
    optimizer = torch.optim.RMSprop(network.parameters(), lr = args['lr'])

    Nt = int(np.ceil((args['t_max'] - args['t_min']) / args['dt']))
    
    # Optimize
    memory = ReplayBuffer(args['capacity'])
    
    reward, loss = train(
        sim, 
        actuator, 
        memory, 
        network, 
        optimizer, 
        None, 
        args['gamma'], 
        args['eps_clip'], 
        args['entropy_coeff'],
        "cpu", 
        args['num_episode'], 
        Nt, 
        args['verbose'], 
        args['save_weight']
    )
    
    # Trajectory of the system's state
    pos_list = []
    vel_list = []
    E_list = []
    PE_list = []
    
    # Trajectory of the input control
    coeff_cos = []
    coeff_sin = []
    
    for idx_t in range(Nt):
        
        # Update coefficients
        actuator.update_E()
        
        # Get action
        E_external = actuator.compute_E()

        # Update motion
        sim.update_state(E_external)
    
        E = sim.get_energy()
        PE = sim.get_electric_energy()

        pos_list.append(sim.x.copy())
        vel_list.append(sim.v.copy())
        E_list.append(E)
        PE_list.append(PE)
        
        coeff_cos.append(actuator.coeff_cos.copy())
        coeff_sin.append(actuator.coeff_sin.copy())
        
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
    if args["simcase"] == "two-stream":
        savemat(file_name = os.path.join(savepath, "{}.mat".format(tag)), mdict=mdic, do_compression=True)

    elif args["simcase"] == "bump-on-tail":
        savemat(file_name = os.path.join(savepath, "{}.mat".format(tag)), mdict=mdic, do_compression=True)