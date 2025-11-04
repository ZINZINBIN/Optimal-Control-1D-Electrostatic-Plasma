import numpy as np
import os, argparse
from scipy.io import savemat
from tqdm.auto import tqdm
from src.env.pic import PIC
from src.env.dist import BumpOnTail, TwoStream
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
    parser = argparse.ArgumentParser(description="Vlasov-Poisson plasma kinetic simulation without E-field control")

    # Simulation setting
    parser.add_argument("--simcase", type = str, default = "two-stream", choices=["two-stream", "bump-on-tail"])
    parser.add_argument("--interpol", type=str, default = "CIC", choices=["CIC", "TSC"])
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--save_file", type=str, default="./dataset/")
    parser.add_argument("--save_plot", type=str, default="./result/")

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
    
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()

    filepath = os.path.join(args["save_file"], args['simcase'], "wo-oc")
    savepath = os.path.join(args["save_plot"], args['simcase'], "wo-oc")

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
    
    Nt = int(np.ceil((args['t_max'] - args['t_min']) / args['dt']))
    
    # Trajectory of the system's state
    pos_list = []
    vel_list = []
    E_list = []
    PE_list = []
    
    # Compute the cost function
    reward = Reward(sim.init_dist.get_init_state(), args['num_mesh'], args['L'], -25.0, 25.0, args['n0'], 1.0, 1.0)
    
    cost_kl_list = []
    cost_ee_list = []
        
    for idx_t in tqdm(range(Nt), "PIC simulation without E-field control"):

        # Update motion
        sim.update_state(None)
    
        E = sim.get_energy()
        PE = sim.get_electric_energy()

        pos_list.append(sim.x.copy())
        vel_list.append(sim.v.copy())
        E_list.append(E)
        PE_list.append(PE)
        
        cost_kl = reward.compute_kl_divergence(sim.get_state())
        cost_ee = reward.compute_electric_energy(sim.get_state())
        
        cost_kl_list.append(cost_kl)
        cost_ee_list.append(cost_ee)
        
    qs = np.concatenate(pos_list, axis = 1)
    ps = np.concatenate(vel_list, axis = 1)
    snapshot = np.concatenate([qs, ps], axis=0)

    E = np.array(E_list)
    PE = np.array(PE_list)
    
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
    }

    # save data
    savemat(file_name = os.path.join(filepath, "data.mat"), mdict=mdic, do_compression=True)
    
    # Plot cost function
    cost = {
        r"$J_{KL}$":cost_kl_list,
        r"$J_{ee}$":cost_ee_list,
    }
    plot_cost_over_time(args['t_max'], Nt, cost, savepath, "cost.pdf")
    
    # plot electric field
    # Electric energy over time
    plot_log_E(args['t_max'], args['L'], args['L'] / args['num_mesh'], args['num_mesh'], snapshot, savepath, "log_E.pdf")
    
    # Fourier spectrum of electric field 
    plot_E_k_spectrum(args['t_max'], args['L'], args['L'] / args['num_mesh'], args['num_mesh'], snapshot, savepath, "Ek_spectrum.pdf")
    
    # Fourier coefficient over time
    plot_E_k_over_time(args['t_max'], args['L'], args['L'] / args['num_mesh'], args['num_mesh'], 5, snapshot, savepath, "Ek_t.pdf")
    
    if args['simcase'] == "two-stream":
        plot_two_stream_evolution(snapshot, savepath, "phase_space_evolution.pdf", 0, args['L'], -10.0, 10.0)

    elif args['simcase'] == "bump-on-tail":
        plot_bump_on_tail_evolution(snapshot, savepath, "phase_space_evolution.pdf", 0, args['L'], -10.0, 10.0)