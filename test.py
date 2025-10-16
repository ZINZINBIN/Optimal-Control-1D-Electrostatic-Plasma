# import sys, os
# import numpy as np
# import matplotlib.pyplot as plt

from scipy.io import loadmat
import numpy as np
import os, argparse
from scipy.io import savemat
from tqdm.auto import tqdm
from src.env.pic import PIC
from src.env.dist import BumpOnTail, TwoStream
from src.control.actuator import E_field
from src.control.rl.reward import Reward
from src.env.dist import TwoStream
from src.env.util import compute_E

def parsing():
    parser = argparse.ArgumentParser(description="Vlasov-Poisson plasma kinetic simulation with an external electric field")

    # Simulation setting
    parser.add_argument("--simcase", type = str, default = "two-stream", choices=["two-stream", "bump-on-tail"])
    parser.add_argument("--interpol", type=str, default = "CIC", choices=["CIC", "TSC"])
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--save_file", type=str, default="./dataset/")
    parser.add_argument("--tag", type=str, default="test")

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

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = parsing()

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
    action = np.concatenate([actuator.coeff_cos, actuator.coeff_sin]).ravel()
    
    init_state = dist.get_init_state()
    reward = Reward(init_state)
    
    state = sim.get_state()
    
    print(reward.compute_kl_divergence(state))
    print(reward.compute_electric_energy(state))
    print(reward.compute_input_energy(action))
    
    print(reward.compute_reward_kl_divergence(state))
    print(reward.compute_reward_electric_energy(state))
    print(reward.compute_reward_input_energy(action))
    
    with open(os.path.join(args["save_file"], args['simcase'], "wo-oc", "data.mat"), "rb") as file:
        mdat = loadmat(file)
        
    snapshot = mdat["snapshot"]
    E = mdat["E"]
    PE = mdat["PE"]

    N = mdat["N"].item()
    Nt = snapshot.shape[1]
    N_mesh = mdat["N_mesh"].item()
    L = mdat["L"].item()
    dx = L / N_mesh
    tmin = mdat["tmin"][0]
    tmax = mdat["tmax"][0]
    dt = mdat["dt"][0].item()
    ts = np.linspace(tmin, tmax, Nt)
    
    print(reward.compute_reward_kl_divergence(snapshot[:,0]))
    print(reward.compute_reward_kl_divergence(snapshot[:,500]))
    print(reward.compute_reward_kl_divergence(snapshot[:,-1]))
    
    from src.control.objective import estimate_f, estimate_KL_divergence
    f0 = estimate_f(init_state.reshape(-1,1), N_mesh, L, -25.0, 25.0, 1.0)
    fi = estimate_f(snapshot[:,0].reshape(-1,1), N_mesh, L, -25.0, 25.0, 1.0)
    fm = estimate_f(snapshot[:,500].reshape(-1,1), N_mesh, L, -25.0, 25.0, 1.0)
    ff = estimate_f(snapshot[:,-1].reshape(-1,1), N_mesh, L, -25.0, 25.0, 1.0)
    
    print(estimate_KL_divergence(fi, f0, dx, 0.1))
    print(estimate_KL_divergence(fm, f0, dx, 0.1))
    print(estimate_KL_divergence(ff, f0, dx, 0.1))
    
    from src.interpret.spectrum import compute_E_k_spectrum
    ks, Eks = compute_E_k_spectrum(args['n0'], args['L'], args['L'] / args['num_mesh'], args['num_mesh'], snapshot, False)

    ks = ks[1:args['max_mode'] + 1]
    Eks = Eks[1:args['max_mode'] + 1,:]
    
    # Trajectory of the input control
    coeff_cos = [np.real(Eks[:,idx_t]).reshape(-1,1) for idx_t in range(Nt)]
    coeff_sin = [(-1) * np.imag(Eks[:,idx_t]).reshape(-1,1) for idx_t in range(Nt)]
    
    actuator.update_E(coeff_cos[-1], coeff_sin[-1])
    
    _, E_m = compute_E(snapshot[:,-1].reshape(-1,1), args['L'] / args['num_mesh'], args['num_mesh'], 1.0, args['L'], args['num_particle'])
    
    import matplotlib.pyplot as plt
    
    print(np.linalg.norm(actuator.compute_E() - E_m))
    
    plt.plot(actuator.xm, actuator.compute_E(), label = "IFFT")
    plt.plot(actuator.xm, E_m, label = "GT")
    plt.legend()
    plt.savefig("e_field.png")