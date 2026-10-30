import os
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np

# Hardware acceleration toggle
try:
    import cupy as xp
    GPU_ENABLED = True
    print("GPU Acceleration ENABLED (CuPy loaded)")
except ImportError:
    import numpy as xp
    GPU_ENABLED = False
    print("GPU Acceleration DISABLED. Using CPU (NumPy)")

def to_numpy(arr):
    if GPU_ENABLED:
        return arr.get()
    return arr

def solve_pi_elm(t, r0, v0, E_val, B_val, H_nodes=2000):
    N = t.shape[0]
    t_sq = t**2
    t_2 = 2.0 * t
    t_4 = 4.0 * t

    # Randomly initialize locked hidden layer to map high-frequency sine basis
    xp.random.seed(42)
    W1 = xp.random.randn(1, H_nodes) * 2.0
    b1 = xp.random.uniform(-10, 10, (1, H_nodes))

    z = t @ W1 + b1
    H = xp.sin(z)
    H_prime = W1 * xp.cos(z)
    H_dprime = -(W1**2) * xp.sin(z)

    # Augment basis with bias parameter column
    ones = xp.ones((N, 1))
    zeros = xp.zeros((N, 1))
    H_aug = xp.hstack((H, ones))
    H_prime_aug = xp.hstack((H_prime, zeros))
    H_dprime_aug = xp.hstack((H_dprime, zeros))

    # Analytical constraint matrices based on Ansatz derivatives
    Phi_v = t_2 * H_aug + t_sq * H_prime_aug
    Phi_a = 2.0 * H_aug + t_4 * H_prime_aug + t_sq * H_dprime_aug

    Ex = E_val[:, 0:1]
    Ey = E_val[:, 1:2]
    Ez = E_val[:, 2:3]
    Bx = B_val[:, 0:1]
    By = B_val[:, 1:2]
    Bz = B_val[:, 2:3]

    # Construct the sparse block matrix A for Lorentz Force constraints
    A11 = Phi_a
    A12 = -Bz * Phi_v
    A13 = By * Phi_v

    A21 = Bz * Phi_v
    A22 = Phi_a
    A23 = -Bx * Phi_v

    A31 = -By * Phi_v
    A32 = Bx * Phi_v
    A33 = Phi_a

    A_row1 = xp.hstack((A11, A12, A13))
    A_row2 = xp.hstack((A21, A22, A23))
    A_row3 = xp.hstack((A31, A32, A33))
    A = xp.vstack((A_row1, A_row2, A_row3))

    # Construct RHS target matrix Y
    Y1 = Ex + Bz * v0[1] - By * v0[2]
    Y2 = Ey + Bx * v0[2] - Bz * v0[0]
    Y3 = Ez + By * v0[0] - Bx * v0[1]
    Y = xp.vstack((Y1, Y2, Y3))

    # Solve the massive linear system for exact global minimum
    W2_flat, _, _, _ = xp.linalg.lstsq(A, Y, rcond=None)

    H_dim = H_nodes + 1
    wx = W2_flat[0:H_dim]
    wy = W2_flat[H_dim:2*H_dim]
    wz = W2_flat[2*H_dim:3*H_dim]
    W2 = xp.hstack((wx, wy, wz))

    # Predict final physically guaranteed trajectory
    N_pred = H_aug @ W2
    r_pred = r0.reshape(1, 3) + v0.reshape(1, 3) * t + t_sq * N_pred
    
    return to_numpy(r_pred)

def simulate_case_1(t):
    print("Solving Case 1: Cyclotron Motion")
    r0 = xp.array([0.0, 1.0, 0.0])
    v0 = xp.array([1.0, 0.0, 0.0])
    
    E_val = xp.zeros((t.shape[0], 3))
    B_val = xp.zeros((t.shape[0], 3))
    B_val[:, 2] = 1.0
    
    r_pred = solve_pi_elm(t, r0, v0, E_val, B_val)
    
    t_flat = to_numpy(t).flatten()
    r_true = np.zeros_like(r_pred)
    r_true[:, 0] = np.sin(t_flat)
    r_true[:, 1] = np.cos(t_flat)
    r_true[:, 2] = 0.0
    
    return r_pred, r_true

def simulate_case_2(t):
    print("Solving Case 2: ExB Drift")
    r0 = xp.array([0.0, 0.0, 0.0])
    v0 = xp.array([0.0, 0.0, 0.0])
    
    E_val = xp.zeros((t.shape[0], 3))
    E_val[:, 1] = 0.5
    B_val = xp.zeros((t.shape[0], 3))
    B_val[:, 2] = 1.0
    
    r_pred = solve_pi_elm(t, r0, v0, E_val, B_val)
    
    t_flat = to_numpy(t).flatten()
    r_true = np.zeros_like(r_pred)
    r_true[:, 0] = 0.5 * (t_flat - np.sin(t_flat))
    r_true[:, 1] = 0.5 * (1.0 - np.cos(t_flat))
    r_true[:, 2] = 0.0
    
    return r_pred, r_true

def simulate_case_3(t):
    print("Solving Case 3: Parallel Fields")
    r0 = xp.array([0.0, 1.0, 0.0])
    v0 = xp.array([1.0, 0.0, 0.0])
    
    E_val = xp.zeros((t.shape[0], 3))
    E_val[:, 2] = 0.2
    B_val = xp.zeros((t.shape[0], 3))
    B_val[:, 2] = 1.0
    
    r_pred = solve_pi_elm(t, r0, v0, E_val, B_val)
    
    t_flat = to_numpy(t).flatten()
    r_true = np.zeros_like(r_pred)
    r_true[:, 0] = np.sin(t_flat)
    r_true[:, 1] = np.cos(t_flat)
    r_true[:, 2] = 0.1 * (t_flat**2)
    
    return r_pred, r_true

def plot_case(r_pred, r_true, title, filename, out_dir):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_true[:,0], r_true[:,1], r_true[:,2], 'k--', linewidth=2, label='Analytical Target')
    ax.plot(r_pred[:,0], r_pred[:,1], r_pred[:,2], 'r-', alpha=0.7, label='PI-ELM Solution')
    ax.set_title(title)
    ax.legend()
    plt.savefig(f'{out_dir}/{filename}')
    plt.close()
    print(f"Saved exact mapping plot: {filename}")

def execute_pipeline():
    out_dir = "../output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # High resolution time domain needed for visually smooth plotting 
    t_base = np.linspace(0, 10, 500).reshape(-1, 1)
    t = xp.array(t_base)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future1 = executor.submit(simulate_case_1, t)
        future2 = executor.submit(simulate_case_2, t)
        future3 = executor.submit(simulate_case_3, t)
        
        pred1, true1 = future1.result()
        pred2, true2 = future2.result()
        pred3, true3 = future3.result()

    plot_case(pred1, true1, 'Case 1: Cyclotron Motion', 'case1_cyclotron.png', out_dir)
    plot_case(pred2, true2, 'Case 2: ExB Drift', 'case2_drift.png', out_dir)
    plot_case(pred3, true3, 'Case 3: Accelerating Cyclotron', 'case3_parallel_fields.png', out_dir)

if __name__ == "__main__":
    execute_pipeline()