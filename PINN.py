import numpy as np
import matplotlib.pyplot as plt
import os

H = 32
N_PARAMS = H + H + H * 3 + 3

def unpack_params(params):
    w1 = params[0:H].reshape(1, H)
    b1 = params[H:2*H]
    w2 = params[2*H:5*H].reshape(H, 3)
    b2 = params[5*H:5*H+3]
    return w1, b1, w2, b2

def forward_pass(t, params, r0, v0):
    w1, b1, w2, b2 = unpack_params(params)
    
    z = t @ w1 + b1
    h = np.tanh(z)
    N = h @ w2 + b2
    
    dh_dt = (1.0 - h**2) * w1
    dN_dt = dh_dt @ w2
    
    d2h_dt2 = -2.0 * h * (1.0 - h**2) * (w1**2)
    d2N_dt2 = d2h_dt2 @ w2
    
    r = r0 + v0 * t + (t**2) * N
    v = v0 + 2.0 * t * N + (t**2) * dN_dt
    a = 2.0 * N + 4.0 * t * dN_dt + (t**2) * d2N_dt2
    
    return r, v, a

def compute_loss(params, t, r0, v0, E_func, B_func):
    r, v, a = forward_pass(t, params, r0, v0)
    E = E_func(r, t)
    B = B_func(r, t)
    a_target = E + np.cross(v, B)
    res = a - a_target
    return np.mean(res**2)

def train_pinn(t, r0, v0, E_func, B_func, epochs=3000, lr=0.01):
    np.random.seed(42)
    params = np.random.randn(N_PARAMS) * 0.1
    m_opt = np.zeros_like(params)
    v_opt = np.zeros_like(params)
    
    eps_fd = 1e-5
    eps_adam = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    
    for ep in range(epochs):
        base_loss = compute_loss(params, t, r0, v0, E_func, B_func)
        grad = np.zeros_like(params)
        
        for i in range(N_PARAMS):
            params[i] += eps_fd
            l_plus = compute_loss(params, t, r0, v0, E_func, B_func)
            params[i] -= eps_fd
            grad[i] = (l_plus - base_loss) / eps_fd
            
        m_opt = beta1 * m_opt + (1.0 - beta1) * grad
        v_opt = beta2 * v_opt + (1.0 - beta2) * (grad**2)
        m_hat = m_opt / (1.0 - beta1**(ep+1))
        v_hat = v_opt / (1.0 - beta2**(ep+1))
        
        params -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        
        if ep % 500 == 0:
            print(f"Epoch {ep} | Loss: {base_loss:.6f}")
            
    return params

def e_field_case1(r, t):
    return np.zeros_like(r)

def b_field_case1(r, t):
    B = np.zeros_like(r)
    B[:, 2] = 1.0
    return B

def e_field_case2(r, t):
    E = np.zeros_like(r)
    E[:, 1] = 0.5
    return E

def b_field_case2(r, t):
    B = np.zeros_like(r)
    B[:, 2] = 1.0
    return B

def e_field_case3(r, t):
    E = np.zeros_like(r)
    E[:, 2] = 0.2
    return E

def b_field_case3(r, t):
    B = np.zeros_like(r)
    B[:, 2] = 1.0
    return B

def run_simulations_and_plot():
    out_dir = "./output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    t = np.linspace(0, 10, 100).reshape(-1, 1)
    
    print("Training Case 1: Cyclotron Motion")
    r0_1 = np.array([1.0, 0.0, 0.0])
    v0_1 = np.array([0.0, 1.0, 0.0])
    params1 = train_pinn(t, r0_1, v0_1, e_field_case1, b_field_case1)
    r_pred1, _, _ = forward_pass(t, params1, r0_1, v0_1)
    
    r_true1 = np.zeros_like(r_pred1)
    r_true1[:, 0] = np.cos(t.flatten())
    r_true1[:, 1] = np.sin(t.flatten())
    r_true1[:, 2] = 0.0
    
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(r_true1[:,0], r_true1[:,1], r_true1[:,2], 'k--', linewidth=2, label='Analytical')
    ax1.plot(r_pred1[:,0], r_pred1[:,1], r_pred1[:,2], 'r-', alpha=0.7, label='PINN')
    ax1.set_title('Case 1: Cyclotron Motion')
    ax1.legend()
    plt.savefig(f'{out_dir}/case1_cyclotron.png')
    plt.close()
    
    print("Training Case 2: ExB Drift")
    r0_2 = np.array([0.0, 0.0, 0.0])
    v0_2 = np.array([0.0, 0.0, 0.0])
    params2 = train_pinn(t, r0_2, v0_2, e_field_case2, b_field_case2)
    r_pred2, _, _ = forward_pass(t, params2, r0_2, v0_2)
    
    r_true2 = np.zeros_like(r_pred2)
    r_true2[:, 0] = 0.5 * (t.flatten() - np.sin(t.flatten()))
    r_true2[:, 1] = 0.5 * (1.0 - np.cos(t.flatten()))
    r_true2[:, 2] = 0.0
    
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(r_true2[:,0], r_true2[:,1], r_true2[:,2], 'k--', linewidth=2, label='Analytical')
    ax2.plot(r_pred2[:,0], r_pred2[:,1], r_pred2[:,2], 'b-', alpha=0.7, label='PINN')
    ax2.set_title('Case 2: ExB Drift')
    ax2.legend()
    plt.savefig(f'{out_dir}/case2_drift.png')
    plt.close()
    
    print("Training Case 3: Parallel E and B Fields")
    r0_3 = np.array([1.0, 0.0, 0.0])
    v0_3 = np.array([0.0, 1.0, 0.0])
    params3 = train_pinn(t, r0_3, v0_3, e_field_case3, b_field_case3)
    r_pred3, _, _ = forward_pass(t, params3, r0_3, v0_3)
    
    r_true3 = np.zeros_like(r_pred3)
    r_true3[:, 0] = np.cos(t.flatten())
    r_true3[:, 1] = np.sin(t.flatten())
    r_true3[:, 2] = 0.1 * (t.flatten()**2)
    
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot(r_true3[:,0], r_true3[:,1], r_true3[:,2], 'k--', linewidth=2, label='Analytical')
    ax3.plot(r_pred3[:,0], r_pred3[:,1], r_pred3[:,2], 'g-', alpha=0.7, label='PINN')
    ax3.set_title('Case 3: Accelerating Cyclotron')
    ax3.legend()
    plt.savefig(f'{out_dir}/case3_parallel_fields.png')
    plt.close()

if __name__ == "__main__":
    run_simulations_and_plot()