import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# [A] 기준 해 생성을 위한 RK4
def rk4_reference(a, b, x0, y0, T_end=25.0, h=0.05):
    def f(x, y): return x * (1 - x - a * y)
    def g(x, y): return y * (1 - y - b * x)
    num_steps = int(T_end / h)
    t_vals = np.linspace(0, T_end, num_steps)
    x_vals, y_vals = np.zeros(num_steps), np.zeros(num_steps)
    x_vals[0], y_vals[0] = x0, y0
    for i in range(1, num_steps):
        xn, yn = x_vals[i-1], y_vals[i-1]
        k1x, k1y = h*f(xn, yn), h*g(xn, yn)
        k2x, k2y = h*f(xn+k1x/2, yn+k1y/2), h*g(xn+k1x/2, yn+k1y/2)
        k3x, k3y = h*f(xn+k2x/2, yn+k2y/2), h*g(xn+k2x/2, yn+k2y/2)
        k4x, k4y = h*f(xn+k3x, yn+k3y), h*g(xn+k3x, yn+k3y)
        x_vals[i] = xn + (k1x + 2*k2x + 2*k3x + k4x)/6
        y_vals[i] = yn + (k1y + 2*k2y + 2*k3y + k4y)/6
    return t_vals, x_vals, y_vals

# [B] 소프트 제약조건(Soft Constraint) 기반 아키텍처
class SoftPINN(nn.Module):
    def __init__(self, T_max):
        super(SoftPINN, self).__init__()
        self.T_max = T_max
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t):
        # 시간 정규화 유지 (기울기 소실 방지)
        t_norm = t / self.T_max
        # 하드 제약 수식을 제거하고 순수 신경망 출력 반환
        return self.net(t_norm)

def pde_loss_soft(model, t, a, b, x0, y0):
    t.requires_grad = True
    pred = model(t)
    x, y = pred[:, 0:1], pred[:, 1:2]

    # 물리 잔차 (PDE Loss)
    dx_dt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    res_x = dx_dt - x * (1.0 - x - a * y)
    res_y = dy_dt - y * (1.0 - y - b * x)
    loss_pde = torch.mean(res_x**2 + res_y**2)

    # 초기 조건 잔차 (IC Loss) 재추가
    t0 = torch.zeros(1, 1, dtype=torch.float32)
    pred0 = model(t0)
    loss_ic = (pred0[0, 0] - x0)**2 + (pred0[0, 1] - y0)**2

    # 페널티 가중치를 부여하여 두 손실을 병합
    return loss_pde + 100.0 * loss_ic

# [C] Adam 최적화 및 2000 에포크 시각화
def train_and_visualize_adam_soft():
    a, b = 1.5, 0.5
    x0, y0 = 0.2, 0.8
    T_max = 25.0
    epochs = 2000
    
    t_rk4, x_rk4, y_rk4 = rk4_reference(a, b, x0, y0, T_end=T_max)
    
    model = SoftPINN(T_max)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    t_train = torch.linspace(0, T_max, 400).view(-1, 1)
    t_test = torch.linspace(0, T_max, 500).view(-1, 1)
    t_test_np = t_test.numpy().flatten()
    
    log_epochs = [0, 400, 800, 1200, 1600, 2000]
    history = {}

    print("소프트 제약조건 기반 Adam 학습 시작...")
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        loss = pde_loss_soft(model, t_train, a, b, x0, y0)
        loss.backward()
        optimizer.step()
        
        if epoch in log_epochs:
            with torch.no_grad():
                history[epoch] = model(t_test).numpy()
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.6e}")

    plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": "--"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=120)
    axes = axes.flatten()

    for idx, ep in enumerate(log_epochs):
        ax = axes[idx]
        if ep in history:
            x_pred = history[ep][:, 0]
            y_pred = history[ep][:, 1]

            ax.plot(t_rk4, x_rk4, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='RK4 ($x$)')
            ax.plot(t_rk4, y_rk4, color='silver', linestyle='--', alpha=0.6, linewidth=2, label='RK4 ($y$)')
            ax.plot(t_test_np, x_pred, color='#1f77b4', linestyle='-', linewidth=2, label='PINN ($x$)')
            ax.plot(t_test_np, y_pred, color='#d62728', linestyle='-', linewidth=2, label='PINN ($y$)')
            
        ax.set_title(f"Epoch: {ep}")
        ax.set_xlim(0, 25)
        ax.set_ylim(-0.1, 1.2)
        ax.set_xlabel("Time ($t$)")
        ax.set_ylabel("Population Density")
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.suptitle("Soft Constraint PINN with Adam Optimizer", fontsize=16)
    plt.tight_layout()
    plt.show()

train_and_visualize_adam_soft()