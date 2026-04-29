import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 재현성을 위한 시드 고정
torch.manual_seed(42)

# [A] 기준 해(Ground Truth) 생성을 위한 RK4
def rk4_reference(a, b, x0, y0, T_end=25.0, h=0.05):
    def f(x, y): return x * (1 - x - a * y)
    def g(x, y): return y * (1 - y - b * x)
    
    num_steps = int(T_end / h)
    t_vals = np.linspace(0, T_end, num_steps)
    x_vals = np.zeros(num_steps)
    y_vals = np.zeros(num_steps)
    
    x_vals[0], y_vals[0] = x0, y0
    for i in range(1, num_steps):
        xn, yn = x_vals[i-1], y_vals[i-1]
        k1x, k1y = h * f(xn, yn), h * g(xn, yn)
        k2x, k2y = h * f(xn + k1x/2, yn + k1y/2), h * g(xn + k1x/2, yn + k1y/2)
        k3x, k3y = h * f(xn + k2x/2, yn + k2y/2), h * g(xn + k2x/2, yn + k2y/2)
        k4x, k4y = h * f(xn + k3x, yn + k3y), h * g(xn + k3x, yn + k3y)
        
        x_vals[i] = xn + (k1x + 2*k2x + 2*k3x + k4x) / 6
        y_vals[i] = yn + (k1y + 2*k2y + 2*k3y + k4y) / 6
        
    return t_vals, x_vals, y_vals

# [B] PINN 모델 및 손실 함수 정의
class CompetitionPINN(nn.Module):
    def __init__(self):
        super(CompetitionPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t):
        return self.net(t)

def pde_loss(model, t, a, b, x0, y0):
    t.requires_grad = True
    pred = model(t)
    x, y = pred[:, 0:1], pred[:, 1:2]

    dx_dt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    res_x = dx_dt - x * (1.0 - x - a * y)
    res_y = dy_dt - y * (1.0 - y - b * x)
    loss_physics = torch.mean(res_x**2 + res_y**2)

    t0 = torch.zeros(1, 1, dtype=torch.float32)
    pred0 = model(t0)
    loss_ic = (pred0[0, 0] - x0)**2 + (pred0[0, 1] - y0)**2

    return loss_physics + 100.0 * loss_ic

# [C] 200 에포크 한정 학습 및 시계열 진화 시각화
def train_and_visualize_fast_convergence():
    a, b = 1.5, 0.5
    x0, y0 = 0.2, 0.8
    epochs = 1000
    
    t_rk4, x_rk4, y_rk4 = rk4_reference(a, b, x0, y0)
    
    model = CompetitionPINN()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    t_train = torch.linspace(0, 25, 400).view(-1, 1)
    t_test = torch.linspace(0, 25, 500).view(-1, 1)
    t_test_np = t_test.numpy().flatten()
    
    # 40 간격으로 에포크 추적
    log_epochs = [0, 200, 400, 600, 800, 1000]
    history = {}

    print("PINN 고속 수렴 시계열 학습 추적 시작...")
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        loss = pde_loss(model, t_train, a, b, x0, y0)
        loss.backward()
        optimizer.step()
        
        if epoch in log_epochs:
            with torch.no_grad():
                pred_test = model(t_test).numpy()
            history[epoch] = pred_test
            print(f"Epoch {epoch:3d} | Total Loss: {loss.item():.6f}")

    # 고해상도 학술 렌더링 포맷 (디스플레이용)
    plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": "--"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=120)
    axes = axes.flatten()

    for idx, ep in enumerate(log_epochs):
        ax = axes[idx]
        x_pred = history[ep][:, 0]
        y_pred = history[ep][:, 1]

        # RK4 기준선 렌더링
        ax.plot(t_rk4, x_rk4, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='RK4 Ref ($x$)')
        ax.plot(t_rk4, y_rk4, color='silver', linestyle='--', alpha=0.6, linewidth=2, label='RK4 Ref ($y$)')

        # PINN 예측선 렌더링
        ax.plot(t_test_np, x_pred, color='#1f77b4', linestyle='-', linewidth=2, label='PINN Pred ($x$)')
        ax.plot(t_test_np, y_pred, color='#d62728', linestyle='-', linewidth=2, label='PINN Pred ($y$)')
        
        ax.set_title(f"Epoch: {ep}")
        ax.set_xlim(0, 25)
        ax.set_ylim(-0.1, 1.2)
        ax.set_xlabel("Time ($t$)")
        ax.set_ylabel("Population Density")
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.suptitle("Time Series Evolution during Early PINN Training", fontsize=16)
    plt.tight_layout()
    plt.show()  # 파일 저장 기능 제거 및 화면 출력 단일화

train_and_visualize_fast_convergence()