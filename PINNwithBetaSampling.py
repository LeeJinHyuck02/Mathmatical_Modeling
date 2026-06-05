import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(411112)

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

# [B] 소프트 제약조건(Soft Constraint) 기반 표준 아키텍처
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
        t_norm = t / self.T_max
        return self.net(t_norm)

# 물리 잔차 및 초기 조건 잔차 동시 계산
def pde_loss_soft(model, t, a, b, x0, y0):
    t.requires_grad = True
    pred = model(t)
    x, y = pred[:, 0:1], pred[:, 1:2]

    dx_dt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    res_x = dx_dt - x * (1.0 - x - a * y)
    res_y = dy_dt - y * (1.0 - y - b * x)
    loss_pde = torch.mean(res_x**2 + res_y**2)

    # 초기 조건 오차 계산
    t0 = torch.zeros(1, 1, dtype=torch.float32, device=t.device)
    pred0 = model(t0)
    loss_ic = (pred0[0, 0] - x0)**2 + (pred0[0, 1] - y0)**2

    # 페널티 가중치 결합
    return loss_pde + loss_ic

# [C] 베타 분포 기반 적응형 표본추출 훈련
def train_with_beta_distribution():
    a, b = 0.5, 1.5
    x0, y0 = 0.2, 0.8
    T_max = 25.0
    epochs = 4000 
    num_collocation_pts = 600
    
    t_rk4, x_rk4, y_rk4 = rk4_reference(a, b, x0, y0, T_end=T_max)
    
    model = SoftPINN(T_max)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 베타 분포 선언 (alpha=1.0, beta=5.0: 원점 근처에 극단적 밀집)
    beta_dist = torch.distributions.beta.Beta(1.0, 8.0)
    
    print("베타 분포 샘플링을 적용한 Soft PINN 훈련을 시작합니다.")
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        # 매 에포크마다 새로운 훈련 데이터를 0~1 사이에서 추출 후 T_max 곱연산
        t_norm_samples = beta_dist.sample((num_collocation_pts, 1))
        t_train = (t_norm_samples * T_max).clone().detach().requires_grad_(True)
        
        loss = pde_loss_soft(model, t_train, a, b, x0, y0)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.6e}")

    # 평가용 데이터는 전체 도메인을 확인하기 위해 균등 간격 사용
    with torch.no_grad():
        t_test = torch.linspace(0, T_max, 500).view(-1, 1)
        pred_test = model(t_test).numpy()
        t_test_np = t_test.numpy().flatten()

    plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": "--"})
    plt.figure(figsize=(8, 5), dpi=120)
    
    plt.plot(t_rk4, x_rk4, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='RK4 ($x$)')
    plt.plot(t_rk4, y_rk4, color='silver', linestyle='--', alpha=0.7, linewidth=2, label='RK4 ($y$)')
    plt.plot(t_test_np, pred_test[:, 0], color='#1f77b4', linestyle='-', linewidth=2, label='Beta-SoftPINN ($x$)')
    plt.plot(t_test_np, pred_test[:, 1], color='#d62728', linestyle='-', linewidth=2, label='Beta-SoftPINN ($y$)')
    
    plt.title(f"Soft PINN with Beta(1.0, 5.0) Sampling (a={a}, b={b})")
    plt.xlabel("Time ($t$)")
    plt.ylabel("Population Density")
    plt.legend(loc='upper right')
    plt.xlim(0, 25)
    plt.ylim(-0.1, 1.2)
    plt.show()

train_with_beta_distribution()
