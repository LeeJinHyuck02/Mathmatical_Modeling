import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# 1. 아키텍처 정의
class BranchNet(nn.Module):
    def __init__(self, output_dim=64):
        super(BranchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, output_dim)
        )
    def forward(self, u): return self.net(u)

class TrunkNet(nn.Module):
    def __init__(self, output_dim=64):
        super(TrunkNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, output_dim)
        )
    def forward(self, t): return self.net(t)

class DeepONet(nn.Module):
    def __init__(self, p=64):
        super(DeepONet, self).__init__()
        self.branch_x = BranchNet(output_dim=p)
        self.trunk_x = TrunkNet(output_dim=p)
        self.bias_x = nn.Parameter(torch.zeros(1))

        self.branch_y = BranchNet(output_dim=p)
        self.trunk_y = TrunkNet(output_dim=p)
        self.bias_y = nn.Parameter(torch.zeros(1))

    def forward(self, u, t):
        b_x = self.branch_x(u)
        t_x = self.trunk_x(t)
        out_x = torch.sum(b_x * t_x, dim=1, keepdim=True) + self.bias_x

        b_y = self.branch_y(u)
        t_y = self.trunk_y(t)
        out_y = torch.sum(b_y * t_y, dim=1, keepdim=True) + self.bias_y
        return torch.cat([out_x, out_y], dim=1)

# 2. 훈련 데이터 생성 (RK4 기반)
def generate_data(num_samples=50, T_end=15.0, h=0.05):
    a, b_param = 1.5, 0.5
    f = lambda x, y: x * (1 - x - a * y)
    g = lambda x, y: y * (1 - y - b_param * x)

    num_steps = int(T_end / h)
    t_vals = np.linspace(0, T_end, num_steps)

    u_data, t_data, x_target, y_target = [], [], [], []
    for _ in range(num_samples):
        x0 = np.random.uniform(0.01, 1.0)
        y0 = np.random.uniform(0.01, 1.0)

        x_curr, y_curr = x0, y0
        for t in t_vals:
            u_data.append([x0, y0])
            t_data.append([t])
            x_target.append([x_curr])
            y_target.append([y_curr])

            k1x, k1y = h*f(x_curr, y_curr), h*g(x_curr, y_curr)
            k2x, k2y = h*f(x_curr+k1x/2, y_curr+k1y/2), h*g(x_curr+k1x/2, y_curr+k1y/2)
            k3x, k3y = h*f(x_curr+k2x/2, y_curr+k2y/2), h*g(x_curr+k2x/2, y_curr+k2y/2)
            k4x, k4y = h*f(x_curr+k3x, y_curr+k3y), h*g(x_curr+k3x, y_curr+k3y)
            x_curr += (k1x + 2*k2x + 2*k3x + k4x)/6
            y_curr += (k1y + 2*k2y + 2*k3y + k4y)/6

    return (torch.tensor(u_data, dtype=torch.float32),
            torch.tensor(t_data, dtype=torch.float32),
            torch.tensor(x_target, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32))

u_train, t_train, x_train, y_train = generate_data()

# 3. 모델 훈련
model = DeepONet(p=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("DeepONet 훈련 시작...")
for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(u_train, t_train)
    loss = torch.mean((pred[:, 0:1] - x_train)**2 + (pred[:, 1:2] - y_train)**2)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | MSE Loss: {loss.item():.6e}")

# 가중치 저장
torch.save(model.state_dict(), "deeponet_weights.pth")
print("모델 가중치 저장 완료: deeponet_weights.pth")

# 4. 검증 및 시각화
test_x0, test_y0 = 0.8, 0.2
t_test_np = np.linspace(0, 15.0, 300)
u_test = torch.tensor([[test_x0, test_y0]] * 300, dtype=torch.float32)
t_test = torch.tensor(t_test_np, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    pred_test = model(u_test, t_test).numpy()

plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": "--"})
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
ax.plot(t_test_np, pred_test[:, 0], 'b-', linewidth=2, label="DeepONet Pred: Species 1 ($x$)")
ax.plot(t_test_np, pred_test[:, 1], 'r--', linewidth=2, label="DeepONet Pred: Species 2 ($y$)")
ax.set_title(f"DeepONet Offline Validation (IC: x0={test_x0}, y0={test_y0})")
ax.set_xlabel("Time ($t$)")
ax.set_ylabel("Population Density")
ax.legend()
plt.tight_layout()
plt.show()
