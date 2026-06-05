import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.integrate import odeint

# =========================================================
# 1. 환경 설정 및 블록 평균(Block Averaging) 데이터 준비
# =========================================================
font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

INPUT_CSV = "ideology_approval_3way.csv"
BLOCK_SIZE = 5  # "데이터를 묶어서 평균 낼 구간 크기 지정"

if not os.path.exists(INPUT_CSV):
    print(f"[오류] '{INPUT_CSV}' 파일이 존재하지 않습니다.")
    exit()

df = pd.read_csv(INPUT_CSV)
raw_data = df[["진보_합계", "보수_합계", "중도_합계"]].dropna().values
t_raw = np.arange(len(raw_data)).reshape(-1, 1)

# 이동평균 및 다운샘플링을 대체하는 직접 블록 평균 연산
num_blocks = math.ceil(len(raw_data) / BLOCK_SIZE)
x_sampled_list = []
t_sampled_list = []

for i in range(num_blocks):
    start_idx = i * BLOCK_SIZE
    end_idx = min((i + 1) * BLOCK_SIZE, len(raw_data))

    x_chunk = raw_data[start_idx:end_idx]
    t_chunk = t_raw[start_idx:end_idx]

    # 상태 변수(x)와 시간(t) 모두 평균점(Center of mass)으로 치환
    x_sampled_list.append(x_chunk.mean(axis=0))
    t_sampled_list.append(t_chunk.mean(axis=0))

x_sampled = np.array(x_sampled_list)
t_sampled = np.array(t_sampled_list)

T_MAX = float(t_raw.max())
X_MAX = 100.0

t_tensor = torch.tensor(t_sampled / T_MAX, dtype=torch.float32, requires_grad=True)
x_tensor = torch.tensor(x_sampled / X_MAX, dtype=torch.float32)

# =========================================================
# 2. 9-파라미터 PINN 신경망 구조 (r 복원)
# =========================================================
def uniform_softplus_init(tensor_shape, low, high):
    "지정된 물리적 범위 [low, high] 내에서 최종값이 균등 분포를 가지도록 원시 텐서를 초기화하는 함수"
    uniform_phys_vals = torch.empty(tensor_shape, dtype=torch.float32).uniform_(low, high)
    inv_softplus_vals = torch.log(torch.exp(uniform_phys_vals) - 1.0)
    return inv_softplus_vals

class PoliticalPINN_UniformInit(nn.Module):
    def __init__(self):
        super(PoliticalPINN_UniformInit, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

        init_r = uniform_softplus_init((3,), low=0.05, high=0.9)
        self.raw_r = nn.Parameter(init_r)

        init_alpha = uniform_softplus_init((6,), low=0.05, high=2.0)
        self.raw_alpha = nn.Parameter(init_alpha)

    def forward(self, t):
        return self.net(t)

    def get_physics_params(self):
        r = F.softplus(self.raw_r) + 1e-4
        alpha = F.softplus(self.raw_alpha) + 1e-4
        return r, alpha

    def get_real_parameters(self):
        r, alpha = self.get_physics_params()
        return r.detach().numpy(), alpha.detach().numpy()

# =========================================================
# 3. 단일화된 손실 함수 (Total Loss = Data + Physics + Prior)
# =========================================================
def compute_loss(model, t, x_true):
    x_pred = model(t)

    loss_data = torch.mean((x_pred - x_true)**2)

    x1, x2, x3 = x_pred[:, 0:1], x_pred[:, 1:2], x_pred[:, 2:3]

    dx1_dt = torch.autograd.grad(x1, t, grad_outputs=torch.ones_like(x1), create_graph=True)[0] / T_MAX
    dx2_dt = torch.autograd.grad(x2, t, grad_outputs=torch.ones_like(x2), create_graph=True)[0] / T_MAX
    dx3_dt = torch.autograd.grad(x3, t, grad_outputs=torch.ones_like(x3), create_graph=True)[0] / T_MAX

    r, alpha = model.get_physics_params()

    ode_res_1 = dx1_dt - r[0] * x1 * (1 - x1 - alpha[0]*x2 - alpha[1]*x3)
    ode_res_2 = dx2_dt - r[1] * x2 * (1 - alpha[2]*x1 - x2 - alpha[3]*x3)
    ode_res_3 = dx3_dt - r[2] * x3 * (1 - alpha[4]*x1 - alpha[5]*x2 - x3)
    loss_physics = torch.mean(ode_res_1**2 + ode_res_2**2 + ode_res_3**2)

    diff_pred = torch.abs(x_pred[1:] - x_pred[:-1])
    prior_diff = 1e-4 / (torch.mean(diff_pred) + 1e-6)

    penalty_r = torch.mean(torch.relu(r - 5.0)**2)
    penalty_alpha = torch.mean(torch.relu(alpha - 10.0)**2)

    prior = prior_diff + (100.0 * penalty_r) + (100.0 * penalty_alpha)

    total_loss = loss_data + 3* loss_physics
    return total_loss

# =========================================================
# 4. 훈련 루프
# =========================================================
def run_pinn_block_averaged(epochs=5000):
    print(f"데이터 블록 평균(Size={BLOCK_SIZE}) 기반 학습 시작...")
    model = PoliticalPINN_UniformInit()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(model, t_tensor, x_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f" -> Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

    print(f"\n[완료] 훈련 종료. (최종 Loss: {loss.item():.6f})")
    return model.get_real_parameters()

# =========================================================
# 5. 시뮬레이션 및 시각화
# =========================================================
def dimensional_ode(x, t, r, alpha):
    x1, x2, x3 = max(0, x[0]), max(0, x[1]), max(0, x[2])
    dx1 = r[0] * x1 * (1 - x1 - alpha[0]*x2 - alpha[1]*x3)
    dx2 = r[1] * x2 * (1 - alpha[2]*x1 - x2 - alpha[3]*x3)
    dx3 = r[2] * x3 * (1 - alpha[4]*x1 - alpha[5]*x2 - x3)
    return [dx1, dx2, dx3]

if __name__ == "__main__":
    opt_r, opt_alpha = run_pinn_block_averaged(epochs=5000)

    future_steps = 52
    t_extended = np.arange(len(raw_data) + future_steps)
    # 초기값을 블록 평균 데이터의 첫 번째 값으로 설정
    x0_norm = x_sampled[0] / X_MAX

    simulated_data_norm = odeint(dimensional_ode, x0_norm, t_extended, args=(opt_r, opt_alpha))
    simulated_data = simulated_data_norm * X_MAX

    plt.figure(figsize=(16, 8))

    # 1. 원본 데이터 산점도 (흐리게 표시하여 전체 분포 묘사)
    plt.plot(t_raw, raw_data[:, 0], 'o', color='royalblue', alpha=0.15)
    plt.plot(t_raw, raw_data[:, 1], 's', color='firebrick', alpha=0.15)
    plt.plot(t_raw, raw_data[:, 2], '^', color='darkorange', alpha=0.15)

    # 2. 학습에 실제 사용된 블록 평균 데이터 (진하게 표시)
    plt.plot(t_sampled, x_sampled[:, 0], 'o', color='blue', markersize=8, label="학습 데이터 (진보)")
    plt.plot(t_sampled, x_sampled[:, 1], 's', color='red', markersize=8, label="학습 데이터 (보수)")
    plt.plot(t_sampled, x_sampled[:, 2], '^', color='darkorange', markersize=8, label="학습 데이터 (중도)")

    colors = ['blue', 'red', 'darkorange']
    labels = ['진보', '보수', '중도']

    # 3. 모델 피팅 및 예측 곡선
    for i in range(3):
        plt.plot(t_raw, simulated_data[:len(t_raw), i], '-', color=colors[i], linewidth=2.5, label=f"피팅 궤적 ({labels[i]})")
        plt.plot(t_extended[len(t_raw):], simulated_data[len(t_raw):, i], '--', color=colors[i], linewidth=3, label=f"미래 예측 ({labels[i]})")

    plt.axvline(x=len(t_raw)-1, color='black', linestyle='-.', linewidth=2)
    plt.title(f"블록 평균(Block Size={BLOCK_SIZE}) 기반 노이즈 필터링 PINN 예측", fontsize=18, pad=20)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()