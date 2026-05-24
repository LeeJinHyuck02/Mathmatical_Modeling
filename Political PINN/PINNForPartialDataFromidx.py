import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.integrate import odeint

# =========================================================
# 1. 환경 설정 및 데이터 준비
# =========================================================
font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

INPUT_CSV = "ideology_approval_3way.csv"
START_IDX = 70 

df = pd.read_csv(INPUT_CSV)
raw_data = df[["진보_합계", "보수_합계", "중도_합계"]].dropna().values
smoothed_data = pd.DataFrame(raw_data).rolling(window=4, min_periods=1, center=True).mean().values

sliced_smoothed_data = smoothed_data[START_IDX:]
t_raw = np.arange(len(sliced_smoothed_data)).reshape(-1, 1)
x_raw = sliced_smoothed_data

T_MAX = float(t_raw.max())
X_MAX = 100.0

t_tensor = torch.tensor(t_raw / T_MAX, dtype=torch.float32, requires_grad=True)
x_tensor = torch.tensor(x_raw / X_MAX, dtype=torch.float32)

# =========================================================
# 2. 다중 시작 대응 PINN 신경망 구조
# =========================================================
class PoliticalPINN_MultiStart(nn.Module):
    def __init__(self):
        super(PoliticalPINN_MultiStart, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

        target_r_norm = 0.35 / 1.0
        logit_r = math.log(target_r_norm / (1 - target_r_norm))
        self.raw_r = nn.Parameter(torch.randn(3, dtype=torch.float32) * 0.8 + logit_r)

        target_K_norm = 0.75 / 1.2
        logit_K = math.log(target_K_norm / (1 - target_K_norm))
        self.raw_K = nn.Parameter(torch.randn(3, dtype=torch.float32) * 0.8 + logit_K)

        target_alpha_norm = 0.8 / 2.0
        logit_alpha = math.log(target_alpha_norm / (1 - target_alpha_norm))
        self.raw_alpha = nn.Parameter(torch.randn(6, dtype=torch.float32) * 0.8 + logit_alpha)
        
    def forward(self, t):
        return self.net(t)

    def get_physics_params(self):
        r = 0.001 + 0.499 * torch.sigmoid(self.raw_r)
        K = 0.2 + 0.5 * torch.sigmoid(self.raw_K)
        alpha = 2.0 * torch.sigmoid(self.raw_alpha)
        return r, K, alpha

    def get_real_parameters(self):
        r, K, alpha = self.get_physics_params()
        return r.detach().numpy(), K.detach().numpy() * X_MAX, alpha.detach().numpy()

# =========================================================
# 3. 손실 함수 (총합 보존 제약 추가)
# =========================================================
def compute_loss(model, t, x_true):
    x_pred = model(t)
    loss_data = torch.mean((x_pred - x_true)**2)

    # [핵심 수정] 예측된 지지율의 합이 1.0(100%)이 되도록 강제하는 제약항 추가
    loss_sum = torch.mean((torch.sum(x_pred, dim=1) - 1.0)**2)

    x1, x2, x3 = x_pred[:, 0:1], x_pred[:, 1:2], x_pred[:, 2:3]

    dx1_dt = torch.autograd.grad(x1, t, grad_outputs=torch.ones_like(x1), create_graph=True)[0] / T_MAX
    dx2_dt = torch.autograd.grad(x2, t, grad_outputs=torch.ones_like(x2), create_graph=True)[0] / T_MAX
    dx3_dt = torch.autograd.grad(x3, t, grad_outputs=torch.ones_like(x3), create_graph=True)[0] / T_MAX

    r, K, a = model.get_physics_params()

    ode_res_1 = dx1_dt - r[0] * x1 * (1 - (x1 + a[0]*x2 + a[1]*x3) / K[0])
    ode_res_2 = dx2_dt - r[1] * x2 * (1 - (a[2]*x1 + x2 + a[3]*x3) / K[1])
    ode_res_3 = dx3_dt - r[2] * x3 * (1 - (a[4]*x1 + a[5]*x2 + x3) / K[2])

    loss_physics = torch.mean(ode_res_1**2 + ode_res_2**2 + ode_res_3**2)

    diff_pred = torch.abs(x_pred[1:] - x_pred[:-1])
    loss_prior_diff = 1.0 / (torch.mean(diff_pred) + 1e-6)

    # 데이터(1) + 물리(7) + 총합 보존(10) 가중치 통합
    total_loss = loss_data + loss_physics + 10.0 * loss_sum

    return total_loss

# =========================================================
# 4. 2단계 하이브리드 훈련 파이프라인
# =========================================================
def run_single_pinn(epochs_adam=4000, max_iter_lbfgs=1000):
    print("PINN 2단계 하이브리드 모델 훈련 시작...")
    model = PoliticalPINN_MultiStart()
    
    print("\n--- [Phase 1] Adam 최적화 시작 ---")
    optimizer_adam = optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()
        loss = compute_loss(model, t_tensor, x_tensor)
        loss.backward()
        optimizer_adam.step()

        if (epoch + 1) % 1000 == 0:
            print(f" -> Adam Epoch [{epoch+1}/{epochs_adam}] Loss: {loss.item():.6f}")

    print("\n--- [Phase 2] L-BFGS 최적화 시작 ---")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter_lbfgs,
        max_eval=max_iter_lbfgs * 1.25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    lbfgs_step_counter = 0

    def closure():
        nonlocal lbfgs_step_counter
        optimizer_lbfgs.zero_grad()
        loss = compute_loss(model, t_tensor, x_tensor)
        loss.backward()
        
        lbfgs_step_counter += 1
        if lbfgs_step_counter % 100 == 0:
            print(f" -> L-BFGS Iter [{lbfgs_step_counter}/{max_iter_lbfgs}] Loss: {loss.item():.6f}")
            
        return loss

    optimizer_lbfgs.step(closure)

    final_loss = compute_loss(model, t_tensor, x_tensor).item()
    print(f"\n[완료] 훈련 종료. (최종 하이브리드 Loss: {final_loss:.6f})")
    return model.get_real_parameters()

# =========================================================
# 5. 실행 및 순수 수학적 외삽(ODEINT) 시각화
# =========================================================
def lotka_volterra_ode(x, t, r, K, alpha):
    x1, x2, x3 = max(0, x[0]), max(0, x[1]), max(0, x[2])
    dx1 = r[0] * x1 * (1 - (x1 + alpha[0,1]*x2 + alpha[0,2]*x3) / K[0])
    dx2 = r[1] * x2 * (1 - (alpha[1,0]*x1 + x2 + alpha[1,2]*x3) / K[1])
    dx3 = r[2] * x3 * (1 - (alpha[2,0]*x1 + alpha[2,1]*x2 + x3) / K[2])
    return [dx1, dx2, dx3]

if __name__ == "__main__":
    opt_r, opt_K, opt_alpha_flat = run_single_pinn(epochs_adam=4000, max_iter_lbfgs=1000)

    opt_alpha = np.array([
        [1.0, opt_alpha_flat[0], opt_alpha_flat[1]],
        [opt_alpha_flat[2], 1.0, opt_alpha_flat[3]],
        [opt_alpha_flat[4], opt_alpha_flat[5], 1.0]
    ])

    future_steps = 52
    t_extended = np.arange(len(x_raw) + future_steps)
    
    # 갱신된 새로운 시점의 초기 조건 (총합이 100에 근사하도록 보정됨)
    x0 = x_raw[0]

    simulated_data = odeint(lotka_volterra_ode, x0, t_extended, args=(opt_r, opt_K, opt_alpha))
    
    # 원본 데이터 좌표계(0 ~ N)에 맞추어 시뮬레이션 결과의 위상(x축) 평행 이동
    t_original = np.arange(len(raw_data))
    t_shifted_fit = np.arange(START_IDX, START_IDX + len(x_raw))
    t_shifted_predict = np.arange(START_IDX + len(x_raw), START_IDX + len(x_raw) + future_steps)

    plt.figure(figsize=(16, 8))
    
    plt.plot(t_original, raw_data[:, 0], 'o', color='royalblue', alpha=0.2)
    plt.plot(t_original, raw_data[:, 1], 's', color='firebrick', alpha=0.2)
    plt.plot(t_original, raw_data[:, 2], '^', color='darkorange', alpha=0.2)

    colors = ['blue', 'red', 'darkorange']
    labels = ['진보', '보수', '중도']

    for i in range(3):
        plt.plot(t_shifted_fit, simulated_data[:len(x_raw), i], '-', color=colors[i], linewidth=2.5, label=f"피팅 ({labels[i]})")
        plt.plot(t_shifted_predict, simulated_data[len(x_raw):, i], '--', color=colors[i], linewidth=3, label=f"예측 ({labels[i]})")

    plt.axvline(x=START_IDX + len(x_raw) - 1, color='black', linestyle='-.', linewidth=2)
    plt.axvline(x=START_IDX, color='gray', linestyle=':', linewidth=2, label="학습 시작점")
    
    plt.title(f"총합 100% 보존 제약이 적용된 PINN 궤적 예측", fontsize=18, pad=20)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()