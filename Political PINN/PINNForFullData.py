import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.integrate import odeint
import math

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

# 데이터 로드 및 정규화
df = pd.read_csv(INPUT_CSV)
raw_data = df[["진보_합계", "보수_합계", "중도_합계"]].dropna().values
smoothed_data = pd.DataFrame(raw_data).rolling(window=4, min_periods=1, center=True).mean().values

t_raw = np.arange(len(smoothed_data)).reshape(-1, 1)
x_raw = smoothed_data

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

        # [교정] 정답 파라미터를 직접 명시하지 않고, 통계적 중심점(Center)을 간접 참조하여 초기화
        # 난수(randn)에 표준편차(0.8)를 곱해 중심점 주변으로 충분히 넓은 탐색 공간을 확보합니다.
        
        # 1. r: 중심점을 0.7로 설정 (Max 1.0 기준)
        target_r_norm = 0.7 / 1.0
        logit_r = math.log(target_r_norm / (1 - target_r_norm))
        self.raw_r = nn.Parameter(torch.randn(3, dtype=torch.float32) * 0.8 + logit_r)

        # 2. K: 중심점을 75(정규화 0.75)로 설정 (Max 1.2 기준)
        target_K_norm = 0.75 / 1.2
        logit_K = math.log(target_K_norm / (1 - target_K_norm))
        self.raw_K = nn.Parameter(torch.randn(3, dtype=torch.float32) * 0.8 + logit_K)

        # 3. alpha: 중심점을 0.8로 설정 (Max 2.0 기준)
        target_alpha_norm = 0.8 / 2.0
        logit_alpha = math.log(target_alpha_norm / (1 - target_alpha_norm))
        self.raw_alpha = nn.Parameter(torch.randn(6, dtype=torch.float32) * 0.8 + logit_alpha)

    def forward(self, t):
        return self.net(t)

    def get_physics_params(self):
        # 경계(Bounds)는 이전 제언을 반영하여 하한선을 완화(0.001)한 상태로 유지
        r = 0.001 + 0.499 * torch.sigmoid(self.raw_r)
        K = 0.2 + 0.5 * torch.sigmoid(self.raw_K)
        alpha = 2.0 * torch.sigmoid(self.raw_alpha)
        return r, K, alpha

    def get_real_parameters(self):
        r, K, alpha = self.get_physics_params()
        return r.detach().numpy(), K.detach().numpy() * X_MAX, alpha.detach().numpy()

# =========================================================
# 3. 손실 함수 (기울기 페널티 없이 단순 Data + Physics)
# =========================================================
def compute_loss(model, t, x_true):
    x_pred = model(t)
    loss_data = torch.mean((x_pred - x_true)**2)

    x1, x2, x3 = x_pred[:, 0:1], x_pred[:, 1:2], x_pred[:, 2:3]

    dx1_dt = torch.autograd.grad(x1, t, grad_outputs=torch.ones_like(x1), create_graph=True)[0] / T_MAX
    dx2_dt = torch.autograd.grad(x2, t, grad_outputs=torch.ones_like(x2), create_graph=True)[0] / T_MAX
    dx3_dt = torch.autograd.grad(x3, t, grad_outputs=torch.ones_like(x3), create_graph=True)[0] / T_MAX

    r, K, a = model.get_physics_params()

    ode_res_1 = dx1_dt - r[0] * x1 * (1 - (x1 + a[0]*x2 + a[1]*x3) / K[0])
    ode_res_2 = dx2_dt - r[1] * x2 * (1 - (a[2]*x1 + x2 + a[3]*x3) / K[1])
    ode_res_3 = dx3_dt - r[2] * x3 * (1 - (a[4]*x1 + a[5]*x2 + x3) / K[2])

    loss_physics = torch.mean(ode_res_1**2 + ode_res_2**2 + ode_res_3**2)

    # 1. 분산 기반 prior 제거 후 데이터 궤적의 차분 절대값 페널티로 대체
    diff_pred = torch.abs(x_pred[1:] - x_pred[:-1])
    loss_prior_diff = 1.0 / (torch.mean(diff_pred) + 1e-6) # 변동이 없을수록 막대한 페널티 부과 (기울기 소실 없음)

    total_loss = (loss_data * 2) + loss_physics 

    return total_loss

# =========================================================
# 4. Adam 전용 단일(Single) 훈련 파이프라인
# =========================================================
def run_single_pinn(epochs=4000):
    print("PINN 단일 모델 훈련 시작...")

    # 4-1. 단일 신경망 인스턴스 및 옵티마이저 생성
    model = PoliticalPINN_MultiStart()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    # 4-2. L-BFGS 없이 오직 Adam으로 지정된 Epoch만큼 훈련 수행
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(model, t_tensor, x_tensor)
        loss.backward()
        optimizer.step()

        # 1000 에포크 단위로 진행 상황 출력
        if (epoch + 1) % 1000 == 0:
            print(f" -> Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

    final_loss = loss.item()
    print(f"\n[완료] 훈련 종료. (최종 Loss: {final_loss:.6f})")

    # 4-3. 최종 훈련된 모델의 파라미터 즉시 반환
    final_params = model.get_real_parameters()
    return final_params

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
    # 단일 모델을 4000번 Adam 훈련 (다중 시작 대비 연산량 80% 감소)
    opt_r, opt_K, opt_alpha_flat = run_single_pinn(epochs=4000)

    opt_alpha = np.array([
        [1.0, opt_alpha_flat[0], opt_alpha_flat[1]],
        [opt_alpha_flat[2], 1.0, opt_alpha_flat[3]],
        [opt_alpha_flat[4], opt_alpha_flat[5], 1.0]
    ])

    future_steps = 52
    t_extended = np.arange(len(x_raw) + future_steps)
    x0 = x_raw[0]

    simulated_data = odeint(lotka_volterra_ode, x0, t_extended, args=(opt_r, opt_K, opt_alpha))

    # 시각화 수행
    plt.figure(figsize=(16, 8))
    plt.plot(t_raw, raw_data[:, 0], 'o', color='royalblue', alpha=0.2)
    plt.plot(t_raw, raw_data[:, 1], 's', color='firebrick', alpha=0.2)
    plt.plot(t_raw, raw_data[:, 2], '^', color='darkorange', alpha=0.2)

    colors = ['blue', 'red', 'darkorange']
    labels = ['진보', '보수', '중도']

    for i in range(3):
        plt.plot(t_raw, simulated_data[:len(x_raw), i], '-', color=colors[i], linewidth=2.5, label=f"피팅 ({labels[i]})")
        plt.plot(t_extended[len(x_raw):], simulated_data[len(x_raw):, i], '--', color=colors[i], linewidth=3, label=f"예측 ({labels[i]})")

    plt.axvline(x=len(x_raw)-1, color='black', linestyle='-.', linewidth=2)
    plt.title("단일 PINN (Adam Only) 기반 파라미터 역산 및 미래 예측", fontsize=18, pad=20)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()