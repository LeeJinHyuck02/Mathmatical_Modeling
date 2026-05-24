import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 아키텍처 재선언
# ==========================================
class ParametricBranchNet(nn.Module):
    def __init__(self, output_dim=64):
        super(ParametricBranchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
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

class HardConstraintDeepONet(nn.Module):
    def __init__(self, p=64):
        super(HardConstraintDeepONet, self).__init__()
        self.branch_x = ParametricBranchNet(output_dim=p)
        self.trunk_x = TrunkNet(output_dim=p)
        self.bias_x = nn.Parameter(torch.zeros(1))
        
        self.branch_y = ParametricBranchNet(output_dim=p)
        self.trunk_y = TrunkNet(output_dim=p)
        self.bias_y = nn.Parameter(torch.zeros(1))

    def forward(self, u, t):
        x0 = u[:, 0:1]
        y0 = u[:, 1:2]
        
        b_x = self.branch_x(u)
        t_x = self.trunk_x(t)
        raw_x = torch.sum(b_x * t_x, dim=1, keepdim=True) + self.bias_x
        
        b_y = self.branch_y(u)
        t_y = self.trunk_y(t)
        raw_y = torch.sum(b_y * t_y, dim=1, keepdim=True) + self.bias_y
        
        x = x0 + (1.0 - torch.exp(-t)) * raw_x
        y = y0 + (1.0 - torch.exp(-t)) * raw_y
        return torch.cat([x, y], dim=1)

# ==========================================
# 2. 캐싱을 활용한 초고속 모델 로드
# ==========================================
@st.cache_resource
def load_model():
    model = HardConstraintDeepONet(p=64)
    # CPU 환경에서도 안전하게 로드될 수 있도록 map_location 지정
    model.load_state_dict(torch.load("deeponet_weights.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

model = load_model()

# ==========================================
# 3. Streamlit 웹 UI 구성 및 사이드바 제어
# ==========================================
st.set_page_config(page_title="DeepONet Simulator", layout="wide")
st.title("Competition of Two Species: Advanced Simulator")

# 좌측 사이드바에 4개의 슬라이더 배치
st.sidebar.header("Model Parameters")
a = st.sidebar.slider("Parameter $a$ (Species 2 on 1)", 0.1, 2.0, 1.5, 0.05)
b = st.sidebar.slider("Parameter $b$ (Species 1 on 2)", 0.1, 2.0, 0.5, 0.05)

st.sidebar.header("Initial Conditions")
auto_balance = st.sidebar.checkbox("Maintain $x_0 + y_0 = 1.0$", value=False)

x0 = st.sidebar.slider("Initial $x_0$ (Species 1)", 0.01, 1.0, 0.2, 0.01)

# 체크박스 상태에 따른 y0 변수 할당 로직
if auto_balance:
    y0 = 1.0 - x0
    st.sidebar.info(f"Auto-calculated: Initial $y_0$ = {y0:.2f}")
else:
    y0 = st.sidebar.slider("Initial $y_0$ (Species 2)", 0.01, 1.0, 0.8, 0.01)

# ==========================================
# 4. 실시간 순전파 추론 및 시각화
# ==========================================
T_end = 15.0
num_points = 300
t_np = np.linspace(0, T_end, num_points)
t_tensor = torch.tensor(t_np, dtype=torch.float32).view(-1, 1)

# 사용자가 조작한 4개의 변수를 하나의 텐서로 결합
u_tensor = torch.tensor([[x0, y0, a, b]] * num_points, dtype=torch.float32)

# 단 1회의 행렬 연산으로 전체 시간 도메인 궤적 생성
with torch.no_grad():
    prediction = model(u_tensor, t_tensor).numpy()

plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": "--"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

# 시계열 그래프
ax1.plot(t_np, prediction[:, 0], label="Species 1 ($x$)", color='#1f77b4', linewidth=2)
ax1.plot(t_np, prediction[:, 1], linestyle="--", color='#d62728', linewidth=2, label="Species 2 ($y$)")
ax1.set_title("Population Density Time Series")
ax1.set_xlabel("Time ($t$)")
ax1.set_ylabel("Density")
ax1.set_xlim(0, T_end)
ax1.set_ylim(-0.05, 1.5)
ax1.legend(loc='upper right')

# 위상 평면
ax2.plot(prediction[:, 0], prediction[:, 1], color="green", linewidth=2)
ax2.scatter([x0], [y0], color="blue", s=60, label="Start", zorder=5)
ax2.scatter([prediction[-1, 0]], [prediction[-1, 1]], color="black", marker="x", s=80, label="End", zorder=5)

# 평형점(Equilibrium) 계산 및 마킹
if a * b != 1.0:
    x_eq = (1 - a) / (1 - a * b)
    y_eq = (1 - b) / (1 - a * b)
    if x_eq >= 0 and y_eq >= 0:
        ax2.scatter([x_eq], [y_eq], color='purple', marker='*', s=150, zorder=5, label="Equilibrium")

ax2.set_title("Phase Portrait")
ax2.set_xlabel("Species 1 ($x$)")
ax2.set_ylabel("Species 2 ($y$)")
ax2.set_xlim(-0.05, 1.5)
ax2.set_ylim(-0.05, 1.5)
ax2.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)