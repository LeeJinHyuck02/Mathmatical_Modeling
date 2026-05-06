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
st.set_page_config(page_title="DeepONet EV vs ICEV", layout="wide")
st.title("Parametric Operator Learning Simulator")
st.markdown("매개변수화된 딥오넷(Parametric DeepONet)을 활용하여 수치 적분(RK4) 없이 O(1)의 속도로 로트카-볼테라 생태계의 궤적을 실시간 추론합니다.")

# 좌측 사이드바에 4개의 슬라이더 배치
st.sidebar.header("Input Variables")
st.sidebar.markdown("초기 조건 (Initial Conditions)")
x0 = st.sidebar.slider("Init Species 1 (x0)", 0.01, 1.00, 0.20, 0.01)
y0 = st.sidebar.slider("Init Species 2 (y0)", 0.01, 1.00, 0.80, 0.01)

st.sidebar.markdown("경쟁 계수 (Competition Params)")
a = st.sidebar.slider("Param 'a' (ICEV -> EV 억제력)", 0.1, 3.0, 1.5, 0.1)
b = st.sidebar.slider("Param 'b' (EV -> ICEV 억제력)", 0.1, 3.0, 0.5, 0.1)

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
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# 도표 렌더링
ax.plot(t_np, prediction[:, 0], color='#1f77b4', linewidth=3, label="Species 1 ($x$)")
ax.plot(t_np, prediction[:, 1], color='#d62728', linestyle='--', linewidth=3, label="Species 2 ($y$)")

ax.scatter(0, x0, color='blue', s=80, zorder=5)
ax.scatter(0, y0, color='red', s=80, zorder=5)

ax.set_title(f"Interactive Trajectory Inference (a={a:.1f}, b={b:.1f})")
ax.set_xlabel("Time ($t$)")
ax.set_ylabel("Population Density")
ax.set_xlim(-0.5, T_end)
ax.set_ylim(-0.1, 1.3)
ax.legend(loc='upper right', ncol=2)

# Streamlit 화면에 매트플롯립 피규어 전송
st.pyplot(fig)