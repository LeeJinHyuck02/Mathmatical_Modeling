import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="Lotka-Volterra Simulator", layout="wide")
st.title("Competition of Two Species: Advanced Simulator")

# 2. 사이드바 - 매개변수 설정
st.sidebar.header("Model Parameters")
a = st.sidebar.slider("Parameter $a$ (Species 2 on 1)", 0.1, 2.0, 1.5, 0.05)
b = st.sidebar.slider("Parameter $b$ (Species 1 on 2)", 0.1, 2.0, 0.5, 0.05)

# 3. 사이드바 - 초기 조건 및 자동 조정 위젯
st.sidebar.header("Initial Conditions")
auto_balance = st.sidebar.checkbox("Maintain $x_0 + y_0 = 1.0$", value=False)

x0 = st.sidebar.slider("Initial $x_0$ (Species 1)", 0.01, 1.0, 0.2, 0.01)

# 체크박스 상태에 따른 y0 변수 할당 로직
if auto_balance:
    y0 = 1.0 - x0
    st.sidebar.info(f"Auto-calculated: Initial $y_0$ = {y0:.2f}")
else:
    y0 = st.sidebar.slider("Initial $y_0$ (Species 2)", 0.01, 1.0, 0.8, 0.01)

# 4. RK4 알고리즘 엔진
def simulate(a, b, x0, y0, T_end=50.0, h=0.05):
    f = lambda x, y: x * (1 - x - a * y)
    g = lambda x, y: y * (1 - y - b * x)
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

# 5. 결과 계산 및 고정축 시각화
t, x, y = simulate(a, b, x0, y0, T_end=15.0)

# 고품질 시각화를 위한 폰트 및 그리드 설정
plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.linestyle": "--"})
fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=150)

# 시계열 그래프
ax1.plot(t, x, label="Species 1 ($x$)", color='#1f77b4', linewidth=2)
ax1.plot(t, y, label="Species 2 ($y$)", linestyle="--", color='#d62728', linewidth=2)
ax1.set_title("Population Density Time Series")
ax1.set_xlabel("Time ($t$)")
ax1.set_ylabel("Density")
# 시계열 축 고정: 시간은 0~15, 밀도는 0~1.5
ax1.set_xlim(0, 15)
ax1.set_ylim(-0.05, 1.5)
ax1.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)