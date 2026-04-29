import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="Lotka-Volterra Simulator", layout="wide")
st.title("Competition of Two Species: Interactive Simulator")

# 2. 사이드바 - 매개변수 및 초기 조건 설정
st.sidebar.header("Model Parameters")
a = st.sidebar.slider("Parameter $a$ (Species 2 on 1)", 0.1, 2.0, 1.5, 0.05)
b = st.sidebar.slider("Parameter $b$ (Species 1 on 2)", 0.1, 2.0, 0.5, 0.05)

st.sidebar.header("Initial Conditions")
x0 = st.sidebar.slider("Initial $x_0$ (Species 1)", 0.01, 1.0, 0.2, 0.01)
y0 = st.sidebar.slider("Initial $y_0$ (Species 2)", 0.01, 1.0, 0.8, 0.01)

# 3. RK4 알고리즘 엔진
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

# 4. 결과 계산 및 시각화
t, x, y = simulate(a, b, x0, y0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 시계열 그래프
ax1.plot(t, x, label="Species 1 ($x$)")
ax1.plot(t, y, label="Species 2 ($y$)", linestyle="--")
ax1.set_title("Population Density Time Series")
ax1.legend()

# 위상 평면
ax2.plot(x, y, color="green")
ax2.scatter([x0], [y0], color="blue", label="Start")
ax2.scatter([x[-1]], [y[-1]], color="black", marker="x", label="End")
ax2.set_title("Phase Portrait")
ax2.legend()

st.pyplot(fig)