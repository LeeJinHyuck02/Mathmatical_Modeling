import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. 고해상도 학술용 그래프 환경 설정
plt.rcParams.update({
    "font.family": "serif",        
    "font.size": 11,               
    "axes.labelsize": 12,          
    "axes.titlesize": 13,          
    "xtick.labelsize": 10,         
    "ytick.labelsize": 10,         
    "legend.fontsize": 9, 
    "lines.linewidth": 1.8,        
    "figure.dpi": 300,             
    "axes.grid": True,             
    "grid.alpha": 0.4,             
    "grid.linestyle": "--"         
})

# 2. 미분 방정식 및 RK4 알고리즘 정의
def f(x, y, a): return x * (1 - x - a * y)
def g(x, y, b): return y * (1 - y - b * x)

def rk4_step(xn, yn, h, a, b):
    k1x = h * f(xn, yn, a)
    k1y = h * g(xn, yn, b)
    k2x = h * f(xn + k1x/2, yn + k1y/2, a)
    k2y = h * g(xn + k1x/2, yn + k1y/2, b)
    k3x = h * f(xn + k2x/2, yn + k2y/2, a)
    k3y = h * g(xn + k2x/2, yn + k2y/2, b)
    k4x = h * f(xn + k3x, yn + k3y, a)
    k4y = h * g(xn + k3x, yn + k3y, b)
    
    x_next = xn + (k1x + 2*k2x + 2*k3x + k4x) / 6
    y_next = yn + (k1y + 2*k2y + 2*k3y + k4y) / 6
    return x_next, y_next

# 3. 시뮬레이션 초기 조건 설정
T_end = 50.0
h = 0.05
num_steps = int(T_end / h)
t_vals = np.linspace(0, T_end, num_steps)

cases = [(1.5, 0.5), (0.5, 1.5), (0.5, 0.5)]
initial_conditions = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]
colors = ['#1f77b4', '#d62728', '#2ca02c']  

# 4. 2행 3열 도표 렌더링 설정 (수정된 부분)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 종 구분을 위한 가상 선 객체
species_legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='Species 1 ($x$)'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='Species 2 ($y$)')
]

for idx, (a, b) in enumerate(cases):
    # 인덱싱 역전: 1행(0)은 시계열, 2행(1)은 위상 평면
    ax_time = axes[0, idx]
    ax_phase = axes[1, idx]
    
    title_str = f"Case {idx+1}: $a={a}, b={b}$"
    ax_time.set_title(title_str)
    
    ax_time.set_xlabel("Time ($t$)")
    ax_phase.set_xlabel("Species 1 ($x$)")
    
    # 첫 번째 열(왼쪽)에만 y축 레이블 부착
    if idx == 0:
        ax_time.set_ylabel("Population Density")
        ax_phase.set_ylabel("Species 2 ($y$)")
        
    ic_legend_elements = [] 
    
    for ic_idx, (x0, y0) in enumerate(initial_conditions):
        x_vals = np.zeros(num_steps)
        y_vals = np.zeros(num_steps)
        x_vals[0], y_vals[0] = x0, y0
        
        for i in range(1, num_steps):
            x_vals[i], y_vals[i] = rk4_step(x_vals[i-1], y_vals[i-1], h, a, b)
            
        # [1행] 시계열 그래프
        ax_time.plot(t_vals, x_vals, color=colors[ic_idx], linestyle='-')
        ax_time.plot(t_vals, y_vals, color=colors[ic_idx], linestyle='--', alpha=0.8)
                     
        # [2행] 위상 평면 궤적
        ax_phase.plot(x_vals, y_vals, color=colors[ic_idx], linestyle='-')
        ax_phase.scatter([x0], [y0], color=colors[ic_idx], marker='o', s=40)
        ax_phase.scatter([x_vals[-1]], [y_vals[-1]], color=colors[ic_idx], marker='x', s=60)
        
        ic_legend_elements.append(
            Line2D([0], [0], color=colors[ic_idx], lw=2, label=f"IC: ({x0}, {y0})")
        )

    # 평형점 마킹 로직
    if a * b != 1.0:
        x_eq = (1 - a) / (1 - a * b)
        y_eq = (1 - b) / (1 - a * b)
        if x_eq >= 0 and y_eq >= 0:
            ax_phase.scatter([x_eq], [y_eq], color='black', marker='*', s=150, zorder=5, label="Equilibrium")

    # 세 번째 열(오른쪽 끝)에만 범례를 추가하여 각 도표의 데이터 가려짐 최소화
    if idx == 2:
        legend_ic = ax_time.legend(handles=ic_legend_elements, loc='upper right', title="Initial Conditions")
        ax_time.add_artist(legend_ic) 
        ax_time.legend(handles=species_legend_elements, loc='center right', title="Species")
        ax_phase.legend(handles=ic_legend_elements, loc='upper right')

# 행/열 사이 간격 및 전체 여백 자동 최적화
plt.tight_layout()
plt.savefig("lotka_volterra_rk4_2x3_layout.pdf", format="pdf", bbox_inches='tight')
plt.show()