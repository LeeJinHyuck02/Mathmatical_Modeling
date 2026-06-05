import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import odeint

# 1. 생태학적 매개변수 설정 (쌍안정성 위상)
a, b_param = 1.8, 1.5

# 2. 로트카-볼테라 미분방정식 정의
def lotka_volterra(Y, t):
    x, y = Y
    return [x * (1 - x - a * y), y * (1 - y - b_param * x)]

# 3. 위상 공간 격자(Grid) 생성 (상한 1.0으로 설정)
grid_points = 150
x_vals = np.linspace(0.01, 1.0, grid_points)
y_vals = np.linspace(0.01, 1.0, grid_points)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

# 4. 각 초기 조건에 대한 수치적분 및 수렴 지점 판별
t_forward = np.linspace(0, 1000, 200)

print("수치해석을 통한 흡인 영역(Basin of Attraction) 계산 중...")
for i in range(grid_points):
    for j in range(grid_points):
        ic = [X[i, j], Y[i, j]]
        traj = odeint(lotka_volterra, ic, t_forward)
        final_x, final_y = traj[-1]

        # 종 1(x)이 지배하는 평형점으로 수렴하는 경우 (Z = 1, coolwarm에서 붉은색)
        if final_x > final_y:
            Z[i, j] = 1
        # 종 2(y)가 지배하는 평형점으로 수렴하는 경우 (Z = -1, coolwarm에서 푸른색)
        else:
            Z[i, j] = -1

# 5. 안장점 좌표 계산
x_star = (1 - a) / (1 - a * b_param)
y_star = (1 - b_param) / (1 - a * b_param)

# 6. 직선 x0 + y0 = 1.0 및 대수적 분리선(Separatrix) 데이터 생성
x_line = np.linspace(0, 1.0, 100)
y_line = 1.0 - x_line

# 대수적 분리선 y = ((b-1)/(a-1)) * x 계산
k_slope = (b_param - 1) / (a - 1)
y_separatrix = k_slope * x_line

# 7. 시각화 (그래프 렌더링)
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

# 흡인 영역을 등고선 색상 플롯으로 렌더링
mesh = ax.pcolormesh(X, Y, Z, cmap='coolwarm', alpha=0.4, shading='auto')

# 직선 x0 + y0 = 1.0 렌더링
ax.plot(x_line, y_line, 'k--', linewidth=2, label="$x_0 + y_0 = 1.0$")

# 대수적 분리선 기준선 렌더링 (녹색 실선)
ax.plot(x_line, y_separatrix, 'g-', linewidth=2.5, label=r"Analytical Separatrix $y = \frac{b-1}{a-1}x$")

# 안장점 마커 렌더링
ax.plot(x_star, y_star, '^', color='yellow', markersize=12, markeredgecolor='k', label="Saddle Point ($x^*, y^*$)")

# 붉은색 및 푸른색 영역의 의미를 나타내는 커스텀 패치 생성
red_patch = mpatches.Patch(color='#b40426', alpha=0.4, label="Species 1 Domination, $(1, 0)$")
blue_patch = mpatches.Patch(color='#3b4cc0', alpha=0.4, label="Species 2 Domination, $(0, 1)$")

# 원래 존재하던 선/마커 범례 핸들을 가져옵니다.
handles, labels = ax.get_legend_handles_labels()

# 커스텀 패치를 범례 핸들 리스트에 추가합니다.
handles.extend([red_patch, blue_patch])

# 축 설정 및 디자인 (상한 1.0으로 제한)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0) # y_separatrix의 1.0 초과 값은 자동으로 잘림 처리됩니다.
ax.set_xlabel("Initial Species 1 Population ($x_0$)", fontsize=12)
ax.set_ylabel("Initial Species 2 Population ($y_0$)", fontsize=12)

# 업데이트된 핸들 리스트로 범례를 표시합니다.
ax.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()