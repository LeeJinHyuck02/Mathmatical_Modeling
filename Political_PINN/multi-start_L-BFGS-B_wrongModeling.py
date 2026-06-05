import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.integrate import odeint
from scipy.optimize import minimize

# =========================================================
# 1. 시각화 폰트 및 환경 설정
# =========================================================
font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

INPUT_CSV = "ideology_approval_3way.csv"

# =========================================================
# 2. 선형 위상 (진보 <-> 중도 <-> 보수) 로트카-볼테라 모델
# =========================================================
def lotka_volterra_linear_topology(x, t, r, K, alpha):
    # x[0]: 진보, x[1]: 보수, x[2]: 중도
    x0, x1, x2 = max(0, x[0]), max(0, x[1]), max(0, x[2])

    # alpha 배열 (총 4개): [a02, a12, a20, a21]
    # a02: 중도가 진보를 억제 / a12: 중도가 보수를 억제
    # a20: 진보가 중도를 억제 / a21: 보수가 중도를 억제
    a02, a12, a20, a21 = alpha

    # [수정] 진보(x0)와 보수(x1) 간의 직접 교차항 삭제
    dx0dt = r[0] * x0 * (1 - (x0 + a02 * x2) / K[0])
    dx1dt = r[1] * x1 * (1 - (x1 + a12 * x2) / K[1])
    dx2dt = r[2] * x2 * (1 - (a20 * x0 + a21 * x1 + x2) / K[2])

    return [dx0dt, dx1dt, dx2dt]

def objective_function(params, t_data, true_data, x_init):
    # 파라미터 해체 (총 10차원)
    r = params[0:3]
    K = params[3:6]
    alpha = params[6:10]

    try:
        pred_data = odeint(lotka_volterra_linear_topology, x_init, t_data, args=(r, K, alpha))
    except:
        return 1e10

    # 순수 절대 오차(MSE)만 산출
    mse = np.mean((true_data - pred_data)**2)
    return mse

# =========================================================
# 3. 10차원 공간 다중 시작 (Multi-start) 최적화 로직
# =========================================================
def generate_random_guess(bounds):
    return np.array([np.random.uniform(low, high) for low, high in bounds])

def run_multistart_lbfgsb_linear(objective_func, bounds, args_tuple, num_restarts=25):
    best_loss = float('inf')
    best_params = None
    best_iter = 0

    print(f"선형 위상 모델 다중 시작 탐색 (총 {num_restarts}회 수행)...")

    for i in range(num_restarts):
        initial_guess = generate_random_guess(bounds)

        result = minimize(
            objective_func,
            initial_guess,
            args=args_tuple,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )
        print(f" -> [{i+1:02d}/{num_restarts}] 신규 파라미터 갱신 (Loss: {result.fun:.6f})")
        if result.success and result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x
            best_iter = i + 1
            print(f" -> [{i+1:02d}/{num_restarts}] 신규 최적 파라미터 갱신 (Loss: {best_loss:.6f})")

    print(f"\n[완료] {best_iter}번째 탐색 결과 최종 채택 (최종 Loss: {best_loss:.6f})")
    return best_params

# =========================================================
# 4. 데이터 로드 및 실행부
# =========================================================
def forecast_linear_topology_dynamics(future_steps=52):
    if not os.path.exists(INPUT_CSV):
        print(f"[오류] '{INPUT_CSV}' 파일 부재.")
        return

    df = pd.read_csv(INPUT_CSV)

    # 평활화 배제, 원시 데이터 직접 사용
    target_data = df[["진보_합계", "보수_합계", "중도_합계"]].dropna().values

    t_historical = np.arange(len(target_data))
    x_init = target_data[0]

    # [수정] 파라미터 제약 공간 10차원으로 축소 (r:3, K:3, alpha:4)
    bounds = [
        (0.001, 1.0), (0.001, 1.0), (0.001, 1.0),       # r_0, r_1, r_2
        (10.0, 100.0), (10.0, 100.0), (10.0, 100.0),    # K_0, K_1, K_2
        (0.0, 3.0), (0.0, 3.0), (0.0, 3.0), (0.0, 3.0)  # a02, a12, a20, a21
    ]

    args_for_objective = (t_historical, target_data, x_init)

    opt_params = run_multistart_lbfgsb_linear(
        objective_func=objective_function,
        bounds=bounds,
        args_tuple=args_for_objective,
        num_restarts=15
    )

    if opt_params is None:
        return

    opt_r = opt_params[0:3]
    opt_K = opt_params[3:6]
    opt_alpha = opt_params[6:10]

    t_extended = np.arange(len(target_data) + future_steps)
    simulated_data = odeint(lotka_volterra_linear_topology, x_init, t_extended, args=(opt_r, opt_K, opt_alpha))

    # =========================================================
    # 5. 시각화
    # =========================================================
    plt.figure(figsize=(16, 8))

    plt.plot(t_historical, target_data[:, 0], 'o', color='royalblue', alpha=0.3, label="실제 데이터 (진보)")
    plt.plot(t_historical, target_data[:, 1], 's', color='firebrick', alpha=0.3, label="실제 데이터 (보수)")
    plt.plot(t_historical, target_data[:, 2], '^', color='darkorange', alpha=0.3, label="실제 데이터 (중도)")

    colors = ['blue', 'red', 'darkorange']
    labels = ['진보', '보수', '중도']

    for i in range(3):
        plt.plot(t_historical, simulated_data[:len(t_historical), i], '-', color=colors[i], linewidth=2.5, label=f"피팅 ({labels[i]})")
        plt.plot(t_extended[len(t_historical):], simulated_data[len(t_historical):, i], '--', color=colors[i], linewidth=3, label=f"예측 ({labels[i]})")

    current_t = len(t_historical) - 1
    plt.axvline(x=current_t, color='black', linestyle='-.', linewidth=2, label="현재 시점")
    plt.text(current_t + 2, 95, f"미래 예측 구간 (+{future_steps}주)", fontsize=12)

    plt.title("선형 위상(진보-중도-보수) 모델 및 다중 시작 최적화를 활용한 외삽 예측", fontsize=18, pad=20)
    plt.xlabel("조사 차수 누적 경과 (Time steps)", fontsize=12)
    plt.ylabel("지지도 (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    forecast_linear_topology_dynamics(future_steps=52)