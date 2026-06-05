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
# 2. 로트카-볼테라 3종 경쟁 모델 및 기초 목적 함수
# =========================================================
def lotka_volterra_3species(x, t, r, K, alpha):
    x1, x2, x3 = max(0, x[0]), max(0, x[1]), max(0, x[2])

    dx1dt = r[0] * x1 * (1 - (x1 + alpha[0,1]*x2 + alpha[0,2]*x3) / K[0])
    dx2dt = r[1] * x2 * (1 - (alpha[1,0]*x1 + x2 + alpha[1,2]*x3) / K[1])
    dx3dt = r[2] * x3 * (1 - (alpha[2,0]*x1 + alpha[2,1]*x2 + x3) / K[2])

    return [dx1dt, dx2dt, dx3dt]

def objective_function(params, t_data, true_data, x0):
    r = params[0:3]
    K = params[3:6]
    alpha = np.array([
        [1.0, params[6], params[7]],
        [params[8], 1.0, params[9]],
        [params[10], params[11], 1.0]
    ])

    try:
        # ODE 수치 적분을 통한 가상 데이터 생성
        pred_data = odeint(lotka_volterra_3species, x0, t_data, args=(r, K, alpha))
    except:
        return 1e10

    # 기울기 매칭 없이 순수하게 관측치와의 절대 오차(MSE)만 반환
    mse = np.mean((true_data - pred_data)**2)
    return mse

# =========================================================
# 3. 다중 시작 (Multi-start) 기반 최적화 로직
# =========================================================
def generate_random_guess(bounds):
    """지정된 제약 공간 내에서 무작위 초기값을 생성함"""
    return np.array([np.random.uniform(low, high) for low, high in bounds])

def run_multistart_lbfgsb(objective_func, bounds, args_tuple, num_restarts=5):
    best_loss = float('inf')
    best_params = None
    best_iter = 0

    print(f"다중 시작(Multi-start) L-BFGS-B 탐색 시작 (총 {num_restarts}회 수행)...")

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
        print(f" -> [{i+1:02d}/{num_restarts}] 신규 파라미터 확보 (MSE Loss: {result.fun:.6f})")
        # 기존 발견된 최저 오차보다 낮을 경우 갱신
        if result.success and result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x
            best_iter = i + 1
            print(f" -> [{i+1:02d}/{num_restarts}] 신규 최적 파라미터 확보 (MSE Loss: {best_loss:.6f})")

    print(f"\n[탐색 완료] {best_iter}번째 시도 파라미터 최종 채택 (최종 Loss: {best_loss:.6f})")
    return best_params

# =========================================================
# 4. 메인 실행부 및 시각화
# =========================================================
def forecast_political_dynamics_base_multistart(future_steps=52):
    if not os.path.exists(INPUT_CSV):
        print(f"[오류] '{INPUT_CSV}' 파일 부재.")
        return

    df = pd.read_csv(INPUT_CSV)

    # 평활화(Rolling) 없이 원시 데이터(Raw data) 직접 사용
    target_data = df[["진보_합계", "보수_합계", "중도_합계"]].dropna().values

    t_historical = np.arange(len(target_data))
    x0 = target_data[0]

    # 파라미터 탐색 제약 공간 (r, K, alpha)
    bounds = [
        (0.001, 1.0), (0.001, 1.0), (0.001, 1.0),
        (10.0, 100.0), (10.0, 100.0), (10.0, 100.0),
        (0.0, 3.0), (0.0, 3.0), (0.0, 3.0),
        (0.0, 3.0), (0.0, 3.0), (0.0, 3.0)
    ]

    args_for_objective = (t_historical, target_data, x0)

    # 다중 시작 최적화기 호출
    opt_params = run_multistart_lbfgsb(
        objective_func=objective_function,
        bounds=bounds,
        args_tuple=args_for_objective,
        num_restarts=20
    )

    if opt_params is None:
        print("[경고] 유효한 파라미터를 찾지 못함.")
        return

    # 최적 파라미터 해체
    opt_r = opt_params[0:3]
    opt_K = opt_params[3:6]
    opt_alpha = np.array([
        [1.0, opt_params[6], opt_params[7]],
        [opt_params[8], 1.0, opt_params[9]],
        [opt_params[10], opt_params[11], 1.0]
    ])

    # 시간축 확장 및 역학 시뮬레이션 외삽
    t_extended = np.arange(len(target_data) + future_steps)
    simulated_data = odeint(lotka_volterra_3species, x0, t_extended, args=(opt_r, opt_K, opt_alpha))

    # =========================================================
    # 5. 시계열 외삽 예측 시각화
    # =========================================================
    plt.figure(figsize=(16, 8))

    # 과거 관측치 (산점도)
    plt.plot(t_historical, target_data[:, 0], 'o', color='royalblue', alpha=0.3, label="실제 데이터 (진보)")
    plt.plot(t_historical, target_data[:, 1], 's', color='firebrick', alpha=0.3, label="실제 데이터 (보수)")
    plt.plot(t_historical, target_data[:, 2], '^', color='darkorange', alpha=0.3, label="실제 데이터 (중도)")

    colors = ['blue', 'red', 'darkorange']
    labels = ['진보', '보수', '중도']

    # 피팅 및 예측 곡선
    for i in range(3):
        plt.plot(t_historical, simulated_data[:len(t_historical), i], '-', color=colors[i], linewidth=2.5, label=f"피팅 ({labels[i]})")
        plt.plot(t_extended[len(t_historical):], simulated_data[len(t_historical):, i], '--', color=colors[i], linewidth=3, label=f"예측 ({labels[i]})")

    # 현재 시점 분리선
    current_t = len(t_historical) - 1
    plt.axvline(x=current_t, color='black', linestyle='-.', linewidth=2, label="현재 시점")
    plt.text(current_t + 2, 95, f"미래 예측 구간 (+{future_steps}주)", fontsize=12)

    plt.title("기초 로트카-볼테라 모델 및 다중 시작(Multi-start) 기법을 활용한 외삽 예측", fontsize=18, pad=20)
    plt.xlabel("조사 차수 누적 경과 (Time steps)", fontsize=12)
    plt.ylabel("이념별 지지율 (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    forecast_political_dynamics_base_multistart(future_steps=80)