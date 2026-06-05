# 🚀 Mathmatical Modeling Project

본 프로젝트는 크게 두 가지 갈래로 구성됩니다. 첫째, 수치 해석(RK4) 및 신경망(PINN, DeepONet) 로트카-볼테라(Lotka-Volterra) 경쟁 모델을 시뮬레이션합니다. 둘째, 이를 응용하여 실제 대한민국 정당 지지율 시계열 데이터를 수집/정제하고 역문제(Inverse Problem)를 풀어 미래 지지율 동역학을 분석하고 예측합니다.

## 📂 파일별 핵심 역할 요약

### 1. 이론 및 기초 시뮬레이션 연구

#### 1.1 수학적 모델링 및 기초 수치 해석
* RK4.py: 룽게-쿠타 4차(RK4) 수치 적분을 활용하여 다양한 파라미터와 초기 조건에 대한 시계열 및 위상 평면 궤적을 렌더링하는 스크립트입니다.
* graphForSeparatrix.py: 쌍안정성 위상 조건에서 흡인 영역(Basin of Attraction)과 대수적 분리선(Separatrix)을 계산하여 초기 조건에 따른 수렴을 시각화하는 스크립트입니다.
* src.py: 수치 적분(RK4)을 기반으로 로트카-볼테라 모델의 파라미터 변화에 따른 역학을 Streamlit 웹 UI에서 실시간으로 시뮬레이션하는 대시보드 스크립트입니다.

#### 1.2 신경 연산자 (Neural Operators) 연구
* DeepONetTraining.py: DeepONet 아키텍처를 정의하고 RK4 기반 시뮬레이션 데이터로 오프라인 훈련을 수행하여 모델 가중치를 저장하는 스크립트입니다.
* DeepONet.py: 학습된 가중치(deeponet_weights.pth)를 로드하여 DeepONet을 기반으로 로트카-볼테라 모델의 파라미터 변화에 따른 역학을 Streamlit 웹 UI에서 실시간으로 시뮬레이션하는 대시보드 스크립트입니다.

#### 1.3 기초 PINN (Physics-Informed Neural Network) 연구
* PINNwithBetaSampling.py: 시간 정규화 + 중요도 샘플링 기반 PINN 훈련 스크립트입니다.
* PINN.py: 시간 정규화 기반 PINN 훈련 스크립트입니다.
* PINN_wrong.py: PINN 모델을 Adam 옵티마이저로 훈련시키고 에포크별 예측 결과를 시각화하는 기본 실험 스크립트입니다.

### 2. 실제 데이터 활용 연구

#### 2.1 데이터 수집 및 전처리
* webcrawlingFromOriginal.py: 전국지표조사(NBS) 웹사이트를 크롤링하여 정당 지지율 여론조사 리포트 원문을 텍스트 파일로 다운로드하는 데이터 수집 스크립트입니다.
* convertFromOriginalToText.py: 수집된 여러 원본 텍스트 파일에서 날짜와 정당 지지율 정보만 추출한 뒤 시간순으로 오름차순 정렬하여 단일 문서로 병합하는 전처리 스크립트입니다.
* convertFromTextToCsv.py: 병합된  분석용 CSV 데이터를 생성하는 스크립트입니다.

#### 2.2 수치 최적화 기반 매개변수 추정 (Inverse Problem)
* multi-start_L-BFGS-B.py: 다중 시작(Multi-start) 기법과 L-BFGS-B 알고리즘을 사용해 3종 로트카-볼테라 역학 모델의 최적 파라미터를 추정하고 미래 지지율을 예측하는 스크립트입니다.
* multi-start_L-BFGS-B_wrongModeling.py: 진보와 보수 간의 직접적인 상호작용을 배제한 선형 위상(Linear Topology) 모델의 한계와 잘못된 가정이 예측에 미치는 영향을 확인하기 위한 비교 분석용 스크립트입니다.

#### 2.3 응용 PINN (Political PINN)
* PINNForMovingAverageData.py: 블록 평균(Block Averaging)으로 노이즈를 완화한 부분 통합 데이터를 활용하여 PINN 모델을 훈련하는 스크립트입니다.
* PINNForPartialDataFromidx.py:  부분 시계열 데이터를 대상으로 PINN 모델을 훈련하는 스크립트입니다.
* PINNForFullData.py: 전체 시계열 데이터를 대상으로 Adam 옵티마이저만 사용하여  단일 PINN 모델을 훈련하는 스크립트입니다.