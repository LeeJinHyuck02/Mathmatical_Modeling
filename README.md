# 정당 지지율 동역학 및 생태계 경쟁 모델 시뮬레이션 (Mathmatical Modeling & PINN)

본 프로젝트는 크게 두 가지 갈래로 구성됩니다. 첫째, 로트카-볼테라(Lotka-Volterra) 경쟁 모델을 기반으로 한 수치 해석 및 신경망(PINN, DeepONet) 기반의 순방향 역학 시뮬레이션 연구입니다. 둘째, 이를 응용하여 실제 대한민국 정당 지지율 시계열 데이터를 수집/정제하고 역문제(Inverse Problem)를 풀어 미래 지지율 동역학을 분석하고 예측합니다.

## 🚀 프로젝트 실행 순서

1. 기초 수치 해석 및 시각화: RK4.py, graphForSeparatrix.py
2. 인터랙티브 시뮬레이터 실행: streamlit run src.py
3. DeepONet 학습 및 대시보드: DeepONetTraining.py 실행 후 streamlit run DeepONet.py
4. 정당 지지율 데이터 파이프라인: webcrawlingFromOriginal.py -> convertFromOriginalToText.py -> convertFromTextToCsv.py
5. 정당 지지율 모델링 (역문제): multi-start_L-BFGS-B.py 또는 PINNForFullData.py 등 실행

---

## 📂 파일별 핵심 역할 요약

### 1. 이론 및 기초 시뮬레이션 연구

#### 1.1 수학적 모델링 및 기초 수치 해석
* RK4.py: 룽게-쿠타 4차(RK4) 수치 적분을 활용하여 다양한 파라미터와 초기 조건에 대한 시계열 및 위상 평면 궤적을 고해상도로 렌더링하는 스크립트입니다.
* graphForSeparatrix.py: 쌍안정성 위상 조건에서 흡인 영역(Basin of Attraction)과 대수적 분리선(Separatrix)을 계산하여 초기 조건에 따른 수렴을 시각화하는 스크립트입니다.
* src.py: 수치 적분(RK4)을 기반으로 로트카-볼테라 모델의 파라미터 변화에 따른 역학을 Streamlit 웹 UI에서 실시간으로 시뮬레이션하는 대시보드 스크립트입니다.

#### 1.2 신경 연산자 (Neural Operators) 연구
* DeepONetTraining.py: DeepONet 아키텍처를 정의하고 RK4 기반 시뮬레이션 데이터로 오프라인 훈련을 수행하여 모델 가중치를 저장하는 스크립트입니다.
* DeepONet.py: 학습된 가중치(deeponet_weights.pth)를 로드하고 Streamlit을 통해 사용자가 초기 조건과 파라미터를 변경하며 실시간으로 딥러닝 추론 결과를 확인할 수 있는 웹 대시보드 스크립트입니다.

#### 1.3 기초 PINN (Physics-Informed Neural Network) 연구
* PINN.py: 소프트 제약조건(Soft Constraint) 기반의 PINN 모델을 Adam 옵티마이저로 훈련시키고 에포크별 예측 결과를 시각화하는 기본 실험 스크립트입니다.
* PINN_wrong.py: RK4 수치 적분으로 생성한 기준 데이터와 비교하여 기본 PINN 모델의 초기 학습 수렴 과정을 확인하고 시각화하는 시뮬레이션 스크립트입니다.
* PINNwithBetaSampling.py: 훈련 데이터 추출 시 균등 분포 대신 베타 분포를 사용하여 특정 시간대에 집중적으로 샘플링을 진행하는 적응형 PINN 훈련 스크립트입니다.

##텍스트에서 각 정당을 진보, 보수, 중도의 3대 이념으로 묶고 잔차 보간법을 적용해 최종# 2. 실제 데이터 응용: 정당 지지율 분석

#### 2.1 데이터 수집 및 전처리
* webcrawlingFromOriginal.py: 전국지표조사(NBS) 웹사이트를 크롤링하여 정당 지지율 여론조사 리포트 원문을 텍스트 파일로 다운로드하는 데이터 수집 스크립트입니다.
* convertFromOriginalToText.py: 수집된 여러 원본 텍스트 파일에서 날짜와 정당 지지율 정보만 추출한 뒤 시간순으로 오름차순 정렬하여 단일 문서로 병합하는 전처리 스크립트입니다.
* convertFromTextToCsv.py: 병합된  분석용 CSV 데이터를 생성하는 스크립트입니다.

#### 2.2 수치 최적화 기반 매개변수 추정 (Inverse Problem)
* multi-start_L-BFGS-B.py: 다중 시작(Multi-start) 기법과 L-BFGS-B 알고리즘을 사용해 3종 로트카-볼테라 역학 모델의 최적 파라미터를 추정하고 미래 지지율을 예측하는 스크립트입니다.
* multi-start_L-BFGS-B_wrongModeling.py: 진보와 보수 간의 직접적인 상호작용을 배제한 선형 위상(Linear Topology) 모델의 한계와 잘못된 가정이 예측에 미치는 영향을 확인하기 위한 비교 분석용 스크립트입니다.

#### 2.3 응용 PINN (Political PINN)
* PINNForMovingAverageData.py: 블록 평균(Block Averaging)으로 노이즈를 완화한 부분 통합 데이터를 활용하여 물리 정보 신경망(PINN)을 학습시키고 동역학을 예측하는 스크립트입니다.
* PINNForFullData.py: 이동평균으로 평활화된 전체 시계열 데이터를 대상으로 Adam 옵티마이저만 사용하여 통계적 중심점 기반의 단일 PINN 모델을 훈련하는 스크립트입니다.
* PINNForPartialDataFromidx.py: 지지율 총합 100% 보존 물리 제약과 2단계 하이브리드 최적화(Adam + L-BFGS)를 적용하여 최신 구간의 데이터 흐름에 맞춰 PINN을 정밀 학습하는 스크립트입니다.