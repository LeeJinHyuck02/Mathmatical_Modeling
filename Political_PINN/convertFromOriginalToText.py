import os
import re
from tqdm.notebook import tqdm

# =========================================================
# 1. 환경 설정
# =========================================================
INPUT_DIR = "nbs_raw_text"
OUTPUT_FILE = "merged_party_approval.txt"

# '2025년 2월 3주' 형태에서 연도와 월을 추출하는 정규표현식 패턴
DATE_PATTERN = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")

# =========================================================
# 2. 데이터 추출 및 정제 함수
# =========================================================
def extract_and_merge_data_optimized():
    if not os.path.exists(INPUT_DIR):
        print(f"[오류] '{INPUT_DIR}' 폴더가 존재하지 않음.")
        return

    file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    if not file_list:
        print("[경고] 폴더 내에 텍스트 파일이 없음.")
        return

    extracted_data = []

    for filename in tqdm(file_list, desc="정당지지도 정밀 추출"):
        filepath = os.path.join(INPUT_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        year, month = 0, 0
        date_str = "날짜 미상"

        # [STEP 1] 날짜 및 주차 추출 (리포트 상단부 30줄 이내 집중 탐색)
        for line in lines[:30]:
            match = DATE_PATTERN.search(line)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))

                # 주차 정보 추가 적출 시도 (예: '3주')
                week_match = re.search(r"(\d{1,2})\s*주", line)
                week_str = f" {week_match.group(1)}주" if week_match else ""

                # 포맷팅: '2025년 02월 3주'
                date_str = f"{year}년 {month:02d}월{week_str}"
                break

        # [STEP 2] 정당지지도 데이터 정밀 추출
        approval_text = "데이터 추출 실패 (구조 불일치)"

        for i, line in enumerate(lines):
            clean_line = line.strip()

            # 정확한 기준점(Anchor) 탐지
            if clean_line == "정당지지도":

                # 기준점 하위 4줄 이내에서 실제 수치 데이터가 있는 라인 탐색
                for j in range(i + 1, min(i + 5, len(lines))):
                    target_line = lines[j].strip()

                    # '%' 기호가 포함되거나 수치가 나열된 줄을 실제 데이터로 판별
                    if target_line and "%" in target_line:
                        # 불필요한 시작 특수기호(–, -) 제거 및 공백 정리
                        approval_text = target_line.lstrip("–- ").strip()
                        break
                break

        extracted_data.append({
            "year": year,
            "month": month,
            "date_str": date_str,
            "filename": filename,
            "text": approval_text
        })

    # =========================================================
    # 3. 시간순 정렬 및 최종 파일 병합
    # =========================================================
    # 연도, 월, 그리고 주차 문자열을 기준으로 완벽한 오름차순(과거->최신) 시계열 정렬
    extracted_data.sort(key=lambda x: (x["year"], x["month"], x["date_str"]))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("=" * 80 + "\n")
        out_f.write("전국지표조사(NBS) 정당지지도 시계열 통합 리포트\n")
        out_f.write("=" * 80 + "\n\n")

        for data in extracted_data:
            out_f.write(f"[{data['date_str']}]\n")
            out_f.write(f"▶ {data['text']}\n\n")

    print(f"\n[성공] 총 {len(extracted_data)}개의 원본 문서에서 데이터를 적출하여 단일 리포트로 병합함.")
    print(f"저장된 파일 위치: Colab 환경 내 '{OUTPUT_FILE}'")

# =========================================================
# 4. 메인 실행 프로세스
# =========================================================
if __name__ == "__main__":
    extract_and_merge_data_optimized()