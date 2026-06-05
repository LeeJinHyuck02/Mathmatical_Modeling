import os
import re
import pandas as pd

# =========================================================
# 1. 환경 설정
# =========================================================
INPUT_FILE = "merged_party_approval.txt"
OUTPUT_CSV = "ideology_approval_3way.csv"

# =========================================================
# 2. 통합 파서 및 잔차 보간 로직 적용
# =========================================================
def create_3way_ideology_csv():
    if not os.path.exists(INPUT_FILE):
        print(f"[오류] '{INPUT_FILE}' 파일을 찾을 수 없음.")
        return None

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.findall(r"\[(.*?)\]\n▶(.*?)(?=\n\[|\Z)", content, re.DOTALL)
    data_records = []

    for date_str, text_val in blocks:
        date_str = date_str.strip()
        text_val = text_val.strip()

        if "미상" in date_str or "실패" in text_val:
            continue

        record = {
            "Date": date_str,
            "더불어민주당": 0, "조국혁신당": 0, "정의당계": 0, "기타진보": 0,
            "국민의힘": 0,
            "개혁신당": 0, "새로운미래": 0, "국민의당": 0,
            "태도유보": 0
        }

        # '%' 기호 기준 청크 분할 및 병렬 할당
        chunks = text_val.split('%')

        for chunk in chunks:
            if not chunk.strip():
                continue

            num_match = re.search(r"(\d+)", chunk)
            if num_match:
                val = int(num_match.group(1))

                if "더불어민주당" in chunk: record["더불어민주당"] = val
                if "조국혁신당" in chunk: record["조국혁신당"] = val
                if "정의당" in chunk or "녹색정의당" in chunk: record["정의당계"] = val
                if "진보당" in chunk or "열린민주당" in chunk: record["기타진보"] = val

                if "국민의힘" in chunk: record["국민의힘"] = val

                if "개혁신당" in chunk: record["개혁신당"] = val
                if "새로운미래" in chunk: record["새로운미래"] = val
                if "국민의당" in chunk: record["국민의당"] = val

                chunk_no_space = chunk.replace(" ", "")
                if "태도유보" in chunk_no_space or "무응답" in chunk_no_space:
                    record["태도유보"] = val

        # 1차 파생 변수 연산
        record["진보_합계"] = record["더불어민주당"] + record["조국혁신당"] + record["정의당계"] + record["기타진보"]
        record["보수_합계"] = record["국민의힘"]

        # [핵심 로직] 잔차 보간법 (Residual Imputation)
        # 태도유보가 텍스트에 기재되지 않아 0으로 추출되었으나, 진보와 보수 지지율이 존재하는 경우
        party_sum = record["진보_합계"] + record["보수_합계"] + record["개혁신당"] + record["새로운미래"] + record["국민의당"]

        if record["태도유보"] == 0 and party_sum > 0:
            # 100%에서 명시된 정당 지지율 합을 뺀 나머지를 태도유보로 간주 (최소값 0 보장)
            record["태도유보"] = max(0, 100 - party_sum)

        # 보정된 태도유보를 바탕으로 중도_합계 최종 연산
        record["중도_합계"] = record["개혁신당"] + record["새로운미래"] + record["국민의당"] + record["태도유보"]

        # 유효 데이터 판별
        if record["더불어민주당"] > 0 or record["국민의힘"] > 0:
            data_records.append(record)

    df = pd.DataFrame(data_records)

    # 3대 이념 체계로 컬럼 재배치
    cols = [
        "Date",
        "진보_합계", "보수_합계", "중도_합계",
        "더불어민주당", "조국혁신당", "정의당계", "기타진보",
        "국민의힘",
        "개혁신당", "새로운미래", "국민의당", "태도유보"
    ]
    df = df[cols]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[성공] 잔차 보간이 적용된 3대 이념 분류 데이터 {len(df)}행을 '{OUTPUT_CSV}' 파일로 출력함.")

    return df

# =========================================================
# 4. 메인 실행 프로세스
# =========================================================
if __name__ == "__main__":
    ideology_df = create_3way_ideology_csv()

    if ideology_df is not None:
        print("\n[출력 데이터셋 미리보기]")
        print(ideology_df[["Date", "진보_합계", "보수_합계", "중도_합계", "태도유보"]].head())