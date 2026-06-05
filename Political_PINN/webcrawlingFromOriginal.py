import os
import re
import time
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# =========================================================
# 1. 환경 설정 및 신형 Selenium 드라이버 초기화
# =========================================================
BASE_URL = "http://nbsurvey.kr"
START_PAGE = 11
END_PAGE = 45
SAVE_DIR = "nbs_raw_text"

os.makedirs(SAVE_DIR, exist_ok=True)

def get_chrome_driver():
    options = Options()
    # 신형 헤드리스 모드 활성화 (봇 탐지 회피력 강화)
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')

    # 봇 탐지 회피용 헤더
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    # Service 경로 할당 로직 제거: Selenium Manager가 Chrome 버전에 맞는 드라이버를 자동 탐색 및 구동함
    driver = webdriver.Chrome(options=options)
    return driver

# =========================================================
# 2. 브라우저 시뮬레이션 기반 링크 조립
# =========================================================
def collect_archive_links_selenium(start_page, end_page):
    archive_links = []
    driver = get_chrome_driver()

    for page in tqdm(range(start_page, end_page + 1), desc="목록 페이지 렌더링"):
        page_url = f"{BASE_URL}/page/{page}"

        try:
            driver.get(page_url)
            # 서버 방화벽 우회 및 JS 렌더링 대기
            time.sleep(3.0)

            html_text = driver.page_source
            found_numbers = set(re.findall(r"post-(\d+)", html_text))

            page_found_count = 0
            for post_number in found_numbers:
                full_url = f"{BASE_URL}/archives/{post_number}"
                if full_url not in archive_links:
                    archive_links.append(full_url)
                    page_found_count += 1

            print(f" - {page_url} : {page_found_count}개 조립")

        except Exception as e:
            print(f"\n[오류] {page_url} 렌더링 중 문제 발생: {e}")

    driver.quit()
    return archive_links

# =========================================================
# 3. 본문 텍스트 저장
# =========================================================
def sanitize_filename(name):
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    return name[:150]

def save_report_text_selenium(url, driver):
    try:
        driver.get(url)
        time.sleep(2.0)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "untitled"
        title = sanitize_filename(title)

        text = soup.get_text(separator="\n", strip=True)

        archive_id_match = re.search(r"/archives/(\d+)", url)
        archive_id = archive_id_match.group(1) if archive_id_match else "unknown"

        filename = f"{archive_id}_{title}.txt"
        filepath = os.path.join(SAVE_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n")
            f.write(f"TITLE: {title}\n")
            f.write("-" * 50 + "\n")
            f.write(text)

        return True
    except Exception:
        return False

# =========================================================
# 4. 메인 실행 프로세스
# =========================================================
if __name__ == "__main__":
    print("--- 신형 브라우저 시뮬레이션 기반 수집 시작 ---")
    archive_urls = collect_archive_links_selenium(START_PAGE, END_PAGE)

    print(f"\n[확인] 총 {len(archive_urls)}개의 리포트 타깃 URL을 조립함.")

    if archive_urls:
        print("\n--- 데이터 저장 단계 시작 ---")
        success = 0
        main_driver = get_chrome_driver()

        for url in tqdm(archive_urls, desc="본문 텍스트 다운로드 중"):
            if save_report_text_selenium(url, main_driver):
                success += 1

        main_driver.quit()

        print(f"\n작업 완료: {success}/{len(archive_urls)} 저장 성공.")
        print(f"파일 저장 경로: Colab 환경 내 '{SAVE_DIR}' 폴더")