import time
import random
import re
import hashlib
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import pandas as pd
import sys

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

CITIES = [
    {"name": "TP. Hồ Chí Minh", "q": "Ho Chi Minh City"},
    {"name": "Hà Nội", "q": "Hanoi"},
    {"name": "Đà Nẵng", "q": "Da Nang"},
    {"name": "Đà Lạt", "q": "Da Lat"},
    {"name": "Nha Trang", "q": "Nha Trang"}
]

NUM_ADULTS = 2
TARGET_HOTELS_PER_DAY = 30

START_DATE_STR = "2025-10-17"
END_DATE_STR = "2026-03-31"

HEADLESS = False
OUTPUT_CSV = f"booking_multi_city_{START_DATE_STR}_to_{END_DATE_STR}_v16.csv"
BACKUP_CSV = f"backup_v16.csv"
WAIT_BASE = 1.0

def human_wait(a=1.0):
    time.sleep(WAIT_BASE * a + random.random() * 1.5)

def make_driver():
    opts = Options()
    if HEADLESS: opts.add_argument("--headless")
    opts.add_experimental_option('prefs', {'intl.accept_languages': 'vi,vi_VN'})
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(45)
    return driver

def make_id_from_url(url):
    if not url: return None
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def build_search_url(city_q, checkin_date, adults):
    checkout_date = (checkin_date + timedelta(days=1)).strftime('%Y-%m-%d')
    checkin_str = checkin_date.strftime('%Y-%m-%d')
    base = "https://www.booking.com/searchresults.html"
    params = { 
        "ss": city_q, 
        "checkin": checkin_str, 
        "checkout": checkout_date, 
        "group_adults": adults, 
        "no_rooms": 1,
        # "order": "price"  <-- Đã xóa dòng này
    }
    param_str = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{base}?{param_str}"

# ---------------- Parsing Function ----------------
def parse_hotel_card_fully(card, base_url):
    name_el = card.select_one('[data-testid="title"]')
    link_el = card.select_one('a[data-testid="title-link"]')
    distance_el = card.select_one('[data-testid="distance"]')
    price_el = card.select_one('[data-testid="price-and-discounted-price"]')
    star_container = card.select_one('[data-testid="rating-stars"]')
    amenities_list = [el.get_text(strip=True) for el in card.select('[data-testid="facility-group"] span')]
    
    review_container = card.select_one('[data-testid="review-score"]')
    avg_rating, num_reviews = "0", "0"
    if review_container:
        score_el = review_container.find('div', string=re.compile(r'^\d+([,.]\d+)?$'))
        if score_el: avg_rating = score_el.get_text(strip=True).replace(',', '.')
        review_count_text_node = review_container.find(string=re.compile(r'đánh giá'))
        if review_count_text_node:
            match = re.search(r'([\d.,]+)', review_count_text_node)
            if match: num_reviews = re.sub(r'[.,]', '', match.group(1))

    return {
        "listing_url": urljoin(base_url, link_el['href']) if link_el else None,
        "name": name_el.get_text(strip=True) if name_el else None,
        "avg_rating": avg_rating, "num_reviews": num_reviews,
        "distance_to_center": distance_el.get_text(strip=True) if distance_el else None,
        "price_on_list": price_el.get_text(strip=True) if price_el else None,
        "stars": len(star_container.find_all("span")) if star_container else None,
        "amenities": "; ".join(list(dict.fromkeys(amenities_list))) if amenities_list else None,
    }
    
# ---------------- Main scraping function ----------------
def scrape_booking_multi_city():
    all_results = []
    start_date = datetime.strptime(START_DATE_STR, '%Y-%m-%d')
    end_date = datetime.strptime(END_DATE_STR, '%Y-%m-%d')

    for city_info in CITIES:
        city_name = city_info["name"]
        city_query = city_info["q"]
        print(f"\n{'='*70}\n🏢 BẮT ĐẦU CÀO DỮ LIỆU CHO THÀNH PHỐ: {city_name.upper()}\n{'='*70}")
        
        current_date = start_date
        while current_date <= end_date:
            checkin_date_str = current_date.strftime('%Y-%m-%d')
            print(f"\n🗓️  Đang cào ngày: {checkin_date_str} cho {city_name}...")
            
            driver = make_driver()
            hotels_for_this_day = []
            processed_urls_this_day = set()
            search_url = build_search_url(city_query, current_date, NUM_ADULTS)

            try:
                driver.get(search_url)
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="property-card"]')))
            except TimeoutException:
                print(f"  └─ ❌ Lỗi timeout hoặc không tìm thấy khách sạn. Bỏ qua.")
                driver.quit()
                current_date += timedelta(days=1)
                continue
            
            page_num = 1
            while len(hotels_for_this_day) < TARGET_HOTELS_PER_DAY:
                print(f"  - Trang {page_num} | Đã tìm thấy: {len(hotels_for_this_day)}/{TARGET_HOTELS_PER_DAY} khách sạn...")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                human_wait(1)
                soup = BeautifulSoup(driver.page_source, 'lxml')
                cards = soup.select('[data-testid="property-card"]')
                
                if not cards and page_num == 1:
                    print("     └─ Không có khách sạn nào được tìm thấy trên trang.")
                    break

                new_hotels_on_page = 0
                for card in cards:
                    if len(hotels_for_this_day) >= TARGET_HOTELS_PER_DAY: break
                    
                    info = parse_hotel_card_fully(card, "https://www.booking.com")
                    if not info.get("listing_url") or info["listing_url"] in processed_urls_this_day or not info.get("price_on_list"):
                        continue

                    processed_urls_this_day.add(info["listing_url"])
                    
                    full_data = {
                        "city": city_name, "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "hotel_id": make_id_from_url(info["listing_url"]), **info,
                        "ngày đặt": checkin_date_str,
                        "ngày trả": (current_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    }
                    hotels_for_this_day.append(full_data)
                    new_hotels_on_page += 1

                    hotel_name_str = info['name'][:40].ljust(40)
                    found_count_str = f"({len(hotels_for_this_day)}/{TARGET_HOTELS_PER_DAY})".ljust(9)
                    output_line = f"  └─ (+) {found_count_str} {hotel_name_str} | Điểm: {info['avg_rating'].ljust(4)} | ĐG: {info['num_reviews'].ljust(5)} | Giá: {info['price_on_list']}"
                    print(output_line)

                if new_hotels_on_page == 0 and page_num > 1:
                    print("     └─ Không có khách sạn mới được tải thêm. Kết thúc ngày này.")
                    break

                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Trang tiếp theo"]')
                    if not next_button.is_enabled():
                        print("     └─ 🏁 Đã đến trang cuối cùng cho ngày này.")
                        break
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                    human_wait(0.5); next_button.click()
                    page_num += 1
                    human_wait(2.5)
                except NoSuchElementException:
                    print("     └─ 🏁 Không tìm thấy nút chuyển trang. Kết thúc ngày này.")
                    break
            
            driver.quit()
            all_results.extend(hotels_for_this_day)
            
            if hotels_for_this_day:
                pd.DataFrame(all_results).to_csv(BACKUP_CSV, index=False, encoding='utf-8-sig')
                print(f"  └─ 💾 Đã lưu backup. Tổng số dòng đã cào: {len(all_results)}")

            current_date += timedelta(days=1)

    if all_results:
        df = pd.DataFrame(all_results)
        cols_ordered = [
            "city", "scraped_at", "hotel_id", "name", "stars", "avg_rating", "num_reviews", 
            "price_on_list", "distance_to_center", "amenities", "listing_url", "ngày đặt", "ngày trả"
        ]
        df_final = df.reindex(columns=[col for col in cols_ordered if col in df.columns])
        df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n\n{'='*70}\n✅ HOÀN THÀNH TOÀN BỘ! Đã lưu {len(df_final)} dòng vào file: {OUTPUT_CSV}\n{'='*70}")
        
        print("\n📊 TỔNG KẾT SỐ LƯỢNG KHÁCH SẠN DUY NHẤT:\n")
        unique_counts = df.groupby('city')['hotel_id'].nunique()
        for city_name, count in unique_counts.items():
            print(f"- {city_name}: {count} khách sạn")

    else:
        print("\n❌ Không thu thập được dữ liệu nào.")

if __name__ == "__main__":
    scrape_booking_multi_city()
