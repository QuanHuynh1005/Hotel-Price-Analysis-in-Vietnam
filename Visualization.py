"""
Phân tích mô tả và trực quan hóa dữ liệu khách sạn
Bao gồm: Thống kê mô tả, biểu đồ phân phối, và phân tích tương quan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import sys
import io
import warnings

# Thiết lập encoding UTF-8 cho output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# Thiết lập matplotlib để hiển thị tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def norm(s):
    """Chuẩn hóa tên cột: lowercase, strip, thay thế ký tự đặc biệt"""
    return str(s).strip().lower().replace("đ", "d").replace(" ", "_")


def to_numeric_strip(series):
    """Chuyển đổi chuỗi sang số, loại bỏ ký tự không phải số"""
    s = series.astype(str).str.replace(r"[^\d\.]", "", regex=True)
    
    def fix(x):
        if x.count(".") <= 1:
            return x
        last = x.rfind(".")
        return x[:last].replace(".", "") + x[last:]
    
    s = s.apply(fix)
    return pd.to_numeric(s, errors="coerce")


def describe_series(s):
    """Tính thống kê mô tả chi tiết cho một Series"""
    s = s.dropna()
    out = {
        "count": int(s.shape[0]),
        "mean": s.mean(),
        "std": s.std(ddof=1),
        "min": s.min(),
        "q25": s.quantile(0.25),
        "median": s.median(),
        "q75": s.quantile(0.75),
        "max": s.max(),
        "mad": (s - s.median()).abs().median(),
        "coef_var": (s.std(ddof=1) / s.mean()) if s.mean() != 0 else np.nan
    }
    return pd.DataFrame(out, index=[0]).T.rename(columns={0: "Gia_Phong"})


def load_and_clean_data(file_path):
    """Đọc và làm sạch dữ liệu từ file Excel"""
    print("=" * 70)
    print("ĐỌC VÀ LÀM SẠCH DỮ LIỆU")
    print("=" * 70)
    
    # Đọc 2 sheet
    df_price = pd.read_excel(file_path, sheet_name="giá_khách_sạn_theo_ngày")
    df_info = pd.read_excel(file_path, sheet_name="Thông tin khách sạn")
    
    # Chuẩn hóa tên cột
    df_price.columns = [norm(c) for c in df_price.columns]
    df_info.columns = [norm(c) for c in df_info.columns]
    
    print(f"✅ Đã đọc dữ liệu:")
    print(f"   - Sheet giá: {df_price.shape[0]} dòng, {df_price.shape[1]} cột")
    print(f"   - Sheet thông tin: {df_info.shape[0]} dòng, {df_info.shape[1]} cột")
    
    # Định nghĩa tên cột
    PRICE = "price_on_list"
    STAR = "star"
    RATING = "avg_rating"
    DIST = "distance_to_center"
    CITY = "city"
    KEY = "hotel_id"
    
    # Làm sạch giá
    df_price[PRICE] = to_numeric_strip(df_price[PRICE])
    
    # Làm sạch star/rating/distance
    for c in [STAR, RATING, DIST]:
        if c in df_info.columns and not pd.api.types.is_numeric_dtype(df_info[c]):
            df_info[c] = to_numeric_strip(df_info[c])
    
    # Chuyển đổi datetime (nếu có)
    for dcol in ["scraped_at", "ngay_dat", "ngày_đặt", "ngày_dặt", "ngay_tra", "ngày_trả"]:
        if dcol in df_price.columns:
            df_price[dcol] = pd.to_datetime(df_price[dcol], errors="coerce")
    
    # Merge 2 sheet
    cols_price = [c for c in [KEY, CITY, PRICE, "scraped_at"] if c in df_price.columns]
    cols_info = [c for c in [KEY, CITY, STAR, RATING, DIST] if c in df_info.columns]
    
    df = pd.merge(
        df_price[cols_price],
        df_info[cols_info],
        on=KEY, how="left", suffixes=("", "_info")
    )
    
    # Xử lý city_info
    if "city_info" in df.columns:
        df[CITY] = df["city_info"].fillna(df[CITY])
        df.drop(columns=["city_info"], inplace=True, errors="ignore")
    
    # Lọc giá hợp lệ & bỏ trùng
    df = df[df[PRICE].notna() & (df[PRICE] > 0)].drop_duplicates()
    
    print(f"✅ Sau khi làm sạch: {df.shape[0]} dòng, {df.shape[1]} cột")
    
    return df, PRICE, STAR, RATING, DIST, CITY


def save_descriptive_stats(df, PRICE, STAR, RATING, DIST, CITY, OUT):
    """Tính và hiển thị các thống kê mô tả"""
    print("\n" + "=" * 70)
    print("THỐNG KÊ MÔ TÃ")
    print("=" * 70)
    
    # 1. Thống kê tổng quan
    summary_overall = describe_series(df[PRICE])
    print("\n[1] Thống kê tổng quan giá phòng:")
    print(summary_overall)
    
    # 2. Thống kê theo hạng sao
    if STAR in df.columns:
        by_star = df.groupby(STAR, dropna=False)[PRICE].agg(
            count="count", mean="mean", std="std", min="min",
            q25=lambda x: x.quantile(0.25),
            median="median",
            q75=lambda x: x.quantile(0.75),
            max="max"
        ).reset_index().sort_values(by=STAR)
        
        by_star = by_star[by_star[STAR].notna()].reset_index(drop=True)
        print("\n[2] Thống kê giá theo hạng sao:")
        print(by_star.to_string(index=False))
    
    # 3. Thống kê theo thành phố
    if CITY in df.columns:
        by_city = df.groupby(CITY, dropna=False)[PRICE].agg(
            count="count", mean="mean", std="std", min="min",
            q25=lambda x: x.quantile(0.25),
            median="median",
            q75=lambda x: x.quantile(0.75),
            max="max"
        ).reset_index().sort_values(by="mean", ascending=False)
        print("\n[3] Thống kê giá theo thành phố (Top 10):")
        print(by_city.head(10).to_string(index=False))
    
    # 4. Thống kê theo rating
    if RATING in df.columns:
        bins = [-np.inf, 7, 8, 9, np.inf]
        labels = ["<=7", "7-8", "8-9", ">9"]
        df["rating_bin"] = pd.cut(df[RATING], bins=bins, labels=labels)
        by_rating = df.groupby("rating_bin", observed=False)[PRICE].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).reset_index()
        print("\n[4] Thống kê giá theo mức rating:")
        print(by_rating.to_string(index=False))
    
    # 5. Thống kê theo khoảng cách
    if DIST in df.columns:
        bins = [-np.inf, 1, 3, 5, np.inf]
        labels = ["<=1km", "1-3km", "3-5km", ">5km"]
        df["distance_bin"] = pd.cut(df[DIST], bins=bins, labels=labels)
        by_dist = df.groupby("distance_bin", observed=False)[PRICE].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).reset_index()
        print("\n[5] Thống kê giá theo khoảng cách:")
        print(by_dist.to_string(index=False))
    
    return by_star if STAR in df.columns else None, \
           by_rating if RATING in df.columns else None, \
           by_dist if DIST in df.columns else None


def create_visualizations(df, PRICE, STAR, RATING, DIST, CITY, OUT, by_star, by_rating, by_dist):
    """Tạo các biểu đồ trực quan"""
    print("\n" + "=" * 70)
    print("TẠO BIỂU ĐỒ TRỰC QUAN")
    print("=" * 70)
    
    # 1. Histogram - Phân phối giá (chia theo 100 nghìn)
    plt.figure(figsize=(12, 6))
    
    # Tạo bins cụ thể: mỗi bin = 100 nghìn VND
    bins = range(0, 6100, 100)
    plt.hist(df[PRICE], bins=bins, edgecolor='black', color="#1f77b4")
    
    plt.axvline(df[PRICE].mean(), color='red', linestyle='dashed', linewidth=1.5, label='Gia trung binh')
    plt.axvline(df[PRICE].median(), color='green', linestyle='dashed', linewidth=1.5, label='Trung vi')
    
    # Zoom vào vùng 0-6000 và hiển thị nhãn theo 100 nghìn
    plt.xlim(0, 6000)
    tick_positions = range(0, 6100, 100)
    plt.xticks(tick_positions, [f"{x}" for x in tick_positions], rotation=45, ha='right')
    
    plt.title("Phan phoi gia phong khach san - Chia theo 100 nghin VND")
    plt.xlabel("Gia phong (nghin VND/dem)")
    plt.ylabel("Tan suat")
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "chart_hist_price_full.png", dpi=200)
    plt.close()
    print("✅ [1] Đã tạo: chart_hist_price_full.png")
    
    # 2. Histogram - Phân phối giá (zoom 4000 nghìn - để thấy rõ phân hóa)
    plt.figure(figsize=(12, 6))
    
    # Tạo bins cụ thể: mỗi bin = 100 nghìn VND
    bins = range(0, 4100, 100)
    plt.hist(df[PRICE], bins=bins, edgecolor='black', color="#1f77b4")
    
    plt.xlim(0, 4000)
    
    # Chia trục X theo mỗi 100 nghìn
    tick_positions = range(0, 4100, 100)
    plt.xticks(tick_positions, [f"{x}" for x in tick_positions], rotation=45, ha='right')
    
    plt.axvline(df[PRICE].mean(), color='red', linestyle='dashed', linewidth=1.5, label='Gia trung binh')
    plt.axvline(df[PRICE].median(), color='green', linestyle='dashed', linewidth=1.5, label='Trung vi')
    
    plt.title("Phan phoi gia phong khach san (Zoom chi tiet 4000 nghin)")
    plt.xlabel("Gia phong (nghin VND/dem)")
    plt.ylabel("Tan suat")
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "chart_hist_price_zoom.png", dpi=200)
    plt.close()
    print("✅ [2] Đã tạo: chart_hist_price_zoom.png")
    
    # 3. Boxplot theo thành phố (zoom)
    if CITY in df.columns:
        top_cities = df[CITY].value_counts().head(10).index
        dft = df[df[CITY].isin(top_cities)].copy()
        order = dft.groupby(CITY)[PRICE].mean().sort_values().index
        
        plt.figure(figsize=(9, 5))
        sns.boxplot(
            data=dft, x=CITY, y=PRICE, order=order,
            color="#1f77b4", showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 5},
            medianprops={"color": "black", "linewidth": 1.2}
        )
        plt.ylim(0, 4000)
        plt.title("Phan bo gia phong theo thanh pho (Top 10) - Zoom 4 trieu")
        plt.xlabel("Thanh pho")
        plt.ylabel("Gia phong (VND/dem)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(OUT / "chart_box_city_zoom.png", dpi=200)
        plt.close()
        print("✅ [3] Đã tạo: chart_box_city_zoom.png")
        
        # 4. Boxplot theo thành phố (toàn bộ)
        plt.figure(figsize=(9, 5))
        sns.boxplot(
            data=dft, x=CITY, y=PRICE, order=order,
            color="#1f77b4", showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 5},
            medianprops={"color": "black", "linewidth": 1.2}
        )
        plt.title("Phan bo gia phong theo thanh pho (Top 10) - Toan bo du lieu")
        plt.xlabel("Thanh pho")
        plt.ylabel("Gia phong (VND/dem)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(OUT / "chart_box_city_full.png", dpi=200)
        plt.close()
        print("✅ [4] Đã tạo: chart_box_city_full.png")
    
    # 5. Bar chart - Giá trung bình theo hạng sao
    if STAR in df.columns and by_star is not None:
        plt.figure(figsize=(8, 5))
        plt.bar(by_star[STAR].astype(str), by_star["mean"], color="#1f77b4")
        plt.title("Gia trung binh theo hang sao")
        plt.xlabel("Hang sao")
        plt.ylabel("Gia trung binh (VND)")
        plt.tight_layout()
        plt.savefig(OUT / "chart_bar_mean_by_star.png", dpi=200)
        plt.close()
        print("✅ [5] Đã tạo: chart_bar_mean_by_star.png")
        
        # 6. Bar chart - Giá trung vị theo hạng sao
        plt.figure(figsize=(8, 5))
        plt.bar(by_star[STAR].astype(str), by_star["median"], color="#2ca02c")
        plt.title("Gia trung vi theo hang sao")
        plt.xlabel("Hang sao")
        plt.ylabel("Gia trung vi (VND)")
        plt.tight_layout()
        plt.savefig(OUT / "chart_bar_median_by_star.png", dpi=200)
        plt.close()
        print("✅ [6] Đã tạo: chart_bar_median_by_star.png")
    
    # 7. Bar chart - Giá trung bình theo rating
    if RATING in df.columns and by_rating is not None:
        plt.figure(figsize=(8, 6))
        plt.bar(by_rating["rating_bin"].astype(str), by_rating["mean"], color="#ff7f0e")
        plt.title("Gia trung binh theo muc rating")
        plt.xlabel("Nhom rating")
        plt.ylabel("Gia trung binh (VND)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        for i, v in enumerate(by_rating["mean"]):
            plt.text(i, v + 50, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT / "chart_bar_mean_by_rating.png", dpi=200)
        plt.close()
        print("✅ [7] Đã tạo: chart_bar_mean_by_rating.png")
    
    # 8. Bar chart - Giá trung bình theo khoảng cách
    if DIST in df.columns and by_dist is not None:
        plt.figure(figsize=(8, 6))
        plt.bar(by_dist["distance_bin"].astype(str), by_dist["mean"], color="#d62728")
        plt.title("Gia trung binh theo khoang cach den trung tam")
        plt.xlabel("Khoang cach (km)")
        plt.ylabel("Gia trung binh (VND)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        for i, v in enumerate(by_dist["mean"]):
            plt.text(i, v + 50, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT / "chart_bar_mean_by_dist.png", dpi=200)
        plt.close()
        print("✅ [8] Đã tạo: chart_bar_mean_by_dist.png")


def correlation_analysis(df, PRICE, STAR, RATING, DIST, OUT):
    """Phân tích tương quan Pearson"""
    print("\n" + "=" * 70)
    print("PHÂN TÍCH TƯƠNG QUAN")
    print("=" * 70)
    
    # Lọc dữ liệu hợp lệ
    cols = [PRICE, STAR, RATING, DIST]
    df_corr = df[cols].dropna()
    
    print(f"\n✅ Số lượng mẫu hợp lệ: {len(df_corr)}")
    
    # Tính hệ số tương quan Pearson
    corr_results = []
    for var in [STAR, RATING, DIST]:
        r, p = pearsonr(df_corr[PRICE], df_corr[var])
        corr_results.append({
            "Bien so sanh": f"{PRICE} - {var}",
            "He so tuong quan (r)": round(r, 3),
            "Gia tri p": round(p, 4),
            "Y nghia thong ke": "Co y nghia (p<0.05)" if p < 0.05 else "Khong y nghia"
        })
    
    corr_table = pd.DataFrame(corr_results)
    print("\nKết quả tương quan Pearson:")
    print(corr_table.to_string(index=False))
    
    # Ma trận tương quan
    corr_matrix = df_corr.corr(method='pearson')
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Ma tran tuong quan Pearson giua cac bien dinh luong")
    plt.tight_layout()
    plt.savefig(OUT / "chart_heatmap_correlation.png", dpi=200)
    plt.close()
    print("\n✅ [9] Đã tạo: chart_heatmap_correlation.png")


def main():
    """Hàm chính"""
    print("=" * 70)
    print("PHÂN TÍCH MÔ TẢ VÀ TRỰC QUAN HÓA DỮ LIỆU KHÁCH SẠN")
    print("=" * 70)
    
    # Tạo thư mục output
    OUT = Path("outputs_descriptive")
    OUT.mkdir(exist_ok=True)
    print(f"\n✅ Thư mục output: {OUT.absolute()}")
    
    # Đọc và làm sạch dữ liệu
    file_path = "khach_san.xlsx"
    try:
        df, PRICE, STAR, RATING, DIST, CITY = load_and_clean_data(file_path)
    except FileNotFoundError:
        print(f"\n❌ Không tìm thấy file: {file_path}")
        print("Vui lòng đặt file 'khach_san.xlsx' trong cùng thư mục.")
        return
    
    # Thống kê mô tả
    by_star, by_rating, by_dist = save_descriptive_stats(df, PRICE, STAR, RATING, DIST, CITY, OUT)
    
    # Tạo biểu đồ
    create_visualizations(df, PRICE, STAR, RATING, DIST, CITY, OUT, by_star, by_rating, by_dist)
    
    # Phân tích tương quan
    correlation_analysis(df, PRICE, STAR, RATING, DIST, OUT)
    
    print("\n" + "=" * 70)
    print("HOÀN THÀNH!")
    print("=" * 70)
    print(f"✅ Tất cả biểu đồ đã được lưu trong thư mục: {OUT.absolute()}")
    print(f"   - 9 file PNG biểu đồ trực quan")


if __name__ == "__main__":
    main()

