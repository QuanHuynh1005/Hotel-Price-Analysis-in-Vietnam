"""
Phân tích ANOVA - So sánh giá khách sạn theo hạng sao
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from scipy.stats import studentized_range
import sys
import io

# Thiết lập encoding UTF-8 cho output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    # --- Đọc file ---
    file_path = "khach_san.xlsx"  # Thay đổi đường dẫn nếu cần
    
    try:
        sheet1 = pd.read_excel(file_path, sheet_name="giá_khách_sạn_theo_ngày")
        sheet2 = pd.read_excel(file_path, sheet_name="Thông tin khách sạn")
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {file_path}")
        print("Vui lòng đặt file 'khach_san.xlsx' trong cùng thư mục hoặc cập nhật đường dẫn.")
        return
    
    # --- Làm sạch tên cột ---
    sheet1.columns = sheet1.columns.str.strip()
    sheet2.columns = sheet2.columns.str.strip()
    
    # --- Làm sạch cột giá ---
    sheet1["price_on_list"] = (
        sheet1["price_on_list"]
        .astype(str)
        .str.replace("VND", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.strip()
    )
    
    sheet1["price_on_list"] = pd.to_numeric(sheet1["price_on_list"], errors="coerce")
    
    # --- Gộp dữ liệu ---
    data = pd.merge(sheet1, sheet2, on="hotel_id", how="inner")
    
    # --- Lọc dữ liệu hợp lệ ---
    data = data[["hotel_id", "price_on_list", "star"]].dropna()
    data = data[data["price_on_list"] > 0]
    data["star"] = data["star"].astype(int)
    
    # --- Kiểm tra nhóm sao ---
    print("Nhóm sao:", data["star"].unique())
    
    # ===== BẢNG SỐ LƯỢNG MỖI NHÓM =====
    group_count = data.groupby("star")["price_on_list"].count().reset_index()
    group_count.columns = ["Số sao", "Số lượng khách sạn"]
    
    print("\n===== BẢNG SỐ LƯỢNG MỖI NHÓM =====")
    print(group_count.to_markdown(index=False, tablefmt="grid", numalign="left", stralign="left"))
    
    # --- ANOVA ---
    model = ols("price_on_list ~ C(star)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # --- Thêm cột Mean Square (MS) ---
    anova_table["MS"] = anova_table["sum_sq"] / anova_table["df"]
    
    # --- F critical ---
    df_between = anova_table["df"].iloc[0]
    df_within = anova_table["df"].iloc[1]
    f_critical = stats.f.ppf(0.95, df_between, df_within)
    
    anova_display = anova_table.rename(columns={
        "sum_sq": "SS",
        "df": "df",
        "F": "F",
        "PR(>F)": "P-value"
    })[["SS", "df", "MS", "F", "P-value"]]
    
    print("\n===== BẢNG KẾT QUẢ ANOVA =====")
    print(anova_display.to_markdown(tablefmt="grid", numalign="left", stralign="left"))
    print(f"\nF critical (α = 0.05): {f_critical:.4f}")
    
    p_value = anova_table["PR(>F)"].iloc[0]
    
    # --- Kết luận ANOVA ---
    if p_value < 0.05:
        print("\n✅ Có sự khác biệt có ý nghĩa thống kê giữa các nhóm sao (p < 0.05)")
        
        # --- KIỂM ĐỊNH HSD THỦ CÔNG ---
        print("\n===== KIỂM ĐỊNH HSD THỦ CÔNG =====")
        
        MSE = anova_table["MS"].iloc[1]  # Mean Square Error
        df_within = int(anova_table["df"].iloc[1])
        k = data["star"].nunique()
        n_per_group = data.groupby("star")["price_on_list"].count().mean()
        q_critical = studentized_range.ppf(0.95, k, df_within)
        
        HSD = q_critical * np.sqrt(MSE / n_per_group)
        print(f"HSD (ngưỡng chênh lệch cần đạt) = {HSD:,.2f}")
        
        # --- Trung bình từng nhóm ---
        means = data.groupby("star")["price_on_list"].mean().sort_index()
        means_df = means.reset_index()
        means_df.columns = ["star", "price_on_list"]
        means_df["price_on_list"] = means_df["price_on_list"].apply(lambda x: f"{x:,.0f}")
        
        print("\nTrung bình giá theo sao:")
        print(means_df.to_markdown(index=False, tablefmt="grid", numalign="left", stralign="left"))
        
        # --- So sánh từng cặp ---
        pairs = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                diff = abs(means.iloc[i] - means.iloc[j])
                significant = diff > HSD
                pairs.append({
                    "group1": means.index[i],
                    "group2": means.index[j],
                    "mean_diff": diff,
                    "Significant": "✅ Có khác biệt" if significant else "❌ Không khác biệt"
                })
        
        pairs_df = pd.DataFrame(pairs)
        
        # --- In bảng đẹp (căn trái) ---
        pairs_df_display = pairs_df.copy()
        pairs_df_display["mean_diff"] = pairs_df_display["mean_diff"].apply(lambda x: f"{x:,.0f}")
        print("\n===== SO SÁNH THỦ CÔNG THEO HSD =====")
        print(pairs_df_display.to_markdown(index=False, tablefmt="grid", numalign="left", stralign="left"))
        
        # --- Kết luận cuối ---
        sig = pairs_df[pairs_df["Significant"] == "✅ Có khác biệt"]
        if not sig.empty:
            print("\n✅ Các cặp nhóm có khác biệt theo HSD:")
            for _, row in sig.iterrows():
                print(f"   - Nhóm {row['group1']} và nhóm {row['group2']} khác biệt (Δ = {row['mean_diff']:.2f})")
        else:
            print("\n❌ Không có cặp nhóm nào khác biệt theo HSD.")
    
    else:
        print("\n❌ Không có sự khác biệt đáng kể giữa các nhóm sao (p ≥ 0.05)")


if __name__ == "__main__":
    main()

