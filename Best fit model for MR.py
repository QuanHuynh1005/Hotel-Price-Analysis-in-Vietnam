"""
Mô hình hồi quy tuyến tính đa biến - Dự đoán giá khách sạn
Sử dụng Backward Elimination với kiểm tra Adjusted R²

LƯU Ý: Không sử dụng 6 biến rating riêng lẻ (service, facility, cleanliness, 
comfort, value, location) vì các biến này có correlation > 0.9 với nhau,
gây ra MULTICOLLINEARITY nghiêm trọng và làm hệ số bị nghịch lý.
CHI SỬ DỤNG: avg_rating (trung bình của 6 biến trên)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import io

# Thiết lập encoding UTF-8 cho output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def clean_distance(x):
    """Chuyển đổi khoảng cách từ m/km sang km"""
    if pd.isna(x):
        return np.nan
    x = str(x).lower().replace(",", ".")
    if "m" in x:
        num = ''.join(ch for ch in x if ch.isdigit() or ch == ".")
        return float(num) / 1000 if num else np.nan
    elif "km" in x:
        num = ''.join(ch for ch in x if ch.isdigit() or ch == ".")
        return float(num) if num else np.nan
    return np.nan


def backward_elimination_r2(X, y, significance_level=0.05):
    """
    Backward Elimination với kiểm tra Adjusted R²
    Chỉ loại bỏ biến nếu p-value > 0.05 VÀ Adjusted R² không giảm
    """
    variables = X.columns.tolist()
    best_adj_r2 = -np.inf
    best_model = None
    improved = True

    print("\n===== QUÁ TRÌNH LOẠI BỎ BIẾN =====")
    
    while improved and len(variables) > 1:
        improved = False
        model = sm.OLS(y.astype(float), X[variables].astype(float)).fit()
        best_adj_r2 = model.rsquared_adj
        p_values = model.pvalues

        max_p_value = p_values.max()
        excluded_var = p_values.idxmax()

        if max_p_value > significance_level:
            temp_vars = variables.copy()
            temp_vars.remove(excluded_var)
            temp_model = sm.OLS(y.astype(float), X[temp_vars].astype(float)).fit()

            if temp_model.rsquared_adj >= best_adj_r2:
                print(f"✓ Loại '{excluded_var}' (p={max_p_value:.4f}), Adjusted R² tăng lên {temp_model.rsquared_adj:.4f}")
                variables.remove(excluded_var)
                best_model = temp_model
                improved = True
            else:
                print(f"✗ Giữ '{excluded_var}' vì loại bỏ làm giảm Adjusted R² ({temp_model.rsquared_adj:.4f} < {best_adj_r2:.4f})")
        else:
            break

    if best_model is None:
        best_model = sm.OLS(y.astype(float), X[variables].astype(float)).fit()

    return best_model


def main():
    print("=" * 70)
    print("MÔ HÌNH HỒI QUY TUYẾN TÍNH ĐA BIẾN - DỰ ĐOÁN GIÁ KHÁCH SẠN")
    print("=" * 70)
    
    # === 1. ĐỌC & LÀM SẠCH DỮ LIỆU ===
    file_path = "khach_san.xlsx"
    
    try:
        sheet1 = pd.read_excel(file_path, sheet_name="giá_khách_sạn_theo_ngày")
        sheet2 = pd.read_excel(file_path, sheet_name="Thông tin khách sạn")
    except FileNotFoundError:
        print(f"\n❌ Không tìm thấy file: {file_path}")
        print("Vui lòng đặt file 'khach_san.xlsx' trong cùng thư mục.")
        return
    
    # Làm sạch tên cột
    sheet1.columns = sheet1.columns.str.strip()
    sheet2.columns = sheet2.columns.str.strip()
    
    # --- Làm sạch giá ---
    sheet1["price_on_list"] = (
        sheet1["price_on_list"]
        .astype(str)
        .str.replace("VND", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace("\u00A0", "", regex=False)  # Non-breaking space
        .str.strip()
    )
    sheet1["price_on_list"] = pd.to_numeric(sheet1["price_on_list"], errors="coerce")
    
    # --- Chuẩn hóa khoảng cách ---
    sheet2["distance_to_center"] = sheet2["distance_to_center"].apply(clean_distance)
    
    # --- Gộp dữ liệu ---
    data = pd.merge(sheet1, sheet2, on="hotel_id", how="inner")
    
    # --- Xử lý city_x / city_y ---
    if "city_x" in data.columns and "city_y" in data.columns:
        data["city"] = data["city_x"].combine_first(data["city_y"])
    elif "city_x" in data.columns:
        data["city"] = data["city_x"]
    elif "city_y" in data.columns:
        data["city"] = data["city_y"]
    
    # --- Lọc cột cần thiết - CHỈ DÙNG avg_rating ---
    # KHÔNG dùng 6 biến rating riêng lẻ để tránh multicollinearity
    cols = [
        "price_on_list", "city", "star", "avg_rating", "distance_to_center"
    ]
    data = data[cols].dropna(how="any")
    
    # --- Ép kiểu ---
    num_cols = [
        "price_on_list", "star", "avg_rating", "distance_to_center"
    ]
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna()
    data = data[data["price_on_list"] > 0]
    
    if data.empty:
        raise ValueError("❌ Dữ liệu sau khi làm sạch bị rỗng.")
    
    print(f"\n✅ Số lượng mẫu hợp lệ: {len(data)}")
    print(f"📊 Các cột: {list(data.columns)}")
    
    # === 2. CHUẨN BỊ DỮ LIỆU HỒI QUY ===
    # Tạo biến giả cho city (one-hot encoding)
    data = pd.get_dummies(data, columns=["city"], drop_first=True)
    
    # Ép kiểu toàn bộ cột về float
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except (ValueError, TypeError):
            pass
    
    # Biến phụ thuộc và độc lập
    y = data["price_on_list"]
    X = data.drop(columns=["price_on_list"])
    X = sm.add_constant(X)
    
    # === 3. CHẠY BACKWARD ELIMINATION ===
    best_model = backward_elimination_r2(X, y)
    
    # === 4. TỔNG HỢP MÔ HÌNH ===
    print("\n" + "=" * 70)
    print("MÔ HÌNH TỐT NHẤT")
    print("=" * 70)
    print(best_model.summary())
    
    # === 5. ĐÁNH GIÁ MÔ HÌNH ===
    y_pred = best_model.predict(X[best_model.params.index])
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    print("\n" + "=" * 70)
    print("ĐỘ CHÍNH XÁC MÔ HÌNH")
    print("=" * 70)
    print(f"MAE  (Mean Absolute Error)       = {mae:,.2f} VND")
    print(f"MSE  (Mean Squared Error)        = {mse:,.2f}")
    print(f"RMSE (Root Mean Squared Error)   = {rmse:,.2f} VND")
    print(f"MAPE (Mean Absolute % Error)     = {mape:.2f}%")
    
    # === 6. PHƯƠNG TRÌNH HỒI QUY ===
    print("\n" + "=" * 70)
    print("PHƯƠNG TRÌNH HỒI QUY")
    print("=" * 70)
    
    formula_parts = []
    for k, v in best_model.params.items():
        if k == "const":
            formula_parts.append(f"{v:,.2f}")
        else:
            sign = "+" if v >= 0 else ""
            formula_parts.append(f"{sign} {v:,.2f} × {k}")
    
    print("price_on_list = " + " ".join(formula_parts))
    
    # === 7. HỆ SỐ HỒI QUY ===
    coeff_df = pd.DataFrame({
        "Variable": best_model.params.index,
        "Coefficient": best_model.params.values,
        "Std Error": best_model.bse.values,
        "t-Stat": best_model.tvalues.values,
        "P-value": best_model.pvalues.values,
        "Lower 95%": best_model.conf_int()[0].values,
        "Upper 95%": best_model.conf_int()[1].values,
    })
    
    print("\n" + "=" * 70)
    print("HỆ SỐ HỒI QUY CHI TIẾT")
    print("=" * 70)
    
    # Format số đẹp hơn
    coeff_df_display = coeff_df.copy()
    for col in ["Coefficient", "Std Error", "Lower 95%", "Upper 95%"]:
        coeff_df_display[col] = coeff_df_display[col].apply(lambda x: f"{x:,.2f}")
    coeff_df_display["t-Stat"] = coeff_df_display["t-Stat"].apply(lambda x: f"{x:.4f}")
    coeff_df_display["P-value"] = coeff_df_display["P-value"].apply(lambda x: f"{x:.6f}" if x >= 0.0001 else f"{x:.2e}")
    
    print(coeff_df_display.to_string(index=False))
    
    # === 8. PHÂN TÍCH Ý NGHĨA ===
    print("\n" + "=" * 70)
    print("PHÂN TÍCH CÁC BIẾN QUAN TRỌNG (p < 0.05)")
    print("=" * 70)
    
    significant_vars = coeff_df[coeff_df["P-value"] < 0.05].sort_values("P-value")
    
    for _, row in significant_vars.iterrows():
        var = row["Variable"]
        coef = row["Coefficient"]
        p_val = row["P-value"]
        
        if var == "const":
            continue
        
        direction = "TĂNG" if coef > 0 else "GIẢM"
        print(f"• {var:20s}: {direction:5s} {abs(coef):>12,.0f} VND (p={p_val:.6f})")
    
    print("\n" + "=" * 70)
    print("HOÀN THÀNH PHÂN TÍCH")
    print("=" * 70)


if __name__ == "__main__":
    main()

