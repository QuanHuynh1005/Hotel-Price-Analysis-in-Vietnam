# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === 1. ĐỌC & LÀM SẠCH DỮ LIỆU ===
file_path = "/kaggle/input/khach-san/khach_san.xlsx"

sheet1 = pd.read_excel(file_path, sheet_name="giá_khách_sạn_theo_ngày")
sheet2 = pd.read_excel(file_path, sheet_name="Thông tin khách sạn")

sheet1.columns = sheet1.columns.str.strip()
sheet2.columns = sheet2.columns.str.strip()

# --- Làm sạch giá ---
sheet1["price_on_list"] = (
    sheet1["price_on_list"]
    .astype(str)
    .str.replace("VND", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.strip()
)
sheet1["price_on_list"] = pd.to_numeric(sheet1["price_on_list"], errors="coerce")

# --- Chuẩn hóa khoảng cách ---
def clean_distance(x):
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

# --- Lọc cột cần thiết ---
cols = [
    "price_on_list", "city", "star", "avg_rating", "distance_to_center",
    "Nhân viên phục vụ", "Tiện Nghi", "Sạch sẽ", "Thoải Mái",
    "Đáng giá tiền", "Địa điểm"
]
data = data[cols].dropna(how="any")

# --- Ép kiểu ---
num_cols = [
    "price_on_list", "star", "avg_rating", "distance_to_center",
    "Nhân viên phục vụ", "Tiện Nghi", "Sạch sẽ", "Thoải Mái",
    "Đáng giá tiền", "Địa điểm"
]
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")
data = data.dropna()
data = data[data["price_on_list"] > 0]

# --- Đổi tên cột tiếng Anh ---
rename_cols = {
    "Nhân viên phục vụ": "service",
    "Tiện Nghi": "facility",
    "Sạch sẽ": "cleanliness",
    "Thoải Mái": "comfort",
    "Đáng giá tiền": "value",
    "Địa điểm": "location"
}
data = data.rename(columns=rename_cols)

if data.empty:
    raise ValueError("❌ Dữ liệu sau khi làm sạch bị rỗng. Kiểm tra lại file Excel hoặc tên cột.")

print("✅ Số lượng mẫu hợp lệ:", len(data))
print("Các cột mô hình:", list(data.columns))

# === 2. HỒI QUY ĐA BIẾN ===
formula = """
price_on_list ~ star + avg_rating + distance_to_center +
                service + facility + cleanliness + comfort + 
                value + location + C(city)
"""
model = smf.ols(formula=formula, data=data).fit()
y = data["price_on_list"]
y_pred = model.fittedvalues

# === 3. CHỈ SỐ ĐÁNH GIÁ ===
MAE = mean_absolute_error(y, y_pred)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
MAPE = np.mean(np.abs((y - y_pred) / y)) * 100

# === 4. THÔNG TIN MÔ HÌNH ===
print("\n===== MÔ HÌNH HỒI QUY ĐA BIẾN =====")
print(f"Multiple R       : {np.sqrt(model.rsquared):.4f}")
print(f"R Square         : {model.rsquared:.4f}")
print(f"Adjusted R Square: {model.rsquared_adj:.4f}")
print(f"Standard Error   : {np.sqrt(MSE):,.4f}")
print(f"Observations     : {len(data)}")

# === 5. BẢNG ANOVA ===
print("\n===== BẢNG ANOVA =====")
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table.to_markdown(tablefmt='grid', numalign='left', stralign='left'))

# === 6. HỆ SỐ HỒI QUY ===
coeff_df = pd.DataFrame({
    "Coefficients": model.params,
    "Std Error": model.bse,
    "t Stat": model.tvalues,
    "P-value": model.pvalues,
    "Lower 95%": model.conf_int()[0],
    "Upper 95%": model.conf_int()[1],
})
print("\n===== HỆ SỐ HỒI QUY =====")
print(coeff_df.to_markdown(tablefmt='grid', numalign='left', stralign='left'))

# === 7. ĐỘ CHÍNH XÁC MÔ HÌNH ===
print("\n===== ĐỘ CHÍNH XÁC MÔ HÌNH =====")
print(f"MAE  = {MAE:,.2f}")
print(f"MSE  = {MSE:,.2f}")
print(f"RMSE = {RMSE:,.2f}")
print(f"MAPE = {MAPE:.2f}%")

# === 8. PHƯƠNG TRÌNH HỒI QUY ===
formula_str = "price_on_list = "
terms = []
for k, v in model.params.items():
    if k == "Intercept":
        terms.append(f"{v:.2f}")
    elif "C(city)" in k:
        # Lấy đúng tên thành phố từ dạng 'C(city)[T.TP. Hồ Chí Minh]'
        city_name = k.split("[T.")[-1].rstrip("]")
        terms.append(f"{v:.2f} * city({city_name})")
    else:
        terms.append(f"{v:.2f} * {k}")

formula_str += " + ".join(terms)

print("\n===== PHƯƠNG TRÌNH HỒI QUY =====")
print(formula_str)

