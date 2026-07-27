# ============================================================================
# MULTIPLE REGRESSION ANALYSIS FOR HOTEL PRICE PREDICTION
# Equivalent to Multi Regression.py
# ============================================================================

# Load required libraries
library(readxl)      # For reading Excel files
library(dplyr)       # For data manipulation
library(stringr)     # For string operations
library(tidyr)       # For data tidying
library(car)         # For ANOVA Type II
library(knitr)       # For formatted output

# === 1. ĐỌC & LÀM SẠCH DỮ LIỆU ===
cat("=== BẮT ĐẦU ĐỌC VÀ LÀM SẠCH DỮ LIỆU ===\n")

# Đường dẫn file (thay đổi theo môi trường của bạn)
file_path <- "khach_san.xlsx"

# Đọc 2 sheet từ file Excel
sheet1 <- read_excel(file_path, sheet = "giá_khách_sạn_theo_ngày")
sheet2 <- read_excel(file_path, sheet = "Thông tin khách sạn")

# Loại bỏ khoảng trắng trong tên cột
names(sheet1) <- str_trim(names(sheet1))
names(sheet2) <- str_trim(names(sheet2))

# --- Làm sạch giá ---
cat("Đang làm sạch cột giá...\n")
sheet1$price_on_list <- sheet1$price_on_list %>%
  as.character() %>%
  str_replace_all("VND", "") %>%
  str_replace_all("\\.", "") %>%
  str_trim() %>%
  as.numeric()

# --- Chuẩn hóa khoảng cách ---
cat("Đang chuẩn hóa khoảng cách...\n")
clean_distance <- function(x) {
  if (is.na(x)) return(NA)
  
  x <- tolower(as.character(x))
  x <- str_replace_all(x, ",", ".")
  
  # Trích xuất số
  num_str <- str_extract(x, "[0-9.]+")
  
  if (is.na(num_str)) return(NA)
  
  num <- as.numeric(num_str)
  
  if (is.na(num)) return(NA)
  
  # Nếu đơn vị là mét (m), chuyển sang km
  if (str_detect(x, "\\bm\\b") && !str_detect(x, "km")) {
    return(num / 1000)
  } else if (str_detect(x, "km")) {
    return(num)
  }
  
  return(NA)
}

sheet2$distance_to_center <- sapply(sheet2$distance_to_center, clean_distance)

# --- Gộp dữ liệu ---
cat("Đang gộp dữ liệu...\n")
data <- inner_join(sheet1, sheet2, by = "hotel_id")

# --- Xử lý city_x / city_y ---
if ("city.x" %in% names(data) && "city.y" %in% names(data)) {
  data$city <- coalesce(data$city.x, data$city.y)
} else if ("city.x" %in% names(data)) {
  data$city <- data$city.x
} else if ("city.y" %in% names(data)) {
  data$city <- data$city.y
}

# --- Lọc cột cần thiết ---
cols <- c(
  "price_on_list", "city", "star", "avg_rating", "distance_to_center",
  "Nhân viên phục vụ", "Tiện Nghi", "Sạch sẽ", "Thoải Mái",
  "Đáng giá tiền", "Địa điểm"
)

data <- data %>% select(all_of(cols))

# --- Ép kiểu số ---
cat("Đang chuyển đổi kiểu dữ liệu...\n")
num_cols <- c(
  "price_on_list", "star", "avg_rating", "distance_to_center",
  "Nhân viên phục vụ", "Tiện Nghi", "Sạch sẽ", "Thoải Mái",
  "Đáng giá tiền", "Địa điểm"
)

for (col in num_cols) {
  data[[col]] <- as.numeric(data[[col]])
}

# Loại bỏ NA
data <- data %>% drop_na()

# Lọc giá > 0
data <- data %>% filter(price_on_list > 0)

# --- Đổi tên cột tiếng Anh ---
data <- data %>%
  rename(
    service = `Nhân viên phục vụ`,
    facility = `Tiện Nghi`,
    cleanliness = `Sạch sẽ`,
    comfort = `Thoải Mái`,
    value = `Đáng giá tiền`,
    location = `Địa điểm`
  )

# Kiểm tra dữ liệu rỗng
if (nrow(data) == 0) {
  stop("❌ Dữ liệu sau khi làm sạch bị rỗng. Kiểm tra lại file Excel hoặc tên cột.")
}

cat("✅ Số lượng mẫu hợp lệ:", nrow(data), "\n")
cat("Các cột mô hình:", paste(names(data), collapse = ", "), "\n\n")

# === 2. HỒI QUY ĐA BIẾN ===
cat("=== XÂY DỰNG MÔ HÌNH HỒI QUY ĐA BIẾN ===\n")

# Chuyển city thành factor
data$city <- as.factor(data$city)

# Xây dựng mô hình
model <- lm(
  price_on_list ~ star + avg_rating + distance_to_center +
    service + facility + cleanliness + comfort + 
    value + location + city,
  data = data
)

# Dự đoán
y_pred <- predict(model)
y <- data$price_on_list

# === 3. CHỈ SỐ ĐÁNH GIÁ ===
MAE <- mean(abs(y - y_pred))
MSE <- mean((y - y_pred)^2)
RMSE <- sqrt(MSE)
MAPE <- mean(abs((y - y_pred) / y)) * 100

# === 4. THÔNG TIN MÔ HÌNH ===
cat("\n===== MÔ HÌNH HỒI QUY ĐA BIẾN =====\n")
model_summary <- summary(model)

cat(sprintf("Multiple R       : %.4f\n", sqrt(model_summary$r.squared)))
cat(sprintf("R Square         : %.4f\n", model_summary$r.squared))
cat(sprintf("Adjusted R Square: %.4f\n", model_summary$adj.r.squared))
cat(sprintf("Standard Error   : %s\n", format(sqrt(MSE), big.mark = ",", scientific = FALSE)))
cat(sprintf("Observations     : %d\n", nrow(data)))

# === 5. BẢNG ANOVA ===
cat("\n===== BẢNG ANOVA (Type II) =====\n")
anova_table <- Anova(model, type = "II")
print(anova_table)

# === 6. HỆ SỐ HỒI QUY ===
cat("\n===== HỆ SỐ HỒI QUY =====\n")
coeff_summary <- summary(model)$coefficients
conf_int <- confint(model)

coeff_df <- data.frame(
  Coefficients = coeff_summary[, "Estimate"],
  `Std Error` = coeff_summary[, "Std. Error"],
  `t Stat` = coeff_summary[, "t value"],
  `P-value` = coeff_summary[, "Pr(>|t|)"],
  `Lower 95%` = conf_int[, 1],
  `Upper 95%` = conf_int[, 2],
  check.names = FALSE
)

print(coeff_df)

# === 7. ĐỘ CHÍNH XÁC MÔ HÌNH ===
cat("\n===== ĐỘ CHÍNH XÁC MÔ HÌNH =====\n")
cat(sprintf("MAE  = %s\n", format(MAE, big.mark = ",", scientific = FALSE)))
cat(sprintf("MSE  = %s\n", format(MSE, big.mark = ",", scientific = FALSE)))
cat(sprintf("RMSE = %s\n", format(RMSE, big.mark = ",", scientific = FALSE)))
cat(sprintf("MAPE = %.2f%%\n", MAPE))

# === 8. PHƯƠNG TRÌNH HỒI QUY ===
cat("\n===== PHƯƠNG TRÌNH HỒI QUY =====\n")
coeffs <- coef(model)
terms <- c()

for (i in 1:length(coeffs)) {
  name <- names(coeffs)[i]
  value <- coeffs[i]
  
  if (name == "(Intercept)") {
    terms <- c(terms, sprintf("%.2f", value))
  } else if (startsWith(name, "city")) {
    # Trích xuất tên thành phố
    city_name <- sub("city", "", name)
    terms <- c(terms, sprintf("%.2f * city(%s)", value, city_name))
  } else {
    terms <- c(terms, sprintf("%.2f * %s", value, name))
  }
}

formula_str <- paste("price_on_list =", paste(terms, collapse = " + "))
cat(formula_str, "\n")

# === 9. LƯU KẾT QUẢ (TÙY CHỌN) ===
# Bạn có thể lưu kết quả vào file
# write.csv(coeff_df, "regression_coefficients.csv", row.names = TRUE)
# write.csv(anova_table, "anova_table.csv", row.names = TRUE)

cat("\n=== HOÀN TẤT PHÂN TÍCH ===\n")

