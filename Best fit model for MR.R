# Mo hinh hoi quy tuyen tinh da bien - Du doan gia khach san
# Su dung Backward Elimination voi kiem tra Adjusted R^2
# 
# LUU Y: Khong su dung 6 bien rating rieng le (service, facility, cleanliness, 
# comfort, value, location) vi cac bien nay co correlation > 0.9 voi nhau,
# gay ra MULTICOLLINEARITY nghiem trong va lam he so bi nghich ly.
# CHI SU DUNG: avg_rating (trung binh cua 6 bien tren)

library(readxl)
library(dplyr)
library(tidyr)

# Thiet lap thu muc lam viec
setwd("E:/Kha/UIT/hk1_2025")

# === HAM XU LY KHOANG CACH ===
clean_distance <- function(x) {
  if (is.na(x)) return(NA)
  
  x <- tolower(as.character(x))
  x <- gsub(",", ".", x)
  
  if (grepl("m", x) && !grepl("km", x)) {
    # Truong hop "500m"
    num <- as.numeric(gsub("[^0-9.]", "", x))
    return(num / 1000)
  } else if (grepl("km", x)) {
    # Truong hop "5km"
    num <- as.numeric(gsub("[^0-9.]", "", x))
    return(num)
  }
  
  return(NA)
}

# === BACKWARD ELIMINATION VOI ADJUSTED R^2 ===
backward_elimination_r2 <- function(data, dependent_var, significance_level = 0.05) {
  
  cat("\n===== QUA TRINH LOAI BO BIEN =====\n")
  
  # Lay tat ca cac bien doc lap
  independent_vars <- setdiff(names(data), dependent_var)
  current_vars <- independent_vars
  
  best_adj_r2 <- -Inf
  improved <- TRUE
  
  while (improved && length(current_vars) > 0) {
    improved <- FALSE
    
    # Tao formula
    formula_str <- paste(dependent_var, "~", paste(current_vars, collapse = " + "))
    
    # Chay mo hinh
    model <- lm(as.formula(formula_str), data = data)
    best_adj_r2 <- summary(model)$adj.r.squared
    
    # Lay p-values
    p_values <- summary(model)$coefficients[, "Pr(>|t|)"]
    p_values <- p_values[names(p_values) != "(Intercept)"]
    
    # Loai bo NA values
    p_values <- p_values[!is.na(p_values)]
    
    if (length(p_values) == 0) break
    
    max_p_value <- max(p_values, na.rm = TRUE)
    excluded_var <- names(which.max(p_values))
    
    if (max_p_value > significance_level) {
      # Thu loai bo bien
      temp_vars <- setdiff(current_vars, excluded_var)
      
      if (length(temp_vars) == 0) break
      
      temp_formula <- paste(dependent_var, "~", paste(temp_vars, collapse = " + "))
      temp_model <- lm(as.formula(temp_formula), data = data)
      temp_adj_r2 <- summary(temp_model)$adj.r.squared
      
      if (temp_adj_r2 >= best_adj_r2) {
        cat(sprintf("[+] Loai '%s' (p=%.4f), Adjusted R^2 tang len %.4f\n", 
                    excluded_var, max_p_value, temp_adj_r2))
        current_vars <- temp_vars
        best_model <- temp_model
        improved <- TRUE
      } else {
        cat(sprintf("[-] Giu '%s' vi loai bo lam giam Adjusted R^2 (%.4f < %.4f)\n",
                    excluded_var, temp_adj_r2, best_adj_r2))
        break
      }
    } else {
      break
    }
  }
  
  # Tra ve mo hinh tot nhat
  final_formula <- paste(dependent_var, "~", paste(current_vars, collapse = " + "))
  final_model <- lm(as.formula(final_formula), data = data)
  
  return(final_model)
}

# === MAIN PROGRAM ===
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("MO HINH HOI QUY TUYEN TINH DA BIEN - DU DOAN GIA KHACH SAN\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# === 1. DOC & LAM SACH DU LIEU ===
file_path <- "khach_san.xlsx"

if (!file.exists(file_path)) {
  stop("Khong tim thay file: ", file_path)
}

# Doc du lieu
sheet1 <- read_excel(file_path, sheet = 1)
sheet2 <- read_excel(file_path, sheet = 2)

# Lam sach ten cot
names(sheet1) <- trimws(names(sheet1))
names(sheet2) <- trimws(names(sheet2))

# Lam sach gia
sheet1 <- sheet1 %>%
  mutate(
    price_on_list = gsub("VND", "", price_on_list, fixed = TRUE),
    price_on_list = gsub("\\.", "", price_on_list),
    price_on_list = gsub("\u00A0", "", price_on_list, fixed = TRUE),
    price_on_list = as.numeric(trimws(price_on_list))
  )

# Xu ly khoang cach
sheet2$distance_to_center <- sapply(sheet2$distance_to_center, clean_distance)

# Gop du lieu
data <- inner_join(sheet1, sheet2, by = "hotel_id")

# DOI TEN CAC COT RATING BANG INDEX
# Sau merge, biet chac rang 6 cot rating nam o vi tri 15-20
# (sau khi loai tru cot 21 la listing_url)
n_cols <- length(names(data))

# Doi ten bang index: cot 15-20 la cac cot rating
# Cot 15: service, 16: facility, 17: cleanliness, 18: comfort, 19: value, 20: location
if (n_cols >= 20) {
  # Tim cot dau tien chua ten tieng Viet (bat dau tu cot 8 la star)
  # Rating cols bat dau sau distance_to_center
  idx_star <- which(names(data) == "star")[1]
  
  # Rating columns la 6 cot sau star (vi tri idx_star + 1 den idx_star + 6)
  if (!is.na(idx_star) && (idx_star + 6) <= n_cols) {
    names(data)[idx_star + 1] <- "service"
    names(data)[idx_star + 2] <- "facility"
    names(data)[idx_star + 3] <- "cleanliness"
    names(data)[idx_star + 4] <- "comfort"
    names(data)[idx_star + 5] <- "value"
    names(data)[idx_star + 6] <- "location"
  }
}

# LOAI BO CAC COT KHONG CAN THIET
data <- data %>%
  select(-any_of(c("listing_url", "scraped_at", "so_lan_lap_lai", "num_reviews")))

# Xu ly city.x / city.y (R dung dau cham, khong phai gach duoi)
if ("city.x" %in% names(data) && "city.y" %in% names(data)) {
  data$city <- ifelse(is.na(data$city.x), data$city.y, data$city.x)
} else if ("city.x" %in% names(data)) {
  data$city <- data$city.x
} else if ("city.y" %in% names(data)) {
  data$city <- data$city.y
}

# CHI GIU LAI CAC COT CAN THIET - KHONG DUNG 6 BIEN RATING RIENG LE
# VI CAC BIEN RATING CO CORRELATION > 0.9 (MULTICOLLINEARITY NGHIEM TRONG)
# CHI DUNG avg_rating DE TRANH HE SO BI NGHICH LY
data <- data %>%
  select(price_on_list, city, star, avg_rating, distance_to_center) %>%
  drop_na()

# Loc gia > 0
data <- data %>% filter(price_on_list > 0)

cat(sprintf("\n[OK] So luong mau hop le: %d\n", nrow(data)))
cat(sprintf("[*] Cac cot: %s\n", paste(names(data), collapse = ", ")))

# === 2. CHUAN BI DU LIEU HOI QUY ===
# Tao bien gia cho city (dummy variables)
data$city <- as.factor(data$city)
data_encoded <- data

# One-hot encoding (tru baseline)
city_dummies <- model.matrix(~ city - 1, data = data)[, -1]

# Doi ten thanh pho sang khong dau
city_names <- levels(data$city)[-1]

# Thay the ky tu co dau
for (i in 1:length(city_names)) {
  name <- city_names[i]
  # Loai bo ky tu dac biet va dau
  name <- iconv(name, to = "ASCII//TRANSLIT", sub = "")
  if (is.na(name)) {
    # Neu khong chuyen duoc, dung index
    name <- paste0("City_", i)
  }
  # Thay the khoang trang va ky tu dac biet
  name <- gsub("[^A-Za-z0-9]", "_", name)
  city_names[i] <- name
}

colnames(city_dummies) <- paste0("city_", city_names)

data_model <- data %>%
  select(-city) %>%
  bind_cols(as.data.frame(city_dummies))

# === 3. CHAY BACKWARD ELIMINATION ===
best_model <- backward_elimination_r2(data_model, "price_on_list")

# === 4. TONG HOP MO HINH ===
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("MO HINH TOT NHAT\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
print(summary(best_model))

# === 5. DANH GIA MO HINH ===
y_pred <- predict(best_model, data_model)
y_actual <- data_model$price_on_list

mae <- mean(abs(y_actual - y_pred))
mse <- mean((y_actual - y_pred)^2)
rmse <- sqrt(mse)
mape <- mean(abs((y_actual - y_pred) / y_actual)) * 100

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("DO CHINH XAC MO HINH\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat(sprintf("MAE  (Mean Absolute Error)       = %s VND\n", format(mae, big.mark = ",", scientific = FALSE)))
cat(sprintf("MSE  (Mean Squared Error)        = %s\n", format(mse, big.mark = ",", scientific = FALSE)))
cat(sprintf("RMSE (Root Mean Squared Error)   = %s VND\n", format(rmse, big.mark = ",", scientific = FALSE)))
cat(sprintf("MAPE (Mean Absolute %% Error)     = %.2f%%\n", mape))

# === 6. PHUONG TRINH HOI QUY ===
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("PHUONG TRINH HOI QUY\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

coeffs <- coef(best_model)
formula_parts <- c()

for (i in 1:length(coeffs)) {
  name <- names(coeffs)[i]
  value <- coeffs[i]
  
  if (name == "(Intercept)") {
    formula_parts <- c(formula_parts, sprintf("%.2f", value))
  } else {
    sign <- ifelse(value >= 0, "+", "")
    formula_parts <- c(formula_parts, sprintf("%s %.2f × %s", sign, value, name))
  }
}

cat("price_on_list = ", paste(formula_parts, collapse = " "), "\n")

# === 7. HE SO HOI QUY ===
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("HE SO HOI QUY CHI TIET\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

coeff_summary <- summary(best_model)$coefficients

# Loai bo cac hang co NA (singularities)
coeff_summary <- coeff_summary[!is.na(coeff_summary[, "Estimate"]), ]

coeff_df <- data.frame(
  Variable = rownames(coeff_summary),
  Coefficient = coeff_summary[, "Estimate"],
  Std_Error = coeff_summary[, "Std. Error"],
  t_Stat = coeff_summary[, "t value"],
  P_value = coeff_summary[, "Pr(>|t|)"]
)

# Them confidence intervals (chi cho cac bien khong bi NA)
conf_int <- confint(best_model)
# Chi lay cac hang ton tai trong coeff_df
conf_int_filtered <- conf_int[rownames(conf_int) %in% coeff_df$Variable, ]
coeff_df$Lower_95 <- conf_int_filtered[, 1]
coeff_df$Upper_95 <- conf_int_filtered[, 2]

print(coeff_df, row.names = FALSE)

# === 8. PHAN TICH Y NGHIA ===
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("PHAN TICH CAC BIEN QUAN TRONG (p < 0.05)\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

significant_vars <- coeff_df[coeff_df$P_value < 0.05 & coeff_df$Variable != "(Intercept)", ]
significant_vars <- significant_vars[order(significant_vars$P_value), ]

for (i in 1:nrow(significant_vars)) {
  var <- significant_vars$Variable[i]
  coef <- significant_vars$Coefficient[i]
  p_val <- significant_vars$P_value[i]
  
  direction <- ifelse(coef > 0, "TANG", "GIAM")
  cat(sprintf("  %-20s: %s %12s VND (p=%.6f)\n", 
              var, direction, format(abs(coef), big.mark = ",", scientific = FALSE), p_val))
}

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("HOAN THANH PHAN TICH\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

