# Phan tich ANOVA - So sanh gia khach san theo hang sao

# Thiet lap encoding UTF-8
options(encoding = "UTF-8")

# --- Cai dat va load thu vien ---
# install.packages(c("readxl", "dplyr", "tidyr", "knitr"))

library(readxl)
library(dplyr)
library(tidyr)
library(knitr)

# --- Ham dinh dang so ---
format_number <- function(x) {
  format(x, big.mark = ",", scientific = FALSE, trim = TRUE)
}

# --- Doc file ---
# Thiet lap thu muc lam viec
setwd("E:/Kha/UIT/hk1_2025")

file_path <- "khach_san.xlsx"  # Thay doi duong dan neu can

# Kiem tra file ton tai
if (!file.exists(file_path)) {
  cat("Thu muc hien tai:", getwd(), "\n")
  cat("Cac file co san:", paste(list.files(), collapse = ", "), "\n")
  stop("Khong tim thay file: ", file_path, 
       "\nVui long dat file 'khach_san.xlsx' trong thu muc lam viec hien tai.")
}

# Doc du lieu - dung sheet index thay vi ten sheet
sheet1 <- read_excel(file_path, sheet = 1)
sheet2 <- read_excel(file_path, sheet = 2)

# --- Lam sach ten cot ---
names(sheet1) <- trimws(names(sheet1))
names(sheet2) <- trimws(names(sheet2))

# --- Lam sach cot gia ---
sheet1 <- sheet1 %>%
  mutate(
    price_on_list = gsub("VND", "", price_on_list, fixed = TRUE),
    price_on_list = gsub("\\.", "", price_on_list),
    price_on_list = gsub("\u00A0", "", price_on_list, fixed = TRUE),  # Remove non-breaking space
    price_on_list = as.numeric(trimws(price_on_list))
  )

# --- Gop du lieu ---
data <- inner_join(sheet1, sheet2, by = "hotel_id")

# --- Loc du lieu hop le ---
data <- data %>%
  select(hotel_id, price_on_list, star) %>%
  drop_na() %>%
  filter(price_on_list > 0) %>%
  mutate(star = as.integer(star))

# --- Kiem tra nhom sao ---
cat("Nhom sao:", unique(data$star), "\n")

# ===== BANG SO LUONG MOI NHOM =====
group_count <- data %>%
  group_by(star) %>%
  summarise(So_luong_khach_san = n()) %>%
  rename(So_sao = star)

cat("\n===== BANG SO LUONG MOI NHOM =====\n")
print(kable(group_count, format = "pipe"))

# --- ANOVA ---
# Chuyen star thanh factor de ANOVA
data$star <- as.factor(data$star)

# Thuc hien ANOVA
anova_model <- aov(price_on_list ~ star, data = data)
anova_summary <- summary(anova_model)

# Lay bang ANOVA
anova_table <- anova_summary[[1]]

# Tinh toan cac gia tri
SS_between <- anova_table$"Sum Sq"[1]
SS_within <- anova_table$"Sum Sq"[2]
df_between <- anova_table$Df[1]
df_within <- anova_table$Df[2]
MS_between <- anova_table$"Mean Sq"[1]
MS_within <- anova_table$"Mean Sq"[2]
F_value <- anova_table$"F value"[1]
p_value <- anova_table$"Pr(>F)"[1]

# F critical
f_critical <- qf(0.95, df_between, df_within)

# Tao bang hien thi
anova_display <- data.frame(
  Source = c("C(star)", "Residual"),
  SS = c(format_number(SS_between), format_number(SS_within)),
  df = c(df_between, df_within),
  MS = c(format_number(MS_between), format_number(MS_within)),
  F = c(sprintf("%.4f", F_value), ""),
  P_value = c(sprintf("%.6e", p_value), ""),
  check.names = FALSE
)

cat("\n===== BANG KET QUA ANOVA =====\n")
print(kable(anova_display, format = "pipe", align = "l"))
cat(sprintf("\nF critical (alpha = 0.05): %.4f\n", f_critical))

# --- Ket luan ANOVA ---
if (p_value < 0.05) {
  cat("\n>> Co su khac biet co y nghia thong ke giua cac nhom sao (p < 0.05)\n")
  
  # --- KIEM DINH HSD THU CONG ---
  cat("\n===== KIEM DINH HSD THU CONG =====\n")
  
  # Tinh HSD
  MSE <- MS_within
  k <- length(unique(data$star))
  n_per_group <- mean(table(data$star))
  
  # q critical tu phan phoi studentized range
  q_critical <- qtukey(0.95, k, df_within)
  
  HSD <- q_critical * sqrt(MSE / n_per_group)
  cat(sprintf("HSD (nguong chenh lech can dat) = %s\n", format_number(round(HSD, 2))))
  
  # --- Trung binh tung nhom ---
  means <- data %>%
    group_by(star) %>%
    summarise(mean_price = mean(price_on_list)) %>%
    arrange(star)
  
  means_display <- means %>%
    mutate(price_on_list = format_number(round(mean_price, 0))) %>%
    select(star, price_on_list)
  
  cat("\nTrung binh gia theo sao:\n")
  print(kable(means_display, format = "pipe", align = "l"))
  
  # --- So sanh tung cap ---
  pairs <- data.frame()
  
  for (i in 1:(nrow(means) - 1)) {
    for (j in (i + 1):nrow(means)) {
      diff <- abs(means$mean_price[i] - means$mean_price[j])
      significant <- diff > HSD
      
      pairs <- rbind(pairs, data.frame(
        group1 = as.character(means$star[i]),
        group2 = as.character(means$star[j]),
        mean_diff = diff,
        Significant = ifelse(significant, "Co khac biet", "Khong khac biet")
      ))
    }
  }
  
  # --- In bang dep ---
  pairs_display <- pairs %>%
    mutate(mean_diff = format_number(round(mean_diff, 0)))
  
  cat("\n===== SO SANH THU CONG THEO HSD =====\n")
  print(kable(pairs_display, format = "pipe", align = "l", row.names = FALSE))
  
  # --- Ket luan cuoi ---
  sig_pairs <- pairs %>% filter(Significant == "Co khac biet")
  
  if (nrow(sig_pairs) > 0) {
    cat("\n>> Cac cap nhom co khac biet theo HSD:\n")
    for (i in 1:nrow(sig_pairs)) {
      cat(sprintf("   - Nhom %s va nhom %s khac biet (Delta = %.2f)\n",
                  sig_pairs$group1[i], sig_pairs$group2[i], sig_pairs$mean_diff[i]))
    }
  } else {
    cat("\nKhong co cap nhom nao khac biet theo HSD.\n")
  }
  
} else {
  cat("\nKhong co su khac biet dang ke giua cac nhom sao (p >= 0.05)\n")
}

cat("\n=== Hoan tat phan tich ===\n")
