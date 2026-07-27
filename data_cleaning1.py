import pandas as pd

# Cell 1: Read CSV file
df = pd.read_csv(r"C:\Users\admin\Downloads\booking_multi_city_2025-10-17_to_2026-03-31_v16.xls")
print(df)

# Cell 2: Count hotels by name
thong_ke_khach_san = df['name'].value_counts()
print(thong_ke_khach_san)

# Cell 3: Count unique hotels by city
so_khach_san_duy_nhat = df.groupby('city')['name'].nunique()
print(so_khach_san_duy_nhat)

# Cell 4: Filter hotels with at least 15 occurrences
input_filename = 'booking_multi_city_2025-10-17_to_2026-03-31_v16.csv' 
output_filename = 'khach_san_da_loc.csv' 

try:
    df = pd.read_csv(input_filename)
    print(f"Đã đọc thành công tệp '{input_filename}'.")
    print(f"Số lượng khách sạn ban đầu: {len(df)}")

    name_counts = df['name'].value_counts()

    df_filtered = df[df['name'].isin(name_counts[name_counts >= 15].index)]

    df_filtered.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print("Hoàn tất quá trình lọc!")
    print(f"Số lượng khách sạn sau khi lọc: {len(df_filtered)}")
    print(f"Đã lưu kết quả vào tệp '{output_filename}'.")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp có tên '{input_filename}'.")
    print("Vui lòng kiểm tra lại tên tệp và đảm bảo nó nằm cùng thư mục với mã của bạn.")
except KeyError:
    print("Lỗi: Không tìm thấy cột 'name' trong tệp CSV.")
    print("Vui lòng đảm bảo tệp của bạn có một cột với tiêu đề chính xác là 'name'.")

# Cell 5: Create unique hotel list with statistics
ten_file_goc = 'khach_san_da_loc.csv'
ten_file_moi = 'danh_sach_khach_san_day_du.csv'

try:
    df = pd.read_csv(ten_file_goc)
    print(f"✅ Đã đọc file '{ten_file_goc}' thành công. Tìm thấy {len(df)} dòng.")
    df['so_lan_lap_lai'] = df.groupby('name')['name'].transform('count')
    print("✅ Đã thêm cột thống kê số lần lặp lại.")

    cac_cot_can_xuat = [
        'city',
        'name',
        'so_lan_lap_lai', 
        'hotel_id',
        'avg_rating',
        'num_reviews',
        'distance_to_center',
        'listing_url'
    ]
    cac_cot_hien_co = [cot for cot in cac_cot_can_xuat if cot in df.columns]

    cac_cot_bi_thieu = set(cac_cot_can_xuat) - set(cac_cot_hien_co)
    if cac_cot_bi_thieu:
        print(f"⚠️ Lưu ý: Các cột sau không tìm thấy và sẽ được bỏ qua: {list(cac_cot_bi_thieu)}")

    df_khong_trung_lap = df.drop_duplicates(subset=['name'], keep='first')
    print(f"✅ Đã lọc bỏ các tên khách sạn trùng lặp, còn lại {len(df_khong_trung_lap)} khách sạn duy nhất.")

    df_xuat = df_khong_trung_lap[cac_cot_hien_co]

    df_xuat.to_csv(ten_file_moi, index=False, encoding='utf-8-sig')

    print(f"🎉 Hoàn tất! Đã lưu thành công file '{ten_file_moi}'.")

except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file có tên '{ten_file_goc}'.")
    print("👉 Vui lòng kiểm tra lại 2 điều:")
    print("   1. Bạn đã điền đúng tên file chưa (phân biệt cả chữ hoa-thường)?")
    print("   2. File CSV có nằm cùng thư mục với file Jupyter Notebook này không?")
except Exception as e:
    print(f"❌ Đã có lỗi xảy ra: {e}")

