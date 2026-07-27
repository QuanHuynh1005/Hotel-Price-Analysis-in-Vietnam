import pandas as pd
import sys 

prices_file = "khach_san - Copy.xlsx"
info_file = "khach_san - Copy (2).xlsx"

prices_filtered_file_by_name = "gia_khach_san_da_loc_theo_TEN.csv"
info_filtered_file_by_name = "thong_tin_khach_san_da_loc_theo_TEN.csv"

print("--- BẮT ĐẦU QUÁ TRÌNH LỌC THEO TÊN (name) TỪ FILE EXCEL ---")

try:
    print(f"Đang đọc file: {prices_file}")
    df_prices = pd.read_excel(prices_file, sheet_name=0) 
    
    print(f"Đang đọc file: {info_file}")
    df_info = pd.read_excel(info_file, sheet_name=0) 

    print("Đã đọc file thành công.")
    print(f"Số dòng ban đầu trong file 'giá' (file 1): {len(df_prices)}")
    print(f"Số dòng ban đầu trong file 'thông tin' (file 2): {len(df_info)}")
    print("-" * 30)

    if 'star' not in df_info.columns:
        print(f"LỖI: Không tìm thấy cột 'star' trong file {info_file}")
        sys.exit()
    if 'name' not in df_info.columns:
        print(f"LỖI: Không tìm thấy cột 'name' trong file {info_file}")
        sys.exit()
    if 'name' not in df_prices.columns:
        print(f"LỖI: Không tìm thấy cột 'name' trong file {prices_file}")
        sys.exit()
    if 'so_lan_lap_lai' not in df_info.columns:
        print(f"LỖI: Không tìm thấy cột 'so_lan_lap_lai' trong file {info_file}")
        sys.exit()

    df_info['star'] = pd.to_numeric(df_info['star'], errors='coerce')
    df_info['star'] = df_info['star'].fillna(-1)
    df_info['so_lan_lap_lai'] = pd.to_numeric(df_info['so_lan_lap_lai'], errors='coerce')
    df_info['so_lan_lap_lai'] = df_info['so_lan_lap_lai'].fillna(0)
    info_to_remove = df_info[df_info['star'] == 0]
    hotels_to_remove_names = info_to_remove['name']
    
    print(f"Tìm thấy {len(info_to_remove)} khách sạn có 'star' = 0 trong file 'thông tin'.")

    expected_rows_to_remove_from_prices = info_to_remove['so_lan_lap_lai'].sum()
    print(f"Tổng 'so_lan_lap_lai' của các khách sạn này: {expected_rows_to_remove_from_prices}")
    print(f"Đây là số dòng kỳ vọng sẽ được xóa khỏi file 'giá' (file 1).")
    print("-" * 30)

    initial_prices_rows = len(df_prices)
    df_prices_filtered = df_prices[~df_prices['name'].isin(hotels_to_remove_names)]
    actual_prices_rows_removed = initial_prices_rows - len(df_prices_filtered)
    
    print(f"Đã xóa {actual_prices_rows_removed} dòng khỏi file 'giá' (file 1) (dựa trên 'name').")
  
    if actual_prices_rows_removed == expected_rows_to_remove_from_prices:
        print(f"THÀNH CÔNG: Số dòng đã xóa ({actual_prices_rows_removed}) khớp với tổng 'so_lan_lap_lai' ({expected_rows_to_remove_from_prices}).")
    else:
        print(f"CẢNH BÁO: Số dòng đã xóa ({actual_prices_rows_removed}) KHÔNG KHỚP với tổng 'so_lan_lap_lai' ({expected_rows_to_remove_from_prices}).")
    
    print("-" * 30)

    df_info_filtered = df_info[~df_info['name'].isin(hotels_to_remove_names)]
    print(f"Đã xóa {len(df_info) - len(df_info_filtered)} dòng khỏi file 'thông tin' (file 2) (dựa trên 'name').")
    print(f"Số dòng còn lại trong file 'thông tin': {len(df_info_filtered)}")
    print(f"Số dòng còn lại trong file 'giá': {len(df_prices_filtered)}")
    print("-" * 30)

    df_prices_filtered.to_csv(prices_filtered_file_by_name, index=False, encoding='utf-8-sig')
    df_info_filtered.to_csv(info_filtered_file_by_name, index=False, encoding='utf-8-sig')
    
    print(f"Đã lưu file 'giá' đã lọc vào: {prices_filtered_file_by_name} (mã hóa UTF-8)")
    print(f"Đã lưu file 'thông tin' đã lọc vào: {info_filtered_file_by_name} (mã hóa UTF-8)")
    print("--- HOÀN THÀNH ---")

except FileNotFoundError as e:
    print(f"LỖI: Không tìm thấy file. {e}")
    print("Hãy chắc chắn rằng 2 file Excel 'khach_san - Copy.xlsx' và 'khach_san - Copy (2).xlsx' nằm CÙNG THƯ MỤC với file code Python này.")
except KeyError as e:
    print(f"Lỗi: Không tìm thấy cột {e}.")
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")

