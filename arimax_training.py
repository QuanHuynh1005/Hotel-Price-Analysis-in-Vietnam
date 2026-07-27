"""
ARIMAX Training - Train ARIMAX model cho từng hotel
Dự đoán giá phòng dựa trên ngày check-in (time series)

Model: ARIMAX(2,1,1) với exogenous variables
- AR(2): AutoRegressive order 2 - học từ 2 lags trước
- I(1): Integrated order 1 - differencing 1 lần
- MA(1): Moving Average order 1 - học từ 1 error lag
- X: Exogenous variables (is_weekend, is_tet, is_summer, is_holiday_season)

Note: Dùng ARIMAX thay vì SARIMAX vì dữ liệu chưa đủ cho seasonality
Order (2,1,1) được chọn sau grid search 18 combinations (Median MAPE = 22.97%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import pickle
import warnings
from datetime import datetime, timedelta

# ARIMAX library
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Thiết lập
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_and_prepare_data():
    """Load dữ liệu gốc và chuẩn bị time series"""
    print("=" * 70)
    print("BƯỚC 1: LOAD VÀ CHUẨN BỊ DỮ LIỆU")
    print("=" * 70)
    
    # Đọc sheet giá
    df = pd.read_excel("khach_san.xlsx", sheet_name="giá_khách_sạn_theo_ngày")
    
    print(f"\n✅ Đã đọc {len(df):,} dòng")
    
    # Chuẩn hóa tên cột
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    
    # Convert dates
    df['ngày_đặt'] = pd.to_datetime(df['ngày_đặt'], errors='coerce')
    df['ngày_trả'] = pd.to_datetime(df['ngày_trả'], errors='coerce')
    
    # Clean price
    def clean_price(price_str):
        if pd.isna(price_str):
            return np.nan
        s = str(price_str).replace('VND', '').replace('.', '').replace(',', '').replace(' ', '').strip()
        try:
            return float(s)
        except:
            return np.nan
    
    df['price'] = df['price_on_list'].apply(clean_price)
    
    # Filter valid data
    df = df[df['price'].notna() & df['ngày_đặt'].notna()].copy()
    
    print(f"✅ Sau lọc: {len(df):,} dòng")
    print(f"✅ Số hotels unique: {df['name'].nunique()}")
    print(f"✅ Ngày check-in: {df['ngày_đặt'].min().date()} → {df['ngày_đặt'].max().date()}")
    
    return df


def create_time_series_features(df):
    """Tạo features từ ngày check-in"""
    print("\n" + "=" * 70)
    print("BƯỚC 2: TẠO TIME SERIES FEATURES")
    print("=" * 70)
    
    # Time features
    df['month'] = df['ngày_đặt'].dt.month
    df['day'] = df['ngày_đặt'].dt.day
    df['day_of_week'] = df['ngày_đặt'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['week_of_year'] = df['ngày_đặt'].dt.isocalendar().week
    
    # Mùa cao điểm Việt Nam
    df['is_tet'] = ((df['month'] >= 1) & (df['month'] <= 2)).astype(int)
    df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
    df['is_holiday_season'] = ((df['month'] == 4) | (df['month'] == 9)).astype(int)
    
    print(f"\n✅ Đã tạo 8 time features:")
    print(f"   - month, day, day_of_week, week_of_year")
    print(f"   - is_weekend, is_tet, is_summer, is_holiday_season")
    
    return df


def select_hotels_for_training(df, min_observations=30):
    """Chọn hotels có đủ dữ liệu để train ARIMAX"""
    print("\n" + "=" * 70)
    print("BƯỚC 3: LỌC HOTELS CÓ ĐỦ DỮ LIỆU")
    print("=" * 70)
    
    # Đếm số observations cho mỗi hotel
    hotel_counts = df.groupby('name').size()
    
    # Lọc hotels có >= min_observations
    valid_hotels = hotel_counts[hotel_counts >= min_observations].index.tolist()
    
    print(f"\n📊 Thống kê:")
    print(f"   Tổng số hotels: {len(hotel_counts)}")
    print(f"   Hotels có >={min_observations} observations: {len(valid_hotels)}")
    print(f"   → Sẽ train ARIMAX cho {len(valid_hotels)} hotels")
    
    # Phân phối
    print(f"\n📈 Phân phối số observations:")
    print(f"   Min: {hotel_counts.min()}")
    print(f"   Max: {hotel_counts.max()}")
    print(f"   Mean: {hotel_counts.mean():.1f}")
    print(f"   Median: {hotel_counts.median():.0f}")
    
    # Top 5 hotels
    top_5 = hotel_counts.nlargest(5)
    print(f"\n🏆 Top 5 hotels có nhiều dữ liệu nhất:")
    for name, count in top_5.items():
        print(f"   {name[:50]:50s}: {count} observations")
    
    return valid_hotels


def prepare_hotel_time_series(hotel_data):
    """Chuẩn bị time series cho 1 hotel"""
    # Sắp xếp theo ngày
    hotel_data = hotel_data.sort_values('ngày_đặt').copy()
    
    # Set index là ngày check-in
    hotel_data = hotel_data.set_index('ngày_đặt')
    
    # Nếu có nhiều giá cùng 1 ngày → lấy trung bình
    hotel_data = hotel_data.groupby(hotel_data.index).agg({
        'price': 'mean',
        'month': 'first',
        'day_of_week': 'first',
        'is_weekend': 'first',
        'is_tet': 'first',
        'is_summer': 'first',
        'is_holiday_season': 'first'
    })
    
    return hotel_data


def train_arimax_for_hotel(hotel_name, hotel_data, train_ratio=0.8):
    """Train ARIMAX cho 1 hotel"""
    
    # Prepare time series
    ts_data = prepare_hotel_time_series(hotel_data)
    
    # Split train/test
    train_size = int(len(ts_data) * train_ratio)
    train = ts_data.iloc[:train_size]
    test = ts_data.iloc[train_size:]
    
    if len(test) < 5:  # Không đủ test data
        return None
    
    # Exogenous variables
    exog_features = ['is_weekend', 'is_tet', 'is_summer', 'is_holiday_season']

    # ARIMAX parameters (không dùng seasonality vì dữ liệu quá ít)
    # order = (p, d, q): AR, difference, MA
    # Sau grid search 18 combinations, (2,1,1) cho kết quả tốt nhất (Median MAPE = 22.97%)
    order = (2, 1, 1)  # AR(2) + Differencing + MA(1)
    
    try:
        # Train model (ARIMAX = SARIMAX without seasonal component)
        model = SARIMAX(
            train['price'],
            exog=train[exog_features],
            order=order,
            seasonal_order=(0, 0, 0, 0),  # No seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False, maxiter=200)
        
        # Predict on test
        test_pred = results.forecast(steps=len(test), exog=test[exog_features])
        
        # Check for NaN
        if test_pred.isna().any() or test['price'].isna().any():
            print(f"   ⚠️  NaN values detected, skipping...")
            return None
        
        # Metrics
        test_mae = np.mean(np.abs(test['price'].values - test_pred.values))
        test_mape = np.mean(np.abs((test['price'].values - test_pred.values) / test['price'].values)) * 100
        test_rmse = np.sqrt(np.mean((test['price'].values - test_pred.values) ** 2))
        
        return {
            'hotel_name': hotel_name,
            'model': results,
            'train_size': len(train),
            'test_size': len(test),
            'test_mae': test_mae,
            'test_mape': test_mape,
            'test_rmse': test_rmse,
            'aic': results.aic,
            'bic': results.bic,
            'date_range': (ts_data.index.min(), ts_data.index.max()),
            'exog_features': exog_features
        }
        
    except Exception as e:
        print(f"   ❌ Lỗi khi train {hotel_name[:40]}: {str(e)[:50]}")
        return None


def train_all_hotels(df, valid_hotels, max_hotels=50):
    """Train ARIMAX cho nhiều hotels"""
    print("\n" + "=" * 70)
    print("BƯỚC 4: TRAIN ARIMAX CHO TỪNG HOTEL")
    print("=" * 70)
    
    # Giới hạn số hotels để train (nếu quá nhiều)
    if len(valid_hotels) > max_hotels:
        print(f"\n⚠️  Có {len(valid_hotels)} hotels, chỉ train {max_hotels} đầu tiên")
        print(f"   (Bỏ comment để train tất cả)")
        valid_hotels = valid_hotels[:max_hotels]
    
    results = []
    
    for i, hotel_name in enumerate(valid_hotels, 1):
        print(f"\n[{i}/{len(valid_hotels)}] Training: {hotel_name[:50]}...")
        
        hotel_data = df[df['name'] == hotel_name].copy()

        result = train_arimax_for_hotel(hotel_name, hotel_data)
        
        if result:
            results.append(result)
            print(f"   ✅ MAE: {result['test_mae']:>8.0f} | MAPE: {result['test_mape']:>6.1f}% | Train: {result['train_size']} | Test: {result['test_size']}")
        else:
            print(f"   ⚠️  Skipped (không đủ dữ liệu hoặc lỗi)")
    
    print(f"\n{'=' * 70}")
    print(f"✅ Đã train thành công {len(results)}/{len(valid_hotels)} hotels")
    print(f"{'=' * 70}")
    
    return results


def display_results(results):
    """Hiển thị kết quả tổng hợp"""
    print("\n" + "=" * 70)
    print("BƯỚC 5: TỔNG HỢP KẾT QUẢ")
    print("=" * 70)
    
    if len(results) == 0:
        print("\n❌ Không có model nào được train thành công!")
        return
    
    # Tạo DataFrame
    df_results = pd.DataFrame([
        {
            'Hotel': r['hotel_name'][:40],
            'Train': r['train_size'],
            'Test': r['test_size'],
            'MAE': r['test_mae'],
            'MAPE': r['test_mape'],
            'RMSE': r['test_rmse'],
            'AIC': r['aic']
        }
        for r in results
    ])
    
    # Sort by MAE
    df_results = df_results.sort_values('MAE')
    
    print(f"\n📊 Top 10 models tốt nhất (theo MAE):\n")
    print(df_results.head(10).to_string(index=False))
    
    # Thống kê
    print(f"\n📈 THỐNG KÊ TỔNG QUAN:")
    print(f"   Số models: {len(results)}")
    print(f"   Mean MAE: {df_results['MAE'].mean():,.0f} VND")
    print(f"   Median MAE: {df_results['MAE'].median():,.0f} VND")
    print(f"   Mean MAPE: {df_results['MAPE'].mean():.1f}%")
    print(f"   Median MAPE: {df_results['MAPE'].median():.1f}%")
    
    # Best model
    best_idx = df_results['MAE'].idxmin()
    
    if pd.notna(best_idx):
        best = df_results.loc[best_idx]
        
        print(f"\n🏆 BEST MODEL:")
        print(f"   Hotel: {best['Hotel']}")
        print(f"   MAE: {best['MAE']:,.0f} VND")
        print(f"   MAPE: {best['MAPE']:.1f}%")
    else:
        print(f"\n⚠️  Không có model tốt nhất (tất cả đều có lỗi)")


def save_models(results, output_dir="arimax_models"):
    """Lưu tất cả models"""
    print("\n" + "=" * 70)
    print("BƯỚC 6: LƯU MODELS")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Lưu từng model
    for i, result in enumerate(results):
        # Tạo tên file an toàn
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in result['hotel_name'])
        safe_name = safe_name[:50]  # Giới hạn độ dài
        
        model_file = output_path / f"{safe_name}.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(result, f)
    
    print(f"\n✅ Đã lưu {len(results)} models vào: {output_path.absolute()}")
    
    # Lưu metadata
    metadata = {
        'num_models': len(results),
        'hotels': [r['hotel_name'] for r in results],
        'exog_features': results[0]['exog_features'] if len(results) > 0 else [],
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_file = output_path / "metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Đã lưu metadata: {metadata_file}")
    
    # Lưu summary
    summary_file = output_path / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ARIMAX MODELS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Trained on: {metadata['train_date']}\n")
        f.write(f"Number of models: {len(results)}\n\n")
        
        f.write("Models:\n")
        for i, r in enumerate(sorted(results, key=lambda x: x['test_mae']), 1):
            f.write(f"{i:3d}. {r['hotel_name'][:50]:50s} | MAE: {r['test_mae']:>8.0f} | MAPE: {r['test_mape']:>6.1f}%\n")
    
    print(f"✅ Đã lưu summary: {summary_file}")


def plot_sample_predictions(results, df, num_samples=3):
    """Vẽ ví dụ predictions cho một vài hotels"""
    print("\n" + "=" * 70)
    print("BƯỚC 7: VẼ VÍ DỤ PREDICTIONS")
    print("=" * 70)

    output_dir = Path("arimax_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Chọn 3 models tốt nhất
    sorted_results = sorted(results, key=lambda x: x['test_mae'])[:num_samples]
    
    for i, result in enumerate(sorted_results, 1):
        hotel_name = result['hotel_name']
        hotel_data = df[df['name'] == hotel_name].copy()
        ts_data = prepare_hotel_time_series(hotel_data)
        
        # Train/test split
        train_size = result['train_size']
        train = ts_data.iloc[:train_size]
        test = ts_data.iloc[train_size:]
        
        # Predict
        model = result['model']
        exog_features = result['exog_features']
        test_pred = model.forecast(steps=len(test), exog=test[exog_features])
        
        # Plot
        plt.figure(figsize=(14, 6))
        
        plt.plot(train.index, train['price'], label='Train', linewidth=2)
        plt.plot(test.index, test['price'], label='Test (Actual)', linewidth=2, color='orange')
        plt.plot(test.index, test_pred, label='Test (Predicted)', linewidth=2, linestyle='--', color='red')
        
        plt.title(f'{hotel_name[:60]}\nMAE: {result["test_mae"]:.0f} VND | MAPE: {result["test_mape"]:.1f}%')
        plt.xlabel('Check-in Date')
        plt.ylabel('Price (VND)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        file_name = output_dir / f"sample_{i}.png"
        plt.savefig(file_name, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ {file_name}")
    
    print(f"\n✅ Đã lưu {num_samples} biểu đồ vào: {output_dir.absolute()}")


def main():
    """Hàm chính"""
    print("=" * 70)
    print("ARIMAX TRAINING - DỰ ĐOÁN GIÁ THEO NGÀY CHECK-IN")
    print("=" * 70)
    
    try:
        # Bước 1: Load data
        df = load_and_prepare_data()
        
        # Bước 2: Create features
        df = create_time_series_features(df)
        
        # Bước 3: Select hotels
        valid_hotels = select_hotels_for_training(df, min_observations=30)
        
        if len(valid_hotels) == 0:
            print("\n❌ Không có hotel nào đủ dữ liệu để train!")
            return
        
        # Bước 4: Train models
        results = train_all_hotels(df, valid_hotels, max_hotels=20)
        
        if len(results) == 0:
            print("\n❌ Không có model nào được train thành công!")
            return
        
        # Bước 5: Display results
        display_results(results)
        
        # Bước 6: Save models
        save_models(results)
        
        # Bước 7: Plot samples
        plot_sample_predictions(results, df, num_samples=min(3, len(results)))
        
        print("\n" + "=" * 70)
        print("✅ HOÀN THÀNH!")
        print("=" * 70)
        print(f"\nBước tiếp theo: Chạy 'arimax_prediction.py' để dự đoán giá tương lai")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

