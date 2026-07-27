"""
ARIMAX Order Tuning - Tìm order (p,d,q) tốt nhất cho ARIMAX
Test nhiều combinations để tìm model tối ưu
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings
from datetime import datetime, timedelta
from itertools import product

# ARIMAX library
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Thiết lập
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_and_prepare_data():
    """Load dữ liệu gốc"""
    print("=" * 70)
    print("LOAD DỮ LIỆU")
    print("=" * 70)

    # Đọc sheet giá
    df = pd.read_excel("khach_san.xlsx", sheet_name="giá_khách_sạn_theo_ngày")

    # Chuẩn hóa tên cột
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Convert dates
    df['ngày_đặt'] = pd.to_datetime(df['ngày_đặt'], errors='coerce')

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

    # Tạo features
    df['is_weekend'] = (df['ngày_đặt'].dt.dayofweek >= 5).astype(int)
    df['month'] = df['ngày_đặt'].dt.month
    df['day_of_week'] = df['ngày_đặt'].dt.dayofweek

    # Mùa cao điểm Việt Nam
    df['is_tet'] = ((df['month'] >= 1) & (df['month'] <= 2)).astype(int)
    df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
    df['is_holiday_season'] = ((df['month'] == 4) | (df['month'] == 9)).astype(int)

    print(f"✅ Đã load {len(df)} records, {df['name'].nunique()} hotels")

    return df


def prepare_hotel_time_series(hotel_data):
    """Chuẩn bị time series cho 1 hotel"""
    ts_data = hotel_data.sort_values('ngày_đặt').copy()
    ts_data = ts_data.set_index('ngày_đặt')
    
    # Đảm bảo có đầy đủ các ngày (fill missing dates)
    date_range = pd.date_range(start=ts_data.index.min(), end=ts_data.index.max(), freq='D')
    ts_data = ts_data.reindex(date_range)
    
    # Forward fill cho missing values
    ts_data['price'] = ts_data['price'].fillna(method='ffill')
    ts_data['is_weekend'] = ts_data['is_weekend'].fillna(method='ffill').fillna(0).astype(int)
    ts_data['is_tet'] = ts_data['is_tet'].fillna(method='ffill').fillna(0).astype(int)
    ts_data['is_summer'] = ts_data['is_summer'].fillna(method='ffill').fillna(0).astype(int)
    ts_data['is_holiday_season'] = ts_data['is_holiday_season'].fillna(method='ffill').fillna(0).astype(int)
    
    return ts_data


def test_arimax_order(hotel_name, hotel_data, order, train_ratio=0.8):
    """Test 1 order configuration cho ARIMAX"""
    
    # Prepare time series
    ts_data = prepare_hotel_time_series(hotel_data)
    
    if len(ts_data) < 30:
        return None
    
    # Train/test split
    split_idx = int(len(ts_data) * train_ratio)
    train = ts_data.iloc[:split_idx]
    test = ts_data.iloc[split_idx:]
    
    if len(test) < 5:
        return None
    
    # Exogenous variables
    exog_features = ['is_weekend', 'is_tet', 'is_summer', 'is_holiday_season']
    
    try:
        # Train model
        model = SARIMAX(
            train['price'],
            exog=train[exog_features],
            order=order,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False, maxiter=200)
        
        # Predict
        test_pred = results.forecast(steps=len(test), exog=test[exog_features])
        
        # Metrics
        test_mae = np.mean(np.abs(test['price'].values - test_pred.values))
        test_mape = np.mean(np.abs((test['price'].values - test_pred.values) / test['price'].values)) * 100
        test_rmse = np.sqrt(np.mean((test['price'].values - test_pred.values) ** 2))
        
        return {
            'hotel': hotel_name,
            'order': order,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'test_rmse': test_rmse,
            'aic': results.aic,
            'bic': results.bic,
            'observations': len(ts_data)
        }
        
    except Exception as e:
        return None


def grid_search_orders(hotel_name, hotel_data, p_range, d_range, q_range):
    """Grid search để tìm order tốt nhất"""
    
    print(f"\n{'='*70}")
    print(f"Testing hotel: {hotel_name}")
    print(f"{'='*70}")
    
    results = []
    total_tests = len(p_range) * len(d_range) * len(q_range)
    current = 0
    
    for p, d, q in product(p_range, d_range, q_range):
        current += 1
        order = (p, d, q)
        
        print(f"  [{current}/{total_tests}] Testing order {order}...", end=' ')
        
        result = test_arimax_order(hotel_name, hotel_data, order)
        
        if result:
            results.append(result)
            print(f"✅ MAPE={result['test_mape']:.2f}%, AIC={result['aic']:.0f}")
        else:
            print("❌ Failed")
    
    if not results:
        return None
    
    # Sắp xếp theo MAPE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_mape')
    
    return results_df


def test_multiple_hotels(df, num_hotels=5):
    """Test nhiều hotels để tìm order tốt nhất"""
    
    print("\n" + "=" * 70)
    print("GRID SEARCH ARIMAX ORDERS")
    print("=" * 70)
    
    # Chọn hotels có đủ data
    hotel_counts = df.groupby('name').size()
    valid_hotels = hotel_counts[hotel_counts >= 50].index.tolist()
    
    print(f"\n✅ Tìm thấy {len(valid_hotels)} hotels có ≥50 observations")
    print(f"→ Sẽ test {min(num_hotels, len(valid_hotels))} hotels\n")
    
    # Test range
    p_range = [0, 1, 2]  # AR order
    d_range = [0, 1]     # Differencing
    q_range = [0, 1, 2]  # MA order
    
    all_results = []
    
    for i, hotel_name in enumerate(valid_hotels[:num_hotels], 1):
        print(f"\n{'='*70}")
        print(f"Hotel {i}/{min(num_hotels, len(valid_hotels))}: {hotel_name}")
        print(f"{'='*70}")
        
        hotel_data = df[df['name'] == hotel_name].copy()
        
        results_df = grid_search_orders(hotel_name, hotel_data, p_range, d_range, q_range)
        
        if results_df is not None:
            all_results.append(results_df)
            
            # Show top 5 for this hotel
            print(f"\n🏆 Top 5 orders cho {hotel_name}:")
            print(results_df.head(5)[['order', 'test_mape', 'test_mae', 'aic', 'bic']].to_string(index=False))
    
    return all_results


def analyze_results(all_results):
    """Phân tích kết quả tổng hợp"""
    
    print("\n" + "=" * 70)
    print("PHÂN TÍCH KẾT QUẢ TỔNG HỢP")
    print("=" * 70)
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    
    # Group by order
    order_stats = combined.groupby('order').agg({
        'test_mape': ['mean', 'median', 'std', 'count'],
        'aic': 'mean',
        'bic': 'mean'
    }).round(2)
    
    order_stats.columns = ['MAPE_mean', 'MAPE_median', 'MAPE_std', 'Count', 'AIC_mean', 'BIC_mean']
    order_stats = order_stats.sort_values('MAPE_median')
    
    print("\n📊 Kết quả theo Order (p,d,q):")
    print(order_stats.to_string())
    
    # Best order overall
    best_order = order_stats.index[0]
    best_mape = order_stats.iloc[0]['MAPE_median']
    
    print(f"\n🏆 ORDER TỐT NHẤT: {best_order}")
    print(f"   - Median MAPE: {best_mape:.2f}%")
    print(f"   - Mean MAPE: {order_stats.iloc[0]['MAPE_mean']:.2f}%")
    print(f"   - Tested on: {int(order_stats.iloc[0]['Count'])} hotels")

    # Visualization only (no CSV export)
    plot_order_comparison(order_stats)
    
    return best_order, order_stats


def plot_order_comparison(order_stats):
    """Vẽ biểu đồ so sánh các orders"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top 10 orders by MAPE
    top_10 = order_stats.head(10)
    
    # Plot 1: MAPE comparison
    ax1 = axes[0]
    x_pos = np.arange(len(top_10))
    ax1.barh(x_pos, top_10['MAPE_median'], color='steelblue', alpha=0.7)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels([str(order) for order in top_10.index])
    ax1.set_xlabel('Median MAPE (%)', fontsize=11)
    ax1.set_ylabel('Order (p,d,q)', fontsize=11)
    ax1.set_title('Top 10 ARIMAX Orders - MAPE Comparison', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add values
    for i, v in enumerate(top_10['MAPE_median']):
        ax1.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=9)
    
    # Plot 2: AIC vs BIC
    ax2 = axes[1]
    ax2.scatter(top_10['AIC_mean'], top_10['BIC_mean'], s=100, alpha=0.6, c=top_10['MAPE_median'], cmap='RdYlGn_r')
    
    for idx, order in enumerate(top_10.index):
        ax2.annotate(str(order), 
                    (top_10['AIC_mean'].iloc[idx], top_10['BIC_mean'].iloc[idx]),
                    fontsize=8, ha='right')
    
    ax2.set_xlabel('AIC (Mean)', fontsize=11)
    ax2.set_ylabel('BIC (Mean)', fontsize=11)
    ax2.set_title('AIC vs BIC - Top 10 Orders', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("arimax_outputs") / "order_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ: {output_file}")
    plt.close()


if __name__ == "__main__":
    try:
        # Load data
        df = load_and_prepare_data()
        
        # Test multiple hotels
        all_results = test_multiple_hotels(df, num_hotels=10)
        
        if all_results:
            # Analyze
            best_order, order_stats = analyze_results(all_results)
            
            print("\n" + "=" * 70)
            print("✅ HOÀN THÀNH!")
            print("=" * 70)
            print(f"\n💡 Khuyến nghị: Sử dụng order={best_order} cho ARIMAX model")
            print(f"\n📊 Biểu đồ so sánh: arimax_outputs/order_comparison.png")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

