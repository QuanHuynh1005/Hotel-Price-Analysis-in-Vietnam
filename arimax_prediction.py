"""
ARIMAX Prediction - Dự đoán giá phòng trong tương lai
Sử dụng ARIMAX models đã train
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

# Thiết lập
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.family'] = 'DejaVu Sans'

# Ẩn warnings không cần thiết
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='No supported index is available')


def load_models():
    """Load tất cả ARIMAX models đã train"""
    print("=" * 70)
    print("LOAD ARIMAX MODELS")
    print("=" * 70)

    models_dir = Path("arimax_models")

    if not models_dir.exists():
        print("\n❌ Không tìm thấy thư mục arimax_models/")
        print("→ Vui lòng chạy 'arimax_training.py' trước")
        return None, None
    
    # Load metadata
    metadata_file = models_dir / "metadata.pkl"
    if not metadata_file.exists():
        print("\n❌ Không tìm thấy metadata.pkl")
        return None, None
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"\n✅ Metadata:")
    print(f"   Số models: {metadata['num_models']}")
    print(f"   Trained on: {metadata['train_date']}")
    print(f"   Exog features: {', '.join(metadata['exog_features'])}")
    
    # Load all models
    models = {}
    model_files = list(models_dir.glob("*.pkl"))
    model_files = [f for f in model_files if f.name != "metadata.pkl"]
    
    for model_file in model_files:
        with open(model_file, 'rb') as f:
            result = pickle.load(f)
            models[result['hotel_name']] = result
    
    print(f"\n✅ Đã load {len(models)} models")
    
    return models, metadata


def list_available_hotels(models):
    """Hiển thị danh sách hotels có model"""
    print("\n" + "=" * 70)
    print("DANH SÁCH HOTELS CÓ MODEL")
    print("=" * 70)
    
    # Sort by MAE
    sorted_models = sorted(models.items(), key=lambda x: x[1]['test_mae'])
    
    print(f"\n{'#':>3s} {'Hotel':50s} {'MAE':>10s} {'MAPE':>8s} {'Test Size':>10s}")
    print("-" * 85)
    
    for i, (hotel_name, result) in enumerate(sorted_models, 1):
        print(f"{i:>3d} {hotel_name[:50]:50s} {result['test_mae']:>10.0f} {result['test_mape']:>7.1f}% {result['test_size']:>10d}")


def create_future_dates(start_date, num_days=30):
    """Tạo danh sách ngày tương lai"""
    dates = []
    for i in range(num_days):
        date = start_date + timedelta(days=i)
        dates.append(date)
    return dates


def create_exog_features(dates):
    """Tạo exogenous features cho các ngày tương lai"""
    df = pd.DataFrame({'date': dates})
    df = df.set_index('date')
    
    # Time features
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Mùa cao điểm
    df['is_tet'] = ((df['month'] >= 1) & (df['month'] <= 2)).astype(int)
    df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
    df['is_holiday_season'] = ((df['month'] == 4) | (df['month'] == 9)).astype(int)
    
    return df[['is_weekend', 'is_tet', 'is_summer', 'is_holiday_season']]


def predict_hotel_price(model_result, start_date, num_days=30):
    """Dự đoán giá cho 1 hotel trong num_days tương lai"""
    
    # Tạo ngày tương lai
    future_dates = create_future_dates(start_date, num_days)
    
    # Tạo exog features
    exog = create_exog_features(future_dates)
    
    # Dự đoán
    try:
        model = model_result['model']
        predictions = model.forecast(steps=num_days, exog=exog)
        
        # Tạo DataFrame
        result_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predictions.values
        })
        
        return result_df
        
    except Exception as e:
        print(f"   ❌ Lỗi khi dự đoán: {str(e)[:60]}")
        return None


def interactive_prediction(models):
    """Chế độ dự đoán tương tác"""
    print("\n" + "=" * 70)
    print("CHẾ ĐỘ DỰ ĐOÁN TƯƠNG TÁC")
    print("=" * 70)
    
    # List hotels
    hotel_names = list(models.keys())
    
    print(f"\nCó {len(hotel_names)} hotels. Nhập số để chọn hoặc tên hotel:")
    
    # Hiển thị 10 hotels đầu
    for i, name in enumerate(hotel_names[:10], 1):
        print(f"  {i}. {name[:60]}")
    
    if len(hotel_names) > 10:
        print(f"  ... và {len(hotel_names) - 10} hotels khác")
    
    # Nhập lựa chọn
    choice = input("\nChọn hotel (số hoặc tên): ").strip()
    
    # Parse choice
    selected_hotel = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(hotel_names):
            selected_hotel = hotel_names[idx]
    else:
        # Tìm hotel theo tên
        for name in hotel_names:
            if choice.lower() in name.lower():
                selected_hotel = name
                break
    
    if not selected_hotel:
        print("❌ Không tìm thấy hotel!")
        return
    
    print(f"\n✅ Đã chọn: {selected_hotel}")
    
    # Nhập số ngày
    num_days_str = input("\nSố ngày dự đoán (mặc định 30): ").strip()
    num_days = int(num_days_str) if num_days_str else 30
    
    # Nhập ngày bắt đầu
    start_date_str = input("Ngày bắt đầu (YYYY-MM-DD, mặc định ngày mai): ").strip()
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except:
            print("❌ Định dạng ngày không hợp lệ, dùng ngày mai")
            start_date = datetime.now() + timedelta(days=1)
    else:
        start_date = datetime.now() + timedelta(days=1)
    
    # Dự đoán
    print(f"\n🔮 Đang dự đoán giá {num_days} ngày từ {start_date.date()}...")
    
    model_result = models[selected_hotel]
    predictions = predict_hotel_price(model_result, start_date, num_days)
    
    if predictions is None:
        return
    
    # Hiển thị kết quả
    display_predictions(selected_hotel, predictions, model_result)
    
    # Vẽ biểu đồ
    plot_predictions(selected_hotel, predictions, model_result)


def display_predictions(hotel_name, predictions, model_result):
    """Hiển thị kết quả dự đoán"""
    print("\n" + "=" * 70)
    print(f"DỰ ĐOÁN GIÁ: {hotel_name[:60]}")
    print("=" * 70)
    
    print(f"\n📊 Thông tin model:")
    print(f"   Test MAE: {model_result['test_mae']:,.0f} VND")
    print(f"   Test MAPE: {model_result['test_mape']:.1f}%")
    print(f"   Train size: {model_result['train_size']} days")
    
    print(f"\n📅 Dự đoán giá cho {len(predictions)} ngày:\n")
    
    # Thống kê tổng quan
    print(f"   Giá trung bình: {predictions['predicted_price'].mean():>10,.0f} VND")
    print(f"   Giá thấp nhất: {predictions['predicted_price'].min():>10,.0f} VND (ngày {predictions.loc[predictions['predicted_price'].idxmin(), 'date'].strftime('%Y-%m-%d')})")
    print(f"   Giá cao nhất:  {predictions['predicted_price'].max():>10,.0f} VND (ngày {predictions.loc[predictions['predicted_price'].idxmax(), 'date'].strftime('%Y-%m-%d')})")
    
    print(f"\n📋 Chi tiết:")
    print(f"\n   {'Ngày':12s} {'Thứ':10s} {'Giá dự đoán':>15s} {'Ghi chú':20s}")
    print("   " + "-" * 65)
    
    for _, row in predictions.head(min(15, len(predictions))).iterrows():
        date = row['date']
        price = row['predicted_price']
        
        day_name = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN'][date.dayofweek]
        
        # Ghi chú
        notes = []
        if date.dayofweek >= 5:
            notes.append('Cuối tuần')
        if date.month in [1, 2]:
            notes.append('Tết')
        elif date.month in [6, 7, 8]:
            notes.append('Hè')
        
        note_str = ', '.join(notes) if notes else ''
        
        print(f"   {date.strftime('%Y-%m-%d'):12s} {day_name:10s} {price:>15,.0f} {note_str:20s}")
    
    if len(predictions) > 15:
        print(f"   ... và {len(predictions) - 15} ngày khác")

    # Lưu ra file
    output_dir = Path("arimax_predictions")
    output_dir.mkdir(exist_ok=True)
    
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in hotel_name)[:50]
    csv_file = output_dir / f"{safe_name}_predictions.csv"
    
    predictions.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Đã lưu dự đoán vào: {csv_file}")


def plot_predictions(hotel_name, predictions, model_result):
    """Vẽ biểu đồ dự đoán"""
    plt.figure(figsize=(14, 6))
    
    # Plot predictions
    plt.plot(predictions['date'], predictions['predicted_price'], 
             linewidth=2, marker='o', markersize=4, label='Predicted Price')
    
    # Highlight weekends
    for _, row in predictions.iterrows():
        if row['date'].dayofweek >= 5:  # Weekend
            plt.axvspan(row['date'], row['date'] + timedelta(days=1), 
                       alpha=0.1, color='orange')
    
    # Mean line
    mean_price = predictions['predicted_price'].mean()
    plt.axhline(y=mean_price, color='red', linestyle='--', 
                alpha=0.5, label=f'Mean: {mean_price:,.0f} VND')
    
    plt.title(f'{hotel_name[:60]}\nTest MAE: {model_result["test_mae"]:.0f} VND | MAPE: {model_result["test_mape"]:.1f}%')
    plt.xlabel('Check-in Date')
    plt.ylabel('Predicted Price (VND)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save
    output_dir = Path("arimax_predictions")
    output_dir.mkdir(exist_ok=True)
    
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in hotel_name)[:50]
    img_file = output_dir / f"{safe_name}_chart.png"
    
    plt.savefig(img_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Đã lưu biểu đồ vào: {img_file}")


def batch_prediction(models, start_date, num_days=30, top_n=10):
    """Dự đoán cho nhiều hotels cùng lúc"""
    print("\n" + "=" * 70)
    print("CHẾ ĐỘ DỰ ĐOÁN HÀNG LOẠT")
    print("=" * 70)
    
    # Chọn top N hotels tốt nhất
    sorted_models = sorted(models.items(), key=lambda x: x[1]['test_mae'])
    top_models = sorted_models[:top_n]
    
    print(f"\n🔮 Dự đoán cho {len(top_models)} hotels tốt nhất...")
    print(f"   Từ ngày: {start_date.date()}")
    print(f"   Số ngày: {num_days}")
    
    results = []
    
    for i, (hotel_name, model_result) in enumerate(top_models, 1):
        print(f"\n[{i}/{len(top_models)}] {hotel_name[:50]}...")
        
        predictions = predict_hotel_price(model_result, start_date, num_days)
        
        if predictions is not None:
            avg_price = predictions['predicted_price'].mean()
            min_price = predictions['predicted_price'].min()
            max_price = predictions['predicted_price'].max()
            
            results.append({
                'Hotel': hotel_name,
                'Avg_Price': avg_price,
                'Min_Price': min_price,
                'Max_Price': max_price,
                'Test_MAE': model_result['test_mae']
            })
            
            print(f"   ✅ Avg: {avg_price:>10,.0f} | Min: {min_price:>10,.0f} | Max: {max_price:>10,.0f}")
    
    # Tổng hợp
    print("\n" + "=" * 70)
    print("TỔNG HỢP DỰ ĐOÁN")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    
    print(f"\n{'Hotel':50s} {'Avg Price':>12s} {'Min Price':>12s} {'Max Price':>12s}")
    print("-" * 90)
    
    for _, row in df_results.iterrows():
        print(f"{row['Hotel'][:50]:50s} {row['Avg_Price']:>12,.0f} {row['Min_Price']:>12,.0f} {row['Max_Price']:>12,.0f}")

    # Lưu
    output_file = Path("arimax_predictions") / "batch_predictions_summary.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Đã lưu tổng hợp vào: {output_file}")


def main():
    """Hàm chính"""
    print("=" * 70)
    print("ARIMAX PREDICTION - DỰ ĐOÁN GIÁ TƯƠNG LAI")
    print("=" * 70)
    
    try:
        # Load models
        models, metadata = load_models()
        if models is None:
            return
        
        # List hotels
        list_available_hotels(models)
        
        # Menu
        while True:
            print("\n" + "=" * 70)
            print("MENU")
            print("=" * 70)
            print("1. Dự đoán giá cho 1 hotel (interactive)")
            print("2. Dự đoán cho top 10 hotels tốt nhất")
            print("3. Xem lại danh sách hotels")
            print("4. Thoát")
            
            choice = input("\nChọn (1-4): ").strip()
            
            if choice == '1':
                interactive_prediction(models)
            elif choice == '2':
                start_date = datetime.now() + timedelta(days=1)
                batch_prediction(models, start_date, num_days=30, top_n=10)
            elif choice == '3':
                list_available_hotels(models)
            elif choice == '4':
                print("\n👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng chương trình!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



