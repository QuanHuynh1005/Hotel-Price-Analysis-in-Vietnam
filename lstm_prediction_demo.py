"""
LSTM Prediction Demo - Dự đoán giá khách sạn sử dụng models đã train
Demo cách sử dụng LSTM models để dự đoán giá tương lai
"""

import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Import từ lstm_simple.py
from lstm_simple import LSTMModel, prepare_lstm_sequences

# Thiết lập
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_trained_model(hotel_name):
    """Load trained LSTM model và scaler cho một hotel"""
    try:
        # Load metadata trước
        metadata_path = f"lstm_models/{hotel_name.replace('/', '_')}_metadata.pkl"
        if not Path(metadata_path).exists():
            print(f"⚠️  Không tìm thấy metadata cho {hotel_name}")
            return None, None

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        scaler = metadata['scaler']

        # Load model architecture và weights
        model = LSTMModel(input_size=7)  # 7 features
        model_path = f"lstm_models/{hotel_name.replace('/', '_')}_best.pth"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        return model, scaler

    except Exception as e:
        print(f"❌ Lỗi load model {hotel_name}: {str(e)}")
        return None, None


def predict_future_prices(hotel_name, days_ahead=30):
    """Dự đoán giá tương lai cho một hotel"""

    print(f"\n🏨 Dự đoán giá cho: {hotel_name}")
    print(f"   - Ngày dự đoán: {days_ahead} ngày")

    # Load model và scaler
    model, scaler = load_trained_model(hotel_name)
    if model is None or scaler is None:
        return None

    # Load dữ liệu gốc để tạo sequences
    df = pd.read_excel("khach_san.xlsx", sheet_name="giá_khách_sạn_theo_ngày")

    # Clean data
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df['ngày_đặt'] = pd.to_datetime(df['ngày_đặt'], errors='coerce')
    def clean_price(price_str):
        if pd.isna(price_str):
            return np.nan
        s = str(price_str).replace('VND', '').replace('.', '').replace(',', '').replace(' ', '').strip()
        try:
            return float(s)
        except:
            return np.nan
    df['price'] = df['price_on_list'].apply(clean_price)
    df = df[df['price'].notna() & df['ngày_đặt'].notna()].copy()

    # Filter cho hotel này
    hotel_data = df[df['name'] == hotel_name].copy()
    if len(hotel_data) < 30:
        print(f"❌ Quá ít dữ liệu cho {hotel_name}: {len(hotel_data)}")
        return None

    # Tạo time series features
    hotel_data = hotel_data.sort_values('ngày_đặt').copy()
    hotel_data['month'] = hotel_data['ngày_đặt'].dt.month
    hotel_data['day'] = hotel_data['ngày_đặt'].dt.day
    hotel_data['day_of_week'] = hotel_data['ngày_đặt'].dt.dayofweek
    hotel_data['is_weekend'] = (hotel_data['day_of_week'] >= 5).astype(int)
    hotel_data['is_tet'] = ((hotel_data['month'] >= 1) & (hotel_data['month'] <= 2)).astype(int)
    hotel_data['is_summer'] = ((hotel_data['month'] >= 6) & (hotel_data['month'] <= 8)).astype(int)
    hotel_data['is_holiday_season'] = ((hotel_data['month'] == 4) | (hotel_data['month'] == 9)).astype(int)

    # Dự đoán từng ngày một
    predictions = []
    current_data = hotel_data.copy()

    model.eval()
    with torch.no_grad():
        for day in range(days_ahead):
            # Lấy 30 ngày gần nhất để dự đoán
            recent_data = current_data.tail(30).copy()

            # Tạo sequence
            features = ['price', 'is_weekend', 'is_tet', 'is_summer', 'is_holiday_season',
                       'month', 'day_of_week']
            scaled_data = scaler.transform(recent_data[features])

            # Tạo input tensor
            X = torch.FloatTensor(scaled_data).unsqueeze(0)  # (1, 30, 7)

            # Predict
            pred_scaled = model(X)
            pred_price_scaled = pred_scaled.item()

            # Inverse transform để lấy giá thực
            dummy_pred = np.zeros((1, len(features)))
            dummy_pred[0, 0] = pred_price_scaled  # Chỉ thay đổi cột price
            pred_price = scaler.inverse_transform(dummy_pred)[0, 0]

            predictions.append(pred_price)

            # Thêm prediction vào current_data để dự đoán ngày tiếp theo
            next_date = current_data['ngày_đặt'].max() + timedelta(days=1)
            new_row = {
                'name': hotel_name,
                'ngày_đặt': next_date,
                'price': pred_price,
                'month': next_date.month,
                'day': next_date.day,
                'day_of_week': next_date.weekday(),
                'is_weekend': int(next_date.weekday() >= 5),
                'is_tet': int(next_date.month in [1, 2]),
                'is_summer': int(next_date.month in [6, 7, 8]),
                'is_holiday_season': int(next_date.month in [4, 9])
            }
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)

    return predictions


def create_prediction_visualization(hotel_name, predictions, days_ahead=30):
    """Tạo visualization cho predictions"""

    if predictions is None:
        return

    # Load dữ liệu gốc để so sánh
    df = pd.read_excel("khach_san.xlsx", sheet_name="giá_khách_sạn_theo_ngày")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df['ngày_đặt'] = pd.to_datetime(df['ngày_đặt'], errors='coerce')

    def clean_price(price_str):
        if pd.isna(price_str):
            return np.nan
        s = str(price_str).replace('VND', '').replace('.', '').replace(',', '').replace(' ', '').strip()
        try:
            return float(s)
        except:
            return np.nan
    df['price'] = df['price_on_list'].apply(clean_price)

    hotel_data = df[df['name'] == hotel_name].copy()
    hotel_data = hotel_data.sort_values('ngày_đặt')

    # Tạo future dates
    last_date = hotel_data['ngày_đặt'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]

    plt.figure(figsize=(15, 8))

    # Plot 1: Historical + Predictions
    plt.subplot(2, 2, 1)
    plt.plot(hotel_data['ngày_đặt'], hotel_data['price'], 'b-', linewidth=2, label='Historical')
    plt.plot(future_dates, predictions, 'r--', linewidth=2, label='LSTM Predictions')
    plt.xlabel('Ngày', fontsize=10)
    plt.ylabel('Giá (VND)', fontsize=10)
    plt.title(f'{hotel_name} - Historical vs Predictions', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Format y-axis to show prices with thousand separators
    ax1 = plt.gca()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Plot 2: Predictions only (zoom in)
    plt.subplot(2, 2, 2)
    plt.plot(future_dates, predictions, 'r-', linewidth=3, marker='o', markersize=4)
    plt.xlabel('Ngày', fontsize=10)
    plt.ylabel('Giá (VND)', fontsize=10)
    plt.title(f'{hotel_name} - {days_ahead} ngày dự đoán', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Format y-axis to show prices with thousand separators
    ax2 = plt.gca()
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Plot 3: Price distribution
    plt.subplot(2, 2, 3)
    plt.hist(hotel_data['price'], bins=20, alpha=0.7, color='blue', label='Historical')
    plt.axvline(np.mean(predictions), color='red', linestyle='--', linewidth=2,
                label=f'Mean Prediction: {np.mean(predictions):,.0f}')
    plt.xlabel('Giá (VND)', fontsize=10)
    plt.ylabel('Tần suất', fontsize=10)
    plt.title('Phân bố giá', fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # Format x-axis to show prices with thousand separators
    ax3 = plt.gca()
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Plot 4: Prediction statistics
    plt.subplot(2, 2, 4)
    pred_stats = {
        'Min': np.min(predictions),
        'Max': np.max(predictions),
        'Mean': np.mean(predictions),
        'Median': np.median(predictions),
        'Std': np.std(predictions)
    }

    bars = plt.bar(range(len(pred_stats)), list(pred_stats.values()), color='skyblue')
    plt.xticks(range(len(pred_stats)), list(pred_stats.keys()), fontsize=9)
    plt.ylabel('Giá (VND)', fontsize=10)
    plt.title('Thống kê dự đoán', fontsize=11)

    # Add value labels on bars
    for bar, value in zip(bars, pred_stats.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:,.0f}', ha='center', va='bottom', fontsize=9)

    plt.grid(True, alpha=0.3)

    # Format y-axis to show prices with thousand separators
    ax4 = plt.gca()
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    plt.tight_layout()
    plt.savefig(f'lstm_outputs/{hotel_name.replace("/", "_")}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Đã lưu prediction visualization: lstm_outputs/{hotel_name.replace('/', '_')}_predictions.png")


def main():
    """Demo prediction cho một số hotels tốt nhất"""

    print("🚀 LSTM PREDICTION DEMO")
    print("=" * 70)

    # Load results để chọn hotels tốt nhất
    results_df = pd.read_csv('lstm_outputs/lstm_results_summary.csv')
    top_3_hotels = results_df.nsmallest(3, 'mape')['hotel'].tolist()

    print(f"🎯 Demo prediction cho top 3 hotels tốt nhất:")
    for i, hotel in enumerate(top_3_hotels, 1):
        print(f"   {i}. {hotel}")

    print("\n" + "=" * 70)

    # Dự đoán cho từng hotel
    for hotel_name in top_3_hotels:
        predictions = predict_future_prices(hotel_name, days_ahead=30)
        if predictions:
            create_prediction_visualization(hotel_name, predictions, days_ahead=30)

            # In thống kê nhanh
            print("\n📊 Thống kê dự đoán:")
            print(f"   - Giá trung bình: {np.mean(predictions):>10,.0f} VND")
            print(f"   - Giá thấp nhất:  {np.min(predictions):>10,.0f} VND")
            print(f"   - Giá cao nhất:   {np.max(predictions):>10,.0f} VND")
    print("\n" + "=" * 70)
    print("✅ Hoàn thành demo prediction!")
    print("📁 Xem kết quả trong thư mục lstm_outputs/")


if __name__ == "__main__":
    main()
