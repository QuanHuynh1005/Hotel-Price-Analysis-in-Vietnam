"""
LSTM Comparison Visualization
So sánh giá dự đoán vs giá thực tế trên dữ liệu đã có
"""

import sys
import io
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Thiết lập
plt.rcParams['font.family'] = 'DejaVu Sans'

# Import LSTM model từ lstm_simple.py
from lstm_simple import LSTMModel


def create_sequences(data, sequence_length):
    """Tạo sequences từ numpy array"""
    X, y, indices = [], [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 0])  # Price is first column
        indices.append(i + sequence_length)

    return np.array(X), np.array(y), indices


def load_and_prepare_data(file_path="khach_san.xlsx"):
    """Load và chuẩn bị dữ liệu"""
    print("\n📂 LOAD DỮ LIỆU...")
    
    df = pd.read_excel(file_path, sheet_name="giá_khách_sạn_theo_ngày")
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
    df = df.dropna(subset=['price', 'ngày_đặt'])
    df = df.sort_values(['name', 'ngày_đặt'])
    
    print(f"✅ Loaded {len(df)} records, {df['name'].nunique()} hotels")
    return df


def add_features(df):
    """Thêm features giống như khi training"""
    df = df.copy()
    
    # Time features
    df['month'] = df['ngày_đặt'].dt.month
    df['day_of_week'] = df['ngày_đặt'].dt.dayofweek
    
    # Weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Tết (Feb-Mar)
    df['is_tet'] = df['month'].isin([2, 3]).astype(int)
    
    # Summer (Jun-Aug)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    
    # Holiday season (Dec-Jan)
    df['is_holiday_season'] = df['month'].isin([12, 1]).astype(int)
    
    return df


def predict_with_lstm(hotel_name, hotel_data):
    """Dự đoán giá cho hotel sử dụng model đã train"""
    
    # Load metadata
    safe_name = hotel_name.replace('/', '_')
    metadata_path = f"lstm_models/{safe_name}_metadata.pkl"
    model_path = f"lstm_models/{safe_name}_best.pth"
    
    if not Path(metadata_path).exists() or not Path(model_path).exists():
        print(f"   ⚠️  Model không tồn tại cho {hotel_name[:50]}")
        return None
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    scaler = metadata['scaler']
    features = metadata['features']
    sequence_length = metadata['sequence_length']
    
    # Prepare data
    hotel_data = add_features(hotel_data)
    hotel_data = hotel_data.sort_values('ngày_đặt')
    
    # Scale features
    X = hotel_data[features].values
    X_scaled = scaler.transform(X)
    
    # Create sequences
    X_seq, y_seq, indices = create_sequences(X_scaled, sequence_length)
    
    if len(X_seq) == 0:
        print(f"   ⚠️  Không đủ dữ liệu để tạo sequences")
        return None
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=len(features), hidden_size1=64, hidden_size2=32, output_size=1, dropout_rate=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Predict
    X_tensor = torch.FloatTensor(X_seq).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()
    
    # Inverse transform predictions
    y_pred = scaler.inverse_transform(
        np.column_stack([y_pred_scaled] + [np.zeros(len(y_pred_scaled)) for _ in range(len(features)-1)])
    )[:, 0]
    
    # Get actual prices
    y_actual = hotel_data.iloc[indices]['price'].values
    dates = hotel_data.iloc[indices]['ngày_đặt'].values
    
    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
    
    return {
        'dates': dates,
        'actual': y_actual,
        'predicted': y_pred,
        'mae': mae,
        'mape': mape
    }


def create_comparison_plot(hotel_name, result, output_dir="lstm_outputs"):
    """Tạo biểu đồ so sánh actual vs predicted"""
    
    plt.figure(figsize=(14, 6))
    
    dates = result['dates']
    actual = result['actual']
    predicted = result['predicted']
    mae = result['mae']
    mape = result['mape']
    
    # Plot 1: Time series comparison
    plt.subplot(1, 2, 1)
    plt.plot(dates, actual, 'b-', linewidth=2, label='Giá thực tế', alpha=0.7)
    plt.plot(dates, predicted, 'r--', linewidth=2, label='Giá dự đoán', alpha=0.7)
    plt.xlabel('Ngày', fontsize=12)
    plt.ylabel('Giá (VND)', fontsize=12)
    plt.title(f'{hotel_name[:50]}\nMAE: {mae:,.0f} VND | MAPE: {mape:.1f}%', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Plot 2: Scatter plot actual vs predicted
    plt.subplot(1, 2, 2)
    plt.scatter(actual, predicted, alpha=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    plt.xlabel('Giá thực tế (VND)', fontsize=12)
    plt.ylabel('Giá dự đoán (VND)', fontsize=12)
    plt.title('Actual vs Predicted', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Format axes
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    safe_name = hotel_name.replace('/', '_')
    filename = output_path / f"{safe_name}_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filename}")


def main():
    """Main function"""
    print("=" * 70)
    print("LSTM COMPARISON VISUALIZATION")
    print("So sánh giá dự đoán vs giá thực tế")
    print("=" * 70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Load results summary để chọn hotels
    results_df = pd.read_csv('lstm_outputs/lstm_results_summary.csv')
    
    # Chọn top 3 hotels tốt nhất
    top_3_hotels = results_df.nsmallest(3, 'mape')['hotel'].tolist()
    
    print(f"\n🎯 Tạo comparison plots cho top 3 hotels tốt nhất:")
    for i, hotel in enumerate(top_3_hotels, 1):
        print(f"   {i}. {hotel}")
    
    print("\n" + "=" * 70)
    
    # Process each hotel
    comparison_results = []
    
    for i, hotel_name in enumerate(top_3_hotels, 1):
        print(f"\n[{i}/3] Processing: {hotel_name[:50]}...")
        
        hotel_data = df[df['name'] == hotel_name].copy()
        
        if len(hotel_data) < 50:
            print(f"   ⚠️  Không đủ dữ liệu (chỉ có {len(hotel_data)} records)")
            continue
        
        # Predict
        result = predict_with_lstm(hotel_name, hotel_data)
        
        if result is None:
            continue
        
        # Create plot
        create_comparison_plot(hotel_name, result)
        
        # Store results
        comparison_results.append({
            'hotel': hotel_name,
            'mae': result['mae'],
            'mape': result['mape'],
            'num_predictions': len(result['actual'])
        })
        
        print(f"   📊 MAE: {result['mae']:,.0f} VND | MAPE: {result['mape']:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TỔNG KẾT")
    print("=" * 70)
    
    if comparison_results:
        summary_df = pd.DataFrame(comparison_results)
        print("\n" + summary_df.to_string(index=False))
        
        print(f"\n✅ Đã tạo {len(comparison_results)} comparison plots")
        print(f"📁 Xem kết quả trong thư mục: lstm_outputs/")
        print(f"   - Files: *_comparison.png")
    else:
        print("\n❌ Không có kết quả nào!")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH!")
    print("=" * 70)


if __name__ == "__main__":
    main()

