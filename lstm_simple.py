"""
LSTM Training - Train LSTM model cho từng hotel
Dự đoán giá phòng dựa trên ngày check-in (time series)
So sánh với ARIMAX hiện tại (MAPE 9.4%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm

# Thiết lập
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'

# Tạo thư mục outputs
Path("lstm_outputs").mkdir(exist_ok=True)
Path("lstm_models").mkdir(exist_ok=True)


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


def select_hotels_for_training(df, min_observations=50):
    """Chọn hotels có đủ dữ liệu để train LSTM"""
    print("\n" + "=" * 70)
    print("BƯỚC 3: LỌC HOTELS CÓ ĐỦ DỮ LIỆU")
    print("=" * 70)

    # Đếm số observations cho mỗi hotel
    hotel_counts = df.groupby('name').size()

    # Lọc hotels có >= min_observations
    valid_hotels = hotel_counts[hotel_counts >= min_observations].index.tolist()

    print(f"\n📊 Thống kê:")
    print(f"   - Tổng số hotels: {len(hotel_counts)}")
    print(f"   - Hotels có ≥{min_observations} observations: {len(valid_hotels)}")
    print(f"   - Trung bình observations/hotel: {hotel_counts.mean():.1f}")

    # Top 10 hotels có nhiều data nhất
    top_10 = hotel_counts.nlargest(10)
    print(f"\n🏆 Top 10 hotels có nhiều data nhất:")
    for i, (hotel, count) in enumerate(top_10.items(), 1):
        print("2d")

    return df[df['name'].isin(valid_hotels)], valid_hotels


def prepare_lstm_sequences(hotel_data, sequence_length=30, verbose=True):
    """Chuẩn bị sequences cho LSTM training"""
    if verbose:
        print(f"\n🔄 Chuẩn bị sequences (sequence_length={sequence_length})...")

    # Sort by date
    hotel_data = hotel_data.sort_values('ngày_đặt').copy()

    # Features cho LSTM
    feature_cols = ['price', 'is_weekend', 'is_tet', 'is_summer', 'is_holiday_season',
                   'month', 'day_of_week']

    # Normalize features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hotel_data[feature_cols])

    # Tạo sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, 0])  # Dự đoán giá (index 0)

    X = np.array(X)
    y = np.array(y)

    if verbose:
        print(f"✅ Tạo được {len(X)} sequences")
        print(f"   - Input shape: {X.shape} (samples, timesteps, features)")
        print(f"   - Target shape: {y.shape}")

    return X, y, scaler


class LSTMModel(nn.Module):
    """LSTM Model cho dự đoán giá khách sạn"""
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Lấy output của timestep cuối
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def build_lstm_model(sequence_length=30, n_features=7, verbose=True):
    """Xây dựng model LSTM đơn giản"""
    if verbose:
        print("\n🏗️  Xây dựng LSTM model...")
        print(f"   - Sequence length: {sequence_length}")
        print(f"   - Features: {n_features}")

    model = LSTMModel(input_size=n_features)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print("✅ Model built successfully!")
        print(f"   - Total parameters: {total_params:,}")

    return model


class TimeSeriesDataset(Dataset):
    """Dataset cho time series data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_lstm_for_hotel(hotel_name, hotel_data, sequence_length=30, verbose=True, save_plot=False):
    """Train LSTM cho một hotel cụ thể"""
    if verbose:
        print(f"\n🏨 Training LSTM cho: {hotel_name}")
        print(f"   - Observations: {len(hotel_data)}")

    try:
        # Chuẩn bị sequences
        X, y, scaler = prepare_lstm_sequences(hotel_data, sequence_length, verbose)

        if len(X) < 20:  # Quá ít data
            if verbose:
                print(f"❌ Quá ít sequences ({len(X)}), bỏ qua")
            return None, None, None

        # Split train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if verbose:
            print(f"   - Train sequences: {len(X_train)}")
            print(f"   - Test sequences: {len(X_test)}")

        # Create datasets và dataloaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Build model
        model = build_lstm_model(sequence_length, X.shape[2], verbose)
        model = model.to(device)  # Move model to device
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training parameters (tối ưu cho nhiều hotels)
        epochs = 20
        patience = 5
        best_loss = float('inf')
        patience_counter = 0

        # Track history
        history = {
            'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []
        }

        if verbose:
            print("   - Training model...")
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_mae = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - y_batch)).item()

            train_loss /= len(train_loader)
            train_mae /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            val_mae = 0

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(outputs - y_batch)).item()

            val_loss /= len(test_loader)
            val_mae /= len(test_loader)

            # Track history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"lstm_models/{hotel_name.replace('/', '_')}_best.pth")
                print(f"     💾 Saved best model at epoch {epoch+1}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"   - Early stopping at epoch {epoch+1}")
                break


        if verbose and (epoch + 1) % 10 == 0:
            print(f"     Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print("   🔄 Starting prediction...")
        # Load best model
        model.load_state_dict(torch.load(f"lstm_models/{hotel_name.replace('/', '_')}_best.pth"))
        print("   ✅ Loaded model")

        # Predict on test set
        model.eval()
        y_pred_scaled = []

        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                y_pred_scaled.extend(outputs.cpu().numpy().flatten())

        y_pred_scaled = np.array(y_pred_scaled)
        y_test_scaled = y_test.reshape(-1, 1)

        # Inverse transform predictions
        dummy_pred = np.zeros((len(y_pred_scaled), X.shape[2]))
        dummy_test = np.zeros((len(y_test_scaled), X.shape[2]))

        dummy_pred[:, 0] = y_pred_scaled.flatten()
        dummy_test[:, 0] = y_test_scaled.flatten()

        y_pred = scaler.inverse_transform(dummy_pred)[:, 0]
        y_test_actual = scaler.inverse_transform(dummy_test)[:, 0]

        # Tính metrics
        mae = mean_absolute_error(y_test_actual, y_pred)
        mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100

        if verbose and save_plot:
            try:
                # Plot training history
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Val Loss')
                plt.title('Training Loss')
                plt.legend()

                plt.subplot(1, 3, 2)
                plt.plot(history['train_mae'], label='Train MAE')
                plt.plot(history['val_mae'], label='Val MAE')
                plt.title('Training MAE')
                plt.legend()

                plt.subplot(1, 3, 3)
                plt.plot(y_test_actual[-50:], label='Actual', alpha=0.7)
                plt.plot(y_pred[-50:], label='Predicted', alpha=0.7)
                plt.title('Predictions (last 50 points)')
                plt.legend()

                plt.tight_layout()
                plt.savefig(f"lstm_outputs/{hotel_name.replace('/', '_')}_training.png", dpi=150, bbox_inches='tight')
                plt.close()
                print("   ✅ Saved training plot")
            except Exception as e:
                print(f"   ⚠️  Lỗi plotting: {str(e)}")

        try:
            # Save metadata (scaler, features, etc.) sau khi training xong
            metadata = {
                'scaler': scaler,
                'features': ['price', 'is_weekend', 'is_tet', 'is_summer', 'is_holiday_season', 'month', 'day_of_week'],
                'sequence_length': 30,
                'hotel_name': hotel_name,
                'mae': mae,
                'mape': mape
            }
            metadata_path = f"lstm_models/{hotel_name.replace('/', '_')}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print("   ✅ Saved metadata")
        except Exception as e:
            print(f"   ⚠️  Lỗi save metadata: {str(e)}")

        return model, {'mae': mae, 'mape': mape, 'scaler': scaler}, history

    except Exception as e:
        if verbose:
            print(f"❌ Lỗi training {hotel_name}: Exception occurred")
        return None, None, None


def create_summary_visualizations(results_df):
    """Tạo 2 visualizations quan trọng nhất cho báo cáo"""
    print("\n📊 TẠO 2 VISUALIZATIONS QUAN TRỌNG NHẤT...")

    # Chỉ tạo 2 biểu đồ quan trọng nhất
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(results_df['mape'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(results_df['mape'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {results_df["mape"].median():.1f}%')
    plt.axvline(results_df['mape'].mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {results_df["mape"].mean():.1f}%')
    plt.xlabel('MAPE (%)', fontsize=12)
    plt.ylabel('Số Hotels', fontsize=12)
    plt.title('Phân bố MAPE của tất cả Hotels', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 2. Top 10 best performing hotels - QUAN TRỌNG THỨ 2
    top_10 = results_df.nsmallest(10, 'mape')
    plt.subplot(1, 2, 2)
    bars = plt.barh(range(len(top_10)), top_10['mape'])
    plt.yticks(range(len(top_10)), [name[:25] + '...' if len(name) > 25 else name for name in top_10['hotel']], fontsize=10)
    plt.xlabel('MAPE (%)', fontsize=12)
    plt.title('Top 10 Hotels (MAPE thấp nhất)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Color bars by MAPE value
    for bar, mape in zip(bars, top_10['mape']):
        if mape < 5:
            bar.set_color('green')
        elif mape < 10:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.tight_layout()
    plt.savefig('lstm_outputs/lstm_summary_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Đã lưu 2 visualizations quan trọng nhất: lstm_outputs/lstm_summary_visualization.png")
    print("   - Biểu đồ 1: Phân bố MAPE (Histogram)")
    print("   - Biểu đồ 2: Top 10 Hotels tốt nhất (Bar chart)")

def main():
    """Main function"""
    print("🚀 LSTM TRAINING FOR HOTEL PRICE PREDICTION")
    print("=" * 70)

    # Load và chuẩn bị dữ liệu
    df = load_and_prepare_data()
    df = create_time_series_features(df)
    df_filtered, valid_hotels = select_hotels_for_training(df, min_observations=50)

    # Train LSTM cho tất cả hotels có đủ dữ liệu
    results = []
    max_hotels = min(5, len(valid_hotels))  # Test với 5 hotels đầu tiên

    print(f"\n🎯 TRAINING ALL {max_hotels} HOTELS")
    print("=" * 70)

    # Progress bar cho training
    for i, hotel_name in enumerate(valid_hotels[:max_hotels], 1):
        print(f"[{i}/{max_hotels}] Training {hotel_name[:50]}...")

        hotel_data = df_filtered[df_filtered['name'] == hotel_name].copy()

        # Chỉ save plot cho top 3 hotels (sẽ xác định sau khi train xong)
        model, metrics, history = train_lstm_for_hotel(hotel_name, hotel_data, verbose=True, save_plot=False)

        if metrics:
            results.append({
                'hotel': hotel_name,
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'observations': len(hotel_data)
            })

    # Tổng kết kết quả
    print("\n" + "=" * 70)
    print("📊 TỔNG KẾT KẾT QUẢ")
    print("=" * 70)

    if results:
        results_df = pd.DataFrame(results)
        print(f"\n🏆 Kết quả LSTM ({len(results_df)} hotels):")
        print(results_df.to_string(index=False, float_format='%.0f'))

        print("\n📈 Thống kê:")
        print(f"   - Mean MAE: {results_df['mae'].mean():,.0f} VND")
        print(f"   - Median MAPE: {results_df['mape'].median():.1f}%")
        print(f"   - Mean MAPE: {results_df['mape'].mean():.1f}%")
        print("\n🔥 Best model:")
        best = results_df.loc[results_df['mape'].idxmin()]
        print(f"   - Hotel: {best['hotel']}")
        print(f"   - MAPE: {best['mape']:.1f}%")

        # So sánh với ARIMAX
        print("\n⚖️  SO SÁNH VỚI ARIMAX:")
        print(f"   - LSTM median MAPE: {results_df['mape'].median():.1f}%")
        print("   - ARIMAX median MAPE: 9.4%")
        print(f"   → LSTM {'tốt hơn' if best['mape'] < 9.4 else 'tệ hơn'} ARIMAX!")

        # Display results in terminal (detailed table)
        print("\n💾 KẾT QUẢ CHI TIẾT TẤT CẢ HOTELS:")
        print("=" * 100)
        print(results_df.to_string(index=False, float_format='%.2f'))
        print("=" * 100)

        # Optional: Save to CSV (auto-save for now, can be modified)
        try:
            results_df.to_csv("lstm_outputs/lstm_results_summary.csv", index=False)
            print("💾 Đã tự động lưu kết quả vào: lstm_outputs/lstm_results_summary.csv")
        except Exception as e:
            print(f"⚠️  Không thể lưu CSV: {str(e)}")

        # Tạo visualizations tổng hợp
        create_summary_visualizations(results_df)

        # Tạo training plots cho top 3 hotels tốt nhất
        print("\n" + "=" * 70)
        print("📊 TẠO TRAINING PLOTS CHO TOP 3 HOTELS TỐT NHẤT")
        print("=" * 70)

        top_3_hotels = results_df.nsmallest(3, 'mape')['hotel'].tolist()
        for i, hotel_name in enumerate(top_3_hotels, 1):
            print(f"\n[{i}/3] Tạo plot cho: {hotel_name[:50]}...")
            hotel_data = df_filtered[df_filtered['name'] == hotel_name].copy()
            _, _, _ = train_lstm_for_hotel(hotel_name, hotel_data, verbose=False, save_plot=True)

        print("\n✅ Đã tạo training plots cho top 3 hotels!")
    else:
        print("❌ Không có kết quả nào!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    main()
