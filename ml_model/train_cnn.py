import os
import sys
import ctypes
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Windows 환경에서 DLL 로드 실패 시 시스템 에러 다이얼로그 억제
if sys.platform == "win32":
    try:
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        SEM_NOOPENFILEERRORBOX = 0x8000
        ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)
    except Exception:
        pass

# 상위 디렉토리 모듈 참조
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import get_daily_prices, save_prediction, STOCK_INFO, init_db

# PyTorch 지원 여부 체크
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _t = torch.zeros(1)
    TORCH_AVAILABLE = True
except BaseException:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class StockCNN1D(nn.Module):
        def __init__(self, input_len: int = 30, output_dim: int = 3):
            super(StockCNN1D, self).__init__()
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(4)
            )
            self.regressor = nn.Sequential(
                nn.Linear(128 * 4, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, output_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.feature_extractor(x)
            feat = feat.view(feat.size(0), -1)
            return self.regressor(feat)

    class TimeSeriesStockDataset(Dataset):
        def __init__(self, X: np.ndarray, Y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            self.Y = torch.tensor(Y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

# 고속 벡터화 1D-CNN 다중 회귀 모델 (Pure NumPy 기반)
class Vectorized1DCNN:
    def __init__(self, input_len: int = 30, num_filters: int = 16, kernel_size: int = 3, hidden_dim: int = 32, output_dim: int = 3):
        self.input_len = input_len
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim

        # 가중치 He / Xavier 초기화
        self.W_conv = np.random.randn(num_filters, kernel_size) * np.sqrt(2.0 / kernel_size)
        self.b_conv = np.zeros((num_filters, 1))

        self.conv_out_len = (input_len - kernel_size + 1) // 2
        self.flatten_dim = num_filters * self.conv_out_len

        self.W1 = np.random.randn(self.flatten_dim, hidden_dim) * np.sqrt(2.0 / self.flatten_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def _extract_windows(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 30) -> shape (N, 28, 3)"""
        N = X.shape[0]
        windows = np.lib.stride_tricks.sliding_window_view(X, self.kernel_size, axis=1) # (N, 28, 3)
        return windows

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        N = X.shape[0]
        windows = self._extract_windows(X) # (N, 28, 3)
        conv = np.einsum('nwt,ft->nfw', windows, self.W_conv) + self.b_conv # (N, F, 28)
        relu = np.maximum(0, conv)

        # MaxPool1D (size=2, stride=2)
        relu_reshaped = relu.reshape(N, self.num_filters, self.conv_out_len, 2)
        pool_out = np.max(relu_reshaped, axis=3) # (N, F, 14)
        pool_argmax = np.argmax(relu_reshaped, axis=3) # (N, F, 14)

        flatten = pool_out.reshape(N, -1)
        z1 = np.dot(flatten, self.W1) + self.b1
        h1 = np.maximum(0, z1)
        out = np.dot(h1, self.W2) + self.b2

        cache = {
            "X": X, "windows": windows, "conv": conv, "relu": relu,
            "pool_out": pool_out, "pool_argmax": pool_argmax,
            "flatten": flatten, "z1": z1, "h1": h1
        }
        return out, cache

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 120, lr: float = 0.008):
        N = X.shape[0]
        mW_conv, vW_conv = np.zeros_like(self.W_conv), np.zeros_like(self.W_conv)
        mb_conv, vb_conv = np.zeros_like(self.b_conv), np.zeros_like(self.b_conv)
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, epochs + 1):
            pred, cache = self.forward(X)
            loss = np.mean((pred - Y) ** 2)

            # Gradients
            dpred = 2.0 * (pred - Y) / (N * self.output_dim)
            dW2 = np.dot(cache["h1"].T, dpred)
            db2 = np.sum(dpred, axis=0)

            dh1 = np.dot(dpred, self.W2.T)
            dz1 = dh1 * (cache["z1"] > 0)
            dW1 = np.dot(cache["flatten"].T, dz1)
            db1 = np.sum(dz1, axis=0)

            dflatten = np.dot(dz1, self.W1.T)
            dpool_out = dflatten.reshape(N, self.num_filters, self.conv_out_len)

            # Backprop pool
            drelu_reshaped = np.zeros((N, self.num_filters, self.conv_out_len, 2))
            np.put_along_axis(drelu_reshaped, cache["pool_argmax"][:, :, :, None], dpool_out[:, :, :, None], axis=3)
            drelu = drelu_reshaped.reshape(N, self.num_filters, -1)
            dconv = drelu * (cache["conv"] > 0)

            db_conv = np.sum(dconv, axis=(0, 2), keepdims=True)
            dW_conv = np.einsum('nfw,nwt->ft', dconv, cache["windows"])

            # Adam optimizer step
            t = epoch
            for param, dparam, m, v in [
                (self.W2, dW2, mW2, vW2), (self.b2, db2, mb2, vb2),
                (self.W1, dW1, mW1, vW1), (self.b1, db1, mb1, vb1),
                (self.W_conv, dW_conv, mW_conv, vW_conv), (self.b_conv, db_conv, mb_conv, vb_conv)
            ]:
                m[:] = beta1 * m + (1 - beta1) * dparam
                v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if epoch % 40 == 0 or epoch == epochs:
                print(f"    Epoch [{epoch:03d}/{epochs:03d}] - Loss (MSE): {loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        out, _ = self.forward(X)
        return out

def prepare_sliding_windows(prices: np.ndarray, window_size: int = 30) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    종가 배열을 기반으로 슬라이딩 윈도우 X (30일) 및 타겟 Y (1d, 5d, 20d)를 생성합니다.
    시계열 정상성(Stationarity)을 확보하기 위해 각 윈도우의 기준 종가 대비 상대 등락률(Relative Return)로 정규화합니다.
    """
    X, Y = [], []
    total_len = len(prices)
    max_lookahead = 20  # 20거래일 뒤 (1달)

    for i in range(total_len - window_size - max_lookahead + 1):
        w = prices[i : i + window_size]
        base_p = float(w[-1])
        if base_p <= 0:
            continue
        
        # 30일 윈도우 정규화: 윈도우 기준 종가 대비 상대 등락률
        x_norm = (w - base_p) / base_p
        
        # 미래 타겟 등락률 (1일 뒤, 5일 뒤, 20일 뒤)
        y_1d = (prices[i + window_size] - base_p) / base_p
        y_1w = (prices[i + window_size + 4] - base_p) / base_p
        y_1m = (prices[i + window_size + 19] - base_p) / base_p
        
        X.append(x_norm)
        Y.append([y_1d, y_1w, y_1m])

    min_val = float(np.min(prices))
    max_val = float(np.max(prices))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), min_val, max_val

def train_and_predict_stock(stock_code: str, epochs: int = 40) -> Optional[Dict[str, Any]]:
    """
    특정 종목의 데이터를 로드하여 1D-CNN 모델을 학습하고
    최신 30일 종가를 바탕으로 미래 3개 시점(1d, 1w, 1m)을 예측하여 DB에 적재합니다.
    """
    stock_name = STOCK_INFO.get(stock_code, stock_code)

    data = get_daily_prices(stock_code)
    if len(data) < 55:
        return None

    df = pd.DataFrame(data)
    prices = df["close_price"].values.astype(float)
    dates = df["date"].values

    base_date = str(dates[-1])
    current_price = float(prices[-1])

    X, Y, min_val, max_val = prepare_sliding_windows(prices, window_size=30)
    if len(X) == 0:
        return None

    # 1D-CNN 모델 학습 진행 (고속 벡터화 또는 PyTorch)
    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StockCNN1D(input_len=30, output_dim=3).to(device)
        dataset = TimeSeriesStockDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

        model.train()
        for epoch in range(1, epochs + 1):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        latest_30_raw = prices[-30:]
        latest_x = ((latest_30_raw - current_price) / current_price).astype(np.float32)
        input_tensor = torch.tensor(latest_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_returns = model(input_tensor).cpu().numpy()[0]
    else:
        model = Vectorized1DCNN(input_len=30, num_filters=16, kernel_size=3, hidden_dim=32, output_dim=3)
        model.fit(X, Y, epochs=epochs, lr=0.008)

        latest_30_raw = prices[-30:]
        latest_x = ((latest_30_raw - current_price) / current_price).astype(np.float32).reshape(1, 30)
        pred_returns = model.predict(latest_x)[0]

    # 각 예측 헤드별 독립된 등락률 추출
    ret_1d = float(pred_returns[0])
    ret_1w = float(pred_returns[1])
    ret_1m = float(pred_returns[2])

    # 시스템 주 예측 엔진인 predict.py (Huber 손실 다중 직접 타깃 모델) 로직으로 위임
    from predict import predict_stock
    return predict_stock(stock_code, save_csv=True)

def train_all_stocks(epochs: int = 30):
    """시스템 주 예측 엔진(predict.py RandomForest)을 통해 전체 종목의 예측을 실행합니다."""
    from predict import run_predict_pipeline
    return run_predict_pipeline()

if __name__ == "__main__":
    train_all_stocks()

