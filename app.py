import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import time
from scipy import stats

# Import các thuật toán từ module analytics
try:
    from analytics import KalmanFilter, CUSUMDetector
except ImportError:
    # Dự phòng fallback nếu chưa có file analytics.py cục bộ
    class KalmanFilter:
        def __init__(self, q=0.01, r=1.0):
            self.q = q
            self.r = r
            self.x = None
            self.p = 1.0

        def update(self, z):
            if self.x is None:
                self.x = z
                return z, 0.0
            # Prediction
            self.p = self.p + self.q
            # Innovation Residual
            y = z - self.x
            # Update
            k = self.p / (self.p + self.r)
            self.x = self.x + k * y
            self.p = (1 - k) * self.p
            return self.x, y

    class CUSUMDetector:
        def __init__(self, threshold=5.0, drift=0.5):
            self.threshold = threshold
            self.drift = drift
            self.s_pos = 0.0
            self.s_neg = 0.0

        def update(self, val):
            self.s_pos = max(0.0, self.s_pos + val - self.drift)
            self.s_neg = min(0.0, self.s_neg + val + self.drift)
            if self.s_pos > self.threshold or abs(self.s_neg) > self.threshold:
                self.s_pos = 0.0
                self.s_neg = 0.0
                return True
            return False

# ==========================================
# CẤU HÌNH TRANG & GIAO DIỆN CHUNG
# ==========================================
st.set_page_config(
    page_title="Adaptive Stream Analytics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS giao diện chuẩn dashboard nghiên cứu
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #38bdf8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #0f172a;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 18px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR: BỘ ĐIỀU KHIỂN HỆ THỐNG
# ==========================================
st.sidebar.title("⚙️ System Control")

asset = st.sidebar.selectbox(
    "Select Asset",
    ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
)

refresh_rate = st.sidebar.slider(
    "Refresh Interval (seconds)",
    min_value=1,
    max_value=10,
    value=2
)

window_size = st.sidebar.slider(
    "Sliding Window Size",
    min_value=20,
    max_value=200,
    value=60
)

st.sidebar.divider()
st.sidebar.subheader("Model Hyperparameters")

kalman_q = st.sidebar.number_input(
    "Process Noise (Q)",
    value=0.01,
    format="%.4f"
)

kalman_r = st.sidebar.number_input(
    "Measurement Noise (R)",
    value=1.0,
    format="%.2f"
)

cusum_thresh = st.sidebar.slider(
    "CUSUM Threshold",
    1.0, 50.0, 10.0
)

st.sidebar.divider()
st.sidebar.subheader("Data Layers Visibility")
show_raw = st.sidebar.checkbox("Show Raw Stream", value=True)
show_kalman = st.sidebar.checkbox("Show Kalman State", value=True)
show_ma = st.sidebar.checkbox("Show Moving Average", value=True)

st.sidebar.info(
"""
**Pipeline Architecture:**
- **Layer 1:** Public Ticker API
- **Layer 2:** Sliding Ring Buffer
- **Layer 3:** Kalman State Estimator
- **Layer 4:** CUSUM Drift Detector
- **Layer 5:** Statistical Engine
"""
)

# ==========================================
# KHỞI TẠO STATE & ENGINE
# ==========================================
if "current_asset" not in st.session_state or st.session_state.current_asset != asset:
    # Reset state khi chuyển sang token/tài sản khác
    st.session_state.current_asset = asset
    st.session_state.time = []
    st.session_state.raw = []
    st.session_state.filtered = []
    st.session_state.error = []
    st.session_state.events = []
    st.session_state.kalman = KalmanFilter(q=kalman_q, r=kalman_r)
    st.session_state.cusum = CUSUMDetector(threshold=cusum_thresh)
else:
    # Đồng bộ params nếu người dùng đổi trên UI
    st.session_state.kalman.q = kalman_q
    st.session_state.kalman.r = kalman_r
    st.session_state.cusum.threshold = cusum_thresh

# Hàm lấy giá từ Binance API tương ứng asset đang chọn
def get_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        data = requests.get(url, timeout=3).json()
        return float(data["price"])
    except Exception:
        # Fallback tạo dao động giả lập nếu mất kết nối
        last = st.session_state.raw[-1] if len(st.session_state.raw) > 0 else 50000.0
        return last + float(np.random.normal(0, 5))

# ==========================================
# INGESTION & PIPELINE EXECUTION
# ==========================================
price = get_price(asset)
time_now = datetime.datetime.now().strftime("%H:%M:%S")

# Kalman filtering step
filtered, error = st.session_state.kalman.update(price)

# CUSUM step
change = st.session_state.cusum.update(error)

# Cập nhật buffer
st.session_state.time.append(time_now)
st.session_state.raw.append(price)
st.session_state.filtered.append(filtered)
st.session_state.error.append(error)

if change:
    st.session_state.events.append({
        "time": time_now,
        "asset": asset,
        "price": price,
        "residual": error,
        "alert": "Structural Regime Shift"
    })

# Cắt giữ dữ liệu theo Sliding Window Size từ sidebar
while len(st.session_state.time) > window_size:
    st.session_state.time.pop(0)
    st.session_state.raw.pop(0)
    st.session_state.filtered.pop(0)
    st.session_state.error.pop(0)

# Giới hạn event log tối đa 20 bản ghi
if len(st.session_state.events) > 20:
    st.session_state.events.pop(0)

# ==========================================
# GIAO DIỆN CHÍNH (HEADER)
# ==========================================
st.markdown(
f"""
<div style="background:#0f172a; padding:22px; border-radius:12px; margin-bottom: 20px; border: 1px solid #334155;">
    <h2 style="color:white; margin:0; font-family: sans-serif;">
        🚀 Real-Time Adaptive Stream Analytics
    </h2>
    <p style="color:#94a3b8; margin: 6px 0 0 0;">
        Asset Target: <b style="color: #38bdf8;">{asset}</b> | Online Kalman State Estimation & CUSUM Change Point Engine
    </p>
</div>
""",
unsafe_allow_html=True
)

# TẠO CÁC TABS NGHIÊN CỨU
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Monitoring",
    "📊 Statistical Analysis",
    "🚨 Anomaly Detection",
    "🗂 Data Explorer"
])

# =========================================================
# TAB 1: LIVE MONITORING
# =========================================================
with tab1:
    # 4 KPI Cards phía trên
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${price:,.2f}")
    c2.metric("Filtered State (Kalman)", f"${filtered:,.2f}")
    c3.metric("Innovation Residual", f"{error:,.4f}")
    if change:
        c4.error("⚠ REGIME SHIFT")
    else:
        c4.success("✔ STABLE REGIME")

    # Layout: Biểu đồ chính (2/3) + Đồng hồ Gauge nhiễu (1/3)
    col_chart, col_gauge = st.columns([2.2, 0.8])

    with col_chart:
        fig_main = go.Figure()
        if show_raw:
            fig_main.add_trace(go.Scatter(
                x=st.session_state.time,
                y=st.session_state.raw,
                mode="lines+markers",
                name="Raw Price (Observation)",
                line=dict(color="#94a3b8", width=1.5)
            ))
        if show_kalman:
            fig_main.add_trace(go.Scatter(
                x=st.session_state.time,
                y=st.session_state.filtered,
                mode="lines",
                name="Kalman Estimate (Latent State)",
                line=dict(color="#38bdf8", width=2.5)
            ))

        fig_main.update_layout(
            title="Online State Estimation & Noise Reduction",
            template="plotly_dark",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_main, use_container_width=True)

    with col_gauge:
        # Biểu đồ Gauge đo lường độ phân kỳ (Noise Residual Level)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abs(error),
            title={'text': "Residual Magnitude (|y_t|)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, max(5.0, abs(error) * 2)]},
                'bar': {'color': "#f43f5e"},
                'steps': [
                    {'range': [0, 2], 'color': "#1e293b"},
                    {'range': [2, 5], 'color': "#334155"}
                ],
                'threshold': {
                    'line': {'color': "yellow", 'width': 3},
                    'thickness': 0.75,
                    'value': cusum_thresh
                }
            }
        ))
        gauge.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge, use_container_width=True)

    # Biểu đồ Innovation Residual chạy độc lập
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=st.session_state.time,
        y=st.session_state.error,
        mode="lines",
        line=dict(color="#f59e0b", dash="dot"),
        name="Innovation Residual"
    ))
    fig_res.update_layout(
        title="Kalman Filter Residual Sequence (White Noise Check)",
        template="plotly_dark",
        height=220,
        margin=dict(l=20, r=20, t=35, b=20)
    )
    st.plotly_chart(fig_res, use_container_width=True)

# =========================================================
# TAB 2: STATISTICAL ANALYSIS
# =========================================================
with tab2:
    st.subheader("Statistical Properties & Dynamic Volatility")
    
    if len(st.session_state.raw) > 5:
        s_price = pd.Series(st.session_state.raw)
        returns = s_price.pct_change().dropna()
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            # Dynamic Rolling Return
            fig_return = go.Figure()
            fig_return.add_trace(go.Scatter(
                x=st.session_state.time[1:],
                y=returns,
                mode="lines+markers",
                name="Returns",
                line=dict(color="#10b981")
            ))
            fig_return.update_layout(
                title="Log-Linear Returns Dynamic",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig_return, use_container_width=True)

        with col_s2:
            # Distribution Plot (Histogram + Normal curve fit qua Scipy)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=returns,
                nbinsx=15,
                histnorm='probability density',
                name="Return Dist",
                marker_color="#6366f1"
            ))
            
            # Khớp phân phối chuẩn (Gaussian Fit)
            if len(returns) > 3 and returns.std() > 0:
                x_axis = np.linspace(returns.min(), returns.max(), 50)
                y_axis = stats.norm.pdf(x_axis, returns.mean(), returns.std())
                fig_hist.add_trace(go.Scatter(
                    x=x_axis,
                    y=y_axis,
                    mode='lines',
                    line=dict(color='yellow', width=2),
                    name='Fitted Gaussian'
                ))

            fig_hist.update_layout(
                title="Innovation / Return Empirical Distribution",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Moving Average Comparison
        ma_window = 10
        ma = s_price.rolling(ma_window).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=st.session_state.time, y=st.session_state.raw, name="Raw Price", opacity=0.6))
        fig_ma.add_trace(go.Scatter(x=st.session_state.time, y=ma, name=f"SMA ({ma_window})", line=dict(color="#ec4899", width=2)))
        fig_ma.update_layout(
            title=f"Trend Following: Raw Price vs Rolling SMA({ma_window})",
            template="plotly_dark",
            height=280
        )
        st.plotly_chart(fig_ma, use_container_width=True)

    else:
        st.info("Đang tích luỹ mẫu dữ liệu stream để tính toán thống kê (yêu cầu tối thiểu 5 ticks)...")

# =========================================================
# TAB 3: ANOMALY DETECTION
# =========================================================
with tab3:
    st.subheader("CUSUM Change Point Engine")
    
    c_status, c_exp = st.columns([1, 2])
    with c_status:
        if change:
            st.error("🚨 **CUSUM Triggered**: Structural change detected in residual variance!")
        else:
            st.success("✅ **Stationary**: Process stays within stability confidence band.")

    with c_exp:
        st.markdown("""
        **Online Change-Point Detection Theory:**
        - CUSUM (Cumulative Sum Control Chart) tích lũy độ lệch giữa quan sát và trạng thái lọc Kalman:
        $$S_t^+ = \max(0, S_{t-1}^+ + y_t - v), \quad S_t^- = \min(0, S_{t-1}^- + y_t + v)$$
        - Báo động được kích hoạt khi $S_t > h$, nhận diện sự thay đổi đột ngột của chế độ thị trường (regime shift).
        """)

    st.markdown("### 📋 Anomaly Event Log")
    if len(st.session_state.events) > 0:
        event_df = pd.DataFrame(st.session_state.events)
        st.dataframe(event_df.iloc[::-1], use_container_width=True)
    else:
        st.write("Chưa ghi nhận sự kiện bất thường nào trong window hiện tại.")

# =========================================================
# TAB 4: DATA EXPLORER
# =========================================================
with tab4:
    st.subheader("In-Memory Sliding Data Stream")
    
    df = pd.DataFrame({
        "Timestamp": st.session_state.time,
        "Raw Price": st.session_state.raw,
        "Kalman Filtered": st.session_state.filtered,
        "Residual Error": st.session_state.error
    })

    st.dataframe(df.iloc[::-1], use_container_width=True)

    c_dl, _ = st.columns([1, 4])
    with c_dl:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Stream CSV",
            data=csv,
            file_name=f"{asset}_stream_{datetime.datetime.now().strftime('%H%M%S')}.csv",
            mime="text/csv"
        )

# ==========================================
# VÒNG LẶP STREAMING CHU KỲ TIẾP THEO
# ==========================================
time.sleep(refresh_rate)
st.rerun()