import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import os
import hashlib

# Safety import for scipy
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# 0. Configuration Persistence
# ==========================================
CONFIG_FILE = 'strategy_config.json'

DEFAULT_CODES = ["518880", "588000", "513100", "510180"]

DEFAULT_PARAMS = {
    'lookback': 25,
    'smooth': 3,
    'threshold': 0.005,
    'min_holding': 3,
    'allow_cash': True,
    'mom_method': 'Risk-Adjusted (稳健)', 
    'selected_codes': DEFAULT_CODES
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                config = DEFAULT_PARAMS.copy()
                config.update(saved_config)
                return config
        except Exception:
            return DEFAULT_PARAMS.copy()
    return DEFAULT_PARAMS.copy()

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception:
        pass

# ==========================================
# 1. UI Configuration & CSS
# ==========================================
st.set_page_config(
    page_title="AlphaTarget | 核心资产轮动策略终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .metric-card {
        background-color: #ffffff; border: 1px solid #eaeaea; border-radius: 12px;
        padding: 20px 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center; transition: all 0.3s ease; height: 100%;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 16px rgba(0,0,0,0.08); }
    .metric-label { color: #7f8c8d; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { color: #2c3e50; font-size: 1.6rem; font-weight: 700; }
    .signal-banner {
        padding: 25px; border-radius: 12px; margin-bottom: 25px; color: white;
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
    }
    .total-asset-header { font-size: 2.2rem; font-weight: 800; color: #2c3e50; }
    .opt-highlight { background-color: #e8f4f8; border-left: 4px solid #3498db; padding: 10px; border-radius: 4px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

TRANSACTION_COST = 0.0001
PRESET_ETFS = {
    "518880": "黄金ETF", "588000": "科创50", "513100": "纳指100",
    "510180": "上证180", "159915": "创业板指", "510300": "沪深300"
}

def metric_html(label, value, sub="", color="#2c3e50"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 6px;">{sub}</div>
    </div>
    """

# ==========================================
# 2. Data Layer
# ==========================================
@st.cache_data(ttl=43200)
def get_all_etf_list():
    try:
        df = ak.fund_etf_spot_em()
        df['display'] = df['代码'] + " | " + df['名称']
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=14400)
def download_market_data(codes_list, end_date_str):
    start_str = '20150101'
    price_dict = {}
    name_map = {}
    etf_list = get_all_etf_list()
    
    for code in codes_list:
        name = code
        if code in PRESET_ETFS:
            name = PRESET_ETFS[code]
        elif not etf_list.empty:
            match = etf_list[etf_list['代码'] == code]
            if not match.empty: name = match.iloc[0]['名称']
        name_map[code] = name
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                price_dict[name] = df['收盘'].astype(float)
        except: continue

    if not price_dict: return None, None
    data = pd.concat(price_dict, axis=1).sort_index().ffill().dropna(how='all')
    return data, name_map

# ==========================================
# 3. Strategy Core
# ==========================================
def calculate_momentum(data, lookback, smooth, method):
    if method == 'Risk-Adjusted (稳健)':
        ret = data.pct_change(lookback)
        vol = data.pct_change().rolling(lookback).std()
        mom = ret / (vol + 1e-9)
    elif method == 'MA Distance (趋势)':
        ma = data.rolling(lookback).mean()
        mom = (data / ma) - 1
    else:
        mom = data.pct_change(lookback)
    
    return mom.rolling(smooth).mean() if smooth > 1 else mom

def fast_backtest_vectorized(daily_ret, mom_df, threshold, min_holding=1, cost_rate=0.0001, allow_cash=True):
    signal_mom = mom_df.shift(1)
    n_days, n_assets = daily_ret.shape
    p_ret = daily_ret.values
    p_mom = signal_mom.values
    
    strategy_ret = np.zeros(n_days)
    curr_idx = -2 
    trade_count = 0
    days_held = 0 
    
    for i in range(n_days):
        if curr_idx != -2: days_held += 1
        row_mom = p_mom[i]
        if np.isnan(row_mom).all(): continue
            
        clean_mom = np.nan_to_num(row_mom, nan=-999)
        best_idx = np.argmax(clean_mom)
        best_val = clean_mom[best_idx]
        target_idx = curr_idx
        
        if allow_cash and best_val < 0:
            target_idx = -1
        else:
            if curr_idx < 0:
                target_idx = best_idx
            elif days_held >= min_holding:
                if best_idx != curr_idx and best_val > clean_mom[curr_idx] + threshold:
                    target_idx = best_idx
        
        if target_idx != curr_idx:
            if curr_idx != -2:
                strategy_ret[i] -= cost_rate
                trade_count += 1
            curr_idx, days_held = target_idx, 0
            
        if curr_idx >= 0: strategy_ret[i] += p_ret[i, curr_idx]
            
    return (1 + strategy_ret).cumprod(), trade_count

# ==========================================
# 4. Main Application
# ==========================================
def main():
    if 'params' not in st.session_state: st.session_state.params = load_config()
    
    with st.sidebar:
        st.title("🎛️ 策略控制台")
        all_etfs = get_all_etf_list()
        options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES
        selected_display = st.multiselect("核心标的池", options, default=options[:4])
        selected_codes = [x.split(" | ")[0] for x in selected_display]
        
        st.divider()
        start_date_input = st.date_input("开始日期", datetime(2021, 1, 1))
        initial_capital = st.number_input("初始本金", value=100000.0)
        
        with st.form("settings"):
            p_method = st.selectbox("动量逻辑", ['Classic (普通)', 'Risk-Adjusted (稳健)', 'MA Distance (趋势)'], index=1)
            p_lookback = st.slider("周期", 5, 60, 25)
            p_smooth = st.slider("平滑", 1, 10, 3)
            p_threshold = st.number_input("换仓阈值", 0.0, 0.05, 0.005, step=0.001, format="%.3f")
            p_min_holding = st.number_input("最小持仓天数", 1, 20, 3)
            p_allow_cash = st.checkbox("启用避险", True)
            run = st.form_submit_button("🚀 运行回测")

    st.markdown("## 🚀 核心资产轮动策略终端")
    
    if not selected_codes:
        st.info("请在左侧选择资产标的。")
        return

    raw_data, name_map = download_market_data(selected_codes, datetime.now().strftime('%Y%m%d'))
    if raw_data is None: return

    # Processing
    mom_all = calculate_momentum(raw_data, p_lookback, p_smooth, p_method)
    mask = raw_data.index >= pd.Timestamp(start_date_input)
    sliced_data, sliced_mom = raw_data[mask], mom_all[mask]
    daily_ret = sliced_data.pct_change().fillna(0)
    
    nav, trades = fast_backtest_vectorized(daily_ret, sliced_mom, p_threshold, p_min_holding, TRANSACTION_COST, p_allow_cash)
    
    # Results
    total_ret = nav[-1] - 1
    max_dd = ((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)).min()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_html("累计收益", f"{total_ret:.2%}", color="#c0392b"), unsafe_allow_html=True)
    c2.markdown(metric_html("最大回撤", f"{max_dd:.2%}", color="#27ae60"), unsafe_allow_html=True)
    c3.markdown(metric_html("交易次数", f"{trades}次"), unsafe_allow_html=True)
    c4.markdown(metric_html("当前状态", name_map.get(sliced_mom.iloc[-1].idxmax(), "Cash") if not (p_allow_cash and sliced_mom.iloc[-1].max() < 0) else "空仓"), unsafe_allow_html=True)

    # Charting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sliced_data.index, y=nav, name="策略净值", line=dict(color='#c0392b', width=2.5)))
    bm = (1 + daily_ret.mean(axis=1)).cumprod()
    fig.add_trace(go.Scatter(x=sliced_data.index, y=bm, name="等权基准", line=dict(color='#95a5a6', dash='dash')))
    fig.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sliced_data.tail(10).style.format("{:.3f}"), use_container_width=True)

if __name__ == "__main__":
    main()
