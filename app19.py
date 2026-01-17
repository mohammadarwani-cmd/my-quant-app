import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import time
import json
import os
import hashlib

# 安全导入 scipy
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# 0. 配置持久化管理 (Config Persistence)
# ==========================================
CONFIG_FILE = 'strategy_config.json'

# 默认标的池
DEFAULT_CODES = ["518880", "588000", "513100", "510180"]

DEFAULT_PARAMS = {
    'lookback': 25,
    'smooth': 3,
    'threshold': 0.005,
    'min_holding': 3,
    'allow_cash': True,
    'selected_codes': DEFAULT_CODES
}

def load_config():
    """从本地文件加载配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                config = DEFAULT_PARAMS.copy()
                config.update(saved_config)
                return config
        except Exception as e:
            return DEFAULT_PARAMS.copy()
    return DEFAULT_PARAMS.copy()

def save_config(config):
    """保存配置到本地文件"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception:
        pass

# ==========================================
# 1. 投行级页面配置 & CSS样式
# ==========================================
st.set_page_config(
    page_title="AlphaTarget | 核心资产轮动策略终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; font-family: 'Roboto', sans-serif; }
    .metric-card {
        background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .metric-label { color: #6c757d; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 5px; }
    .metric-value { color: #212529; font-size: 1.5rem; font-weight: 700; }
    .metric-sub { font-size: 0.8rem; color: #adb5bd; }
    .signal-banner {
        padding: 20px; border-radius: 8px; margin-bottom: 20px; color: white;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.2);
    }
    .dataframe { font-size: 13px !important; }
    .total-asset-header { font-size: 2rem; font-weight: bold; color: #1e3c72; margin-bottom: 0.5rem; }
    .total-asset-sub { font-size: 1rem; color: #666; }
</style>
""", unsafe_allow_html=True)

TRANSACTION_COST = 0.0001  # 万分之一

PRESET_ETFS = {
    "518880": "黄金ETF (避险)", "588000": "科创50 (硬科技)", "513100": "纳指100 (海外)",
    "510180": "上证180 (蓝筹)", "159915": "创业板指 (成长)", "510300": "沪深300 (大盘)",
    "510500": "中证500 (中盘)", "512890": "红利低波 (防御)", "513500": "标普500 (美股)",
    "512480": "半导体ETF (行业)", "512880": "证券ETF (Beta)"
}

# 辅助函数：根据名称生成柔和的颜色
def get_color_from_name(name):
    if name == 'Cash':
        return 'rgba(200, 200, 200, 0.2)' # 灰色代表空仓
    
    # 简单的哈希生成颜色
    hash_obj = hashlib.md5(name.encode())
    hex_dig = hash_obj.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    
    # 调整为浅色 (Pastel)
    r = (r + 255) // 2
    g = (g + 255) // 2
    b = (b + 255) // 2
    
    return f'rgba({r}, {g}, {b}, 0.25)' # 透明度0.25

# ==========================================
# 2. 数据层 (Data Layer)
# ==========================================

@st.cache_data(ttl=3600*12) 
def get_all_etf_list():
    try:
        df = ak.fund_etf_spot_em()
        df['display'] = df['代码'] + " | " + df['名称']
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def download_market_data(codes_list, end_date_str):
    start_str = '20150101' 
    price_dict = {}
    name_map = {}
    
    etf_list = get_all_etf_list()
    
    for code in codes_list:
        name = code
        if code in PRESET_ETFS:
            name = PRESET_ETFS[code].split(" ")[0]
        elif not etf_list.empty:
            match = etf_list[etf_list['代码'] == code]
            if not match.empty:
                name = match.iloc[0]['名称']
        
        name_map[code] = name
        
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                price_dict[name] = df['收盘'].astype(float)
        except Exception:
            continue

    if not price_dict:
        return None, None

    data = pd.concat(price_dict, axis=1).sort_index().ffill()
    data.dropna(how='all', inplace=True)
    
    if len(data) < 20:
        return None, None
        
    return data, name_map

# ==========================================
# 3. 策略内核 (Strategy Core)
# ==========================================

def calculate_momentum(data, lookback, smooth):
    mom = data.pct_change(lookback)
    if smooth > 1:
        mom = mom.rolling(smooth).mean()
    return mom

def fast_backtest_vectorized(daily_ret, mom_df, threshold, min_holding=1, cost_rate=0.0001, allow_cash=True):
    signal_mom = mom_df.shift(1)
    
    n_days, n_assets = daily_ret.shape
    p_ret = daily_ret.values
    p_mom = signal_mom.values
    
    strategy_ret = np.zeros(n_days)
    curr_idx = -2 # -2: 初始, -1: Cash, 0~N: 资产
    
    trade_count = 0
    days_held = 0 # 记录当前持仓天数
    
    for i in range(n_days):
        # 1. 每日自然持仓时间增加
        if curr_idx != -2:
            days_held += 1
            
        row_mom = p_mom[i]
        
        if np.isnan(row_mom).all(): 
            continue
            
        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)
        
        best_idx = np.argmax(clean_mom)
        best_val = clean_mom[best_idx]
        
        target_idx = curr_idx
        
        # --- 策略逻辑 ---
        if allow_cash and best_val < 0:
            target_idx = -1 # 建议空仓
        else:
            if curr_idx == -2:
                if best_val > -np.inf: 
                    target_idx = best_idx
            elif curr_idx == -1:
                if best_val > 0 or (not allow_cash):
                    target_idx = best_idx
            else:
                is_stop_loss = (target_idx == -1) 
                
                if is_stop_loss:
                    pass
                else:
                    if days_held >= min_holding:
                        curr_val = clean_mom[curr_idx]
                        if best_idx != curr_idx:
                            if best_val > curr_val + threshold:
                                target_idx = best_idx
                    else:
                        target_idx = curr_idx
        
        # --- 交易执行 ---
        if target_idx != curr_idx:
            if curr_idx != -2:
                strategy_ret[i] -= cost_rate
                trade_count += 1
                days_held = 0
            
            curr_idx = target_idx
            
        if curr_idx >= 0:
            strategy_ret[i] += p_ret[i, curr_idx]
            
    equity_curve = (1 + strategy_ret).cumprod()
    total_ret = equity_curve[-1] - 1
    
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    
    return total_ret, max_dd, equity_curve, trade_count

# ==========================================
# 4. 分析师工具箱
# ==========================================

def calculate_pro_metrics(equity_curve, benchmark_curve, trade_count):
    if len(equity_curve) < 2: return {}
    s_eq = pd.Series(equity_curve)
    s_bm = pd.Series(benchmark_curve) if len(benchmark_curve) == len(equity_curve) else None
    daily_ret = s_eq.pct_change().fillna(0)
    bm_ret = s_bm.pct_change().fillna(0) if s_bm is not None else None
    days = len(equity_curve)
    
    total_ret = equity_curve[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / days) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    rf = 0.03
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)
    
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    calmar = ann_ret / (abs(max_dd) + 1e-9)
    
    alpha, beta = 0.0, 0.0
    if HAS_SCIPY and bm_ret is not None and len(bm_ret) > 10:
        try:
            slope, intercept, _, _, _ = stats.linregress(bm_ret.values[1:], daily_ret.values[1:])
            beta = slope
            alpha = intercept * 252
        except: pass
            
    return {
        "Total Return": total_ret, "CAGR": ann_ret, "Volatility": ann_vol,
        "Max Drawdown": max_dd, "Sharpe Ratio": sharpe, "Calmar Ratio": calmar,
        "Alpha": alpha, "Beta": beta, "Trades": trade_count
    }

def optimize_parameters(data, allow_cash, min_holding):
    lookbacks = range(20, 31, 2)
    smooths = range(1, 6, 1)    
    thresholds = np.arange(0.0, 0.013, 0.002) 
    
    daily_ret = data.pct_change().fillna(0)
    n_days = len(daily_ret) 
    results = []
    
    total_iters = len(lookbacks) * len(smooths) * len(thresholds)
    my_bar = st.progress(0, text="正在寻找最佳参数组合...")
    
    idx = 0
    for lb in lookbacks:
        for sm in smooths:
            mom = calculate_momentum(data, lb, sm)
            for th in thresholds:
                ret, dd, _, count = fast_backtest_vectorized(
                    daily_ret, mom, th, 
                    min_holding=min_holding,
                    cost_rate=TRANSACTION_COST, 
                    allow_cash=allow_cash
                )
                score = ret / (abs(dd) + 0.05)
                results.append([lb, sm, th, ret, count, dd, score])
                
                idx += 1
                if idx % 50 == 0:
                    my_bar.progress(min(idx / total_iters, 1.0))
                    
    my_bar.empty()
    df_res = pd.DataFrame(results, columns=['周期', '平滑', '阈值', '累计收益', '调仓次数', '最大回撤', '得分'])
    
    df_res['年化收益'] = (1 + df_res['累计收益']) ** (252 / n_days) - 1
    return df_res

# ==========================================
# 5. 主程序 UI
# ==========================================

def main():
    if 'params' not in st.session_state:
        saved_config = load_config()
        st.session_state.params = saved_config

    with st.sidebar:
        st.title("🎛️ 策略控制台")
        
        st.subheader("1. 资产池配置")
        all_etfs = get_all_etf_list()
        options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES
        current_selection_codes = st.session_state.params.get('selected_codes', DEFAULT_CODES)
        
        default_display = []
        if not all_etfs.empty:
            for code in current_selection_codes:
                match = all_etfs[all_etfs['代码'] == code]
                if not match.empty:
                    default_display.append(match.iloc[0]['display'])
                else:
                    for opt in options:
                        if opt.startswith(code):
                            default_display.append(opt)
                            break
        else:
            default_display = current_selection_codes
            
        valid_defaults = [x for x in default_display if x in options]
        selected_display = st.multiselect("核心标的池", options, default=valid_defaults)
        selected_codes = [x.split(" | ")[0] for x in selected_display]
        
        st.divider()
        st.subheader("2. 资金管理")
        
        date_mode = st.radio("回测区间", ["全历史", "自定义"], index=0)
        start_date = datetime(2018, 1, 1)
        end_date = datetime.now()
        
        if date_mode == "自定义":
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start", datetime(2019, 1, 1))
            end_date = c2.date_input("End", datetime.now())
            start_date = datetime.combine(start_date, datetime.min.time())
            end_date = datetime.combine(end_date, datetime.min.time())

        initial_capital = st.number_input("初始本金", value=100000.0, step=10000.0)

        st.divider()
        st.subheader("3. 策略内核参数")
        
        p_lookback = st.slider("动量周期 (Lookback)", 5, 60, st.session_state.params.get('lookback', 25))
        p_smooth = st.slider("平滑窗口 (Smooth)", 1, 10, st.session_state.params.get('smooth', 3))
        p_threshold = st.number_input("换仓阈值 (Threshold)", 0.0, 0.05, st.session_state.params.get('threshold', 0.005), step=0.001, format="%.3f")
        
        st.markdown("---")
        st.markdown("**🛑 交易频率控制**")
        p_min_holding = st.slider("最小持仓天数 (Min Hold)", 1, 20, st.session_state.params.get('min_holding', 3), help="买入后必须持有的最少交易日数。设置为1即不限制。")
        
        p_allow_cash = st.checkbox("启用绝对动量避险 (Cash Protection)", value=st.session_state.params.get('allow_cash', True))
        
        current_params = {
            'lookback': p_lookback, 'smooth': p_smooth, 'threshold': p_threshold,
            'min_holding': p_min_holding, 'allow_cash': p_allow_cash, 'selected_codes': selected_codes
        }
        
        if current_params != st.session_state.params:
            st.session_state.params = current_params
            save_config(current_params)
            
        if st.button("🔄 重置默认"):
            st.session_state.params = DEFAULT_PARAMS.copy()
            save_config(DEFAULT_PARAMS)
            st.rerun()

    st.markdown("## 🚀 核心资产轮动策略终端 (Anti-Whipsaw Ver.)")
    
    if not selected_codes:
        st.warning("请选择标的。")
        st.stop()
        
    utc_now = datetime.now(timezone.utc)
    beijing_now = utc_now + timedelta(hours=8)
    end_date_str = beijing_now.strftime('%Y%m%d')

    with st.spinner("正在接入市场数据终端 (Smart-Link)..."):
        raw_data, name_map = download_market_data(selected_codes, end_date_str)
        
    if raw_data is None:
        st.error("数据不足或下载失败。")
        st.stop()

    daily_ret_all = raw_data.pct_change().fillna(0)
    mom_all = calculate_momentum(raw_data, p_lookback, p_smooth)
    
    mask = (raw_data.index >= start_date) & (raw_data.index <= end_date)
    sliced_data = raw_data.loc[mask]
    sliced_mom = mom_all.loc[mask] 
    sliced_ret = daily_ret_all.loc[mask]
    
    if sliced_data.empty:
        st.error("区间内无数据")
        st.stop()

    signal_mom = sliced_mom.shift(1)
    dates = sliced_ret.index
    
    cash = initial_capital
    share_val = 0.0
    curr_hold = None
    days_held = 0
    
    holdings_history = []
    asset_curve = []
    trade_count_real = 0
    
    for i, date in enumerate(dates):
        if curr_hold is not None:
            days_held += 1
            
        row = signal_mom.loc[date]
        r_today = sliced_ret.loc[date]
        
        target = curr_hold
        
        if not row.isna().all():
            clean_row = row.fillna(-np.inf)
            best_asset = clean_row.idxmax()
            best_score = clean_row.max()
            
            if p_allow_cash and best_score < 0:
                target = 'Cash'
            else:
                if curr_hold is None or curr_hold == 'Cash':
                    target = best_asset
                else:
                    if days_held >= p_min_holding:
                        curr_score = clean_row.get(curr_hold, -np.inf)
                        if best_asset != curr_hold:
                            if best_score > curr_score + p_threshold:
                                target = best_asset
                    else:
                        target = curr_hold

        day_return = 0.0
        if curr_hold and curr_hold != 'Cash' and curr_hold in r_today:
            day_return = r_today[curr_hold]
        
        share_val = share_val * (1 + day_return)
        
        if target != curr_hold:
            if curr_hold is not None:
                total_equity = share_val + cash
                cost = total_equity * TRANSACTION_COST
                if cash >= cost: cash -= cost
                else: share_val -= cost
                trade_count_real += 1
                days_held = 0
                
            if target == 'Cash':
                cash += share_val
                share_val = 0.0
            else:
                total = share_val + cash
                share_val = total
                cash = 0.0
            curr_hold = target
            
        holdings_history.append(target if target else "Cash")
        asset_curve.append(share_val + cash)

    df_res = pd.DataFrame({
        '总资产': asset_curve,
        '持仓': holdings_history
    }, index=dates)
    
    _, _, nav_series, _ = fast_backtest_vectorized(
        sliced_ret, sliced_mom, p_threshold, 
        min_holding=p_min_holding, 
        cost_rate=TRANSACTION_COST, 
        allow_cash=p_allow_cash
    )
    df_res['策略净值'] = nav_series
    
    bm_curve = (1 + sliced_ret.mean(axis=1)).cumprod()
    
    latest_mom = mom_all.iloc[-1].dropna().sort_values(ascending=False)
    last_hold = holdings_history[-1]
    
    col_sig1, col_sig2 = st.columns([2, 1])
    with col_sig1:
        hold_name = name_map.get(last_hold, last_hold) if last_hold != 'Cash' else '🛡️ 空仓避险 (Cash)'
        lock_msg = f"(已持仓 {days_held} 天)" if last_hold != 'Cash' else ""
        if days_held < p_min_holding and last_hold != 'Cash':
            lock_msg += " 🔒 **锁定中**"
            
        st.markdown(f"""
        <div class="signal-banner">
            <h3 style="margin:0">📌 当前持仓: {hold_name}</h3>
            <div style="margin-top:10px;">
                最小持仓限制: {p_min_holding} 天 {lock_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_sig2:
        st.markdown("**🏆 实时排名**")
        for i, (asset, score) in enumerate(latest_mom.head(3).items()):
            display_name = name_map.get(asset, asset)
            st.markdown(f"{i+1}. **{display_name}**: `{score:.2%}`")

    with st.expander("🛠️ 参数优化 (含持仓天数锁定)", expanded=False):
        if st.button("运行参数寻优"):
            opt_df = optimize_parameters(sliced_data, p_allow_cash, p_min_holding)
            best_r = opt_df.loc[opt_df['累计收益'].idxmax()]
            
            c1, c2, c3 = st.columns([1,1,2])
            with c1: st.metric("最佳年化", f"{best_r['年化收益']:.1%}", f"参数: {int(best_r['周期'])}/{best_r['阈值']:.3f}")
            with c2: st.metric("对应回撤", f"{best_r['最大回撤']:.1%}", f"调仓: {int(best_r['调仓次数'])}")
            with c3:
                pivot = opt_df.pivot_table(index='阈值', columns='周期', values='得分')
                fig = px.imshow(pivot, labels=dict(color="Score"), aspect="auto", origin='lower')
                fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)

    # 报表
    metrics = calculate_pro_metrics(df_res['策略净值'].values, bm_curve.values, trade_count_real)
    
    # --- 新增: 计算近半年收益 ---
    six_months_ago = df_res.index[-1] - timedelta(days=180)
    # 找到最近的交易日索引
    idx_6m = df_res.index.searchsorted(six_months_ago)
    if idx_6m < len(df_res):
        nav_6m_start = df_res['策略净值'].iloc[idx_6m]
        nav_now = df_res['策略净值'].iloc[-1]
        ret_6m = nav_now / nav_6m_start - 1
        
        # Benchmark 6m
        bm_6m_start = bm_curve.iloc[idx_6m]
        bm_now = bm_curve.iloc[-1]
        bm_ret_6m = bm_now / bm_6m_start - 1
    else:
        ret_6m = 0.0
        bm_ret_6m = 0.0

    st.markdown("### 📊 策略表现概览")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("累计收益", f"{metrics.get('Total Return',0):.1%}")
    m2.metric("年化收益", f"{metrics.get('CAGR',0):.1%}")
    # 新增板块
    m3.metric("近半年收益", f"{ret_6m:.1%}", f"超额: {ret_6m - bm_ret_6m:+.1%}")
    m4.metric("最大回撤", f"{metrics.get('Max Drawdown',0):.1%}")
    m5.metric("夏普比率", f"{metrics.get('Sharpe Ratio',0):.2f}")
    m6.metric("交易次数", f"{trade_count_real}")

    tab1, tab2 = st.tabs(["📈 综合图表 (含持仓)", "📝 持仓明细"])
    with tab1:
        # === 构建综合图表 ===
        # 创建上下两个子图，共享X轴
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3], # 上7下3
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )

        # 1. 绘制净值曲线 (Top)
        fig.add_trace(go.Scatter(
            x=df_res.index, y=df_res['策略净值'], 
            name="策略净值", 
            line=dict(color='#d62728', width=2),
            hovertemplate="日期: %{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra></extra>"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_res.index, y=bm_curve, 
            name="基准", 
            line=dict(color='grey', dash='dash'),
            hovertemplate="基准: %{y:.4f}<extra></extra>"
        ), row=1, col=1)

        # 2. 绘制回撤面积图 (Bottom)
        drawdown_series = (df_res['策略净值'] - df_res['策略净值'].cummax()) / df_res['策略净值'].cummax()
        fig.add_trace(go.Scatter(
            x=df_res.index, y=drawdown_series, 
            name="回撤", 
            fill='tozeroy', 
            line=dict(color='darkred', width=1),
            hovertemplate="回撤: %{y:.2%}<extra></extra>"
        ), row=2, col=1)

        # 3. 添加持仓背景色块 (High-Level Visualization)
        # 将连续的持仓合并为一个区间
        df_res['持仓名称'] = df_res['持仓'].map(lambda x: name_map.get(x, x))
        
        # 识别持仓变化点
        df_res['持仓变化'] = df_res['持仓'] != df_res['持仓'].shift(1)
        change_indices = df_res[df_res['持仓变化']].index.tolist()
        
        # 如果第一天没有变化（shift产生的），补上开始时间
        if df_res.index[0] not in change_indices:
            change_indices.insert(0, df_res.index[0])
            
        # 添加结束时间作为哨兵
        change_indices.append(df_res.index[-1] + timedelta(days=1))

        shapes = []
        # 遍历区间添加背景色
        for i in range(len(change_indices) - 1):
            start_t = change_indices[i]
            end_t = change_indices[i+1]
            # 获取该区间的持仓名称 (取start_t的数据)
            # 注意：由于change_indices是基于变化的，start_t那天的持仓就是新持仓
            try:
                # 兼容处理：确保索引存在
                if start_t > df_res.index[-1]: continue
                current_code = df_res.loc[start_t, '持仓']
                current_name = df_res.loc[start_t, '持仓名称']
                
                color = get_color_from_name(current_code)
                
                # 添加背景矩形
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="paper",
                    x0=start_t, x1=end_t,
                    y0=0, y1=1,
                    fillcolor=color,
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                ))
                
                # 为了能在图上直接看到是什么，我们在区间中间加一个隐形的Scatter点用于Hover显示名称
                # 或者更简单：在图表中间加一个Annotation（如果区间够长）
                mid_point = start_t + (end_t - start_t) / 2
                if (end_t - start_t).days > 10: # 只在长区间显示文字，避免拥挤
                    fig.add_annotation(
                        x=mid_point, y=0.05, # 底部显示
                        xref="x", yref="paper", # 相对于第一个子图
                        text=current_name.split(' ')[0], # 简短名称
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        opacity=0.7
                    )
            except Exception:
                pass

        fig.update_layout(
            shapes=shapes,
            height=600,
            title_text="策略综合分析 (背景色代表不同持仓)",
            hovermode="x unified",
            xaxis=dict(
                rangeslider=dict(visible=False), # 默认不显示底部的缩略条，因为支持直接拖动
                type="date"
            )
        )
        
        # Y轴格式
        fig.update_yaxes(title_text="净值", row=1, col=1)
        fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        df_show = df_res.copy()
        df_show['持仓名称'] = df_show['持仓'].map(lambda x: name_map.get(x, x))
        st.dataframe(df_show[['总资产', '持仓名称']].sort_index(ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
