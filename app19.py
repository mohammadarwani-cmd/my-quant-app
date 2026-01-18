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
# 0. 配置持久化管理
# ==========================================
CONFIG_FILE = 'strategy_config.json'

# 默认标的池 (根据您的常用配置)
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
        except Exception:
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
# 1. 页面配置 & CSS样式
# ==========================================
st.set_page_config(
    page_title="AlphaTarget | 核心资产轮动终端 Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', 'Roboto', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    
    /* 指标卡片 */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 20px 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        text-align: center;
        height: 100%;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.05); }
    .metric-label { color: #8898aa; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }
    .metric-value { color: #32325d; font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
    .metric-sub { font-size: 0.8rem; color: #adb5bd; margin-top: 6px; }

    /* 信号横幅 */
    .signal-banner {
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
        background: linear-gradient(135deg, #172a74 0%, #21a1f1 100%);
        box-shadow: 0 10px 20px rgba(33, 161, 241, 0.2);
    }
    
    /* === 交易日记样式优化 (Grid 布局) === */
    .log-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); /* 自动填充网格，最小110px */
        gap: 6px;
        width: 100%;
        align-items: center;
    }
    .market-tag {
        display: flex;
        align-items: center;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 3px 6px;
        font-size: 0.75rem; /* 字体改小 */
        justify-content: space-between;
        white-space: nowrap; /* 不换行 */
        overflow: hidden;
    }
    .tag-name {
        font-weight: 600;
        color: #525f7f;
        padding-left: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis; /* 超长省略 */
        max-width: 55px;
    }
    .tag-val {
        font-family: 'Consolas', monospace;
        font-weight: 700;
        margin-left: 4px;
        font-size: 0.75rem;
    }
    .trend-up { color: #d62728; }
    .trend-down { color: #2ca02c; } 
    .trend-flat { color: #adb5bd; }
    
    .op-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #172a74;
        background-color: #e3e8ff;
        border: 1px solid #c7d0ff;
    }
    .op-badge-none {
        color: #ddd;
        font-size: 0.8rem;
    }
    
    /* 表格覆盖 - 强制对齐 */
    table.dataframe { border-collapse: separate !important; border-spacing: 0 4px !important; width: 100%; border: none !important; table-layout: fixed; }
    table.dataframe th { 
        background-color: transparent !important; 
        color: #8898aa !important; 
        text-transform: uppercase; 
        font-size: 0.75rem; 
        border: none !important;
        padding-bottom: 10px;
        text-align: left !important;
    }
    table.dataframe td { 
        background-color: #ffffff; 
        border-top: 1px solid #f1f3f5; 
        border-bottom: 1px solid #f1f3f5; 
        padding: 10px 8px; 
        vertical-align: middle !important;
        font-size: 0.85rem;
    }
    /* 列宽控制 */
    table.dataframe td:nth-child(1) { width: 90px; white-space: nowrap; color: #8898aa; } /* 日期 */
    table.dataframe td:nth-child(2) { width: 100px; white-space: nowrap; font-weight: bold; } /* 持仓 */
    table.dataframe td:nth-child(3) { width: 160px; } /* 操作 */
    table.dataframe td:nth-child(4) { width: 100px; white-space: nowrap; font-family: monospace; } /* 总资产 */
    table.dataframe td:nth-child(5) { width: auto; } /* 市场全景 (自适应) */
    
    table.dataframe tr:hover td { background-color: #f8f9fe; }
</style>
""", unsafe_allow_html=True)

TRANSACTION_COST = 0.0001  # 万分之一

PRESET_ETFS = {
    "518880": "黄金ETF", "588000": "科创50", "513100": "纳指100",
    "510180": "上证180", "159915": "创业板指", "510300": "沪深300",
    "510500": "中证500", "512890": "红利低波", "513500": "标普500",
    "512480": "半导体", "512880": "证券ETF"
}

# 颜色生成器
def get_color_from_name(name, alpha=0.2):
    if name == 'Cash' or name == '空仓':
        return f'rgba(200, 200, 200, {alpha})'
    hash_obj = hashlib.md5(name.encode())
    hex_dig = hash_obj.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    # 调亮
    r = (r + 255) // 2
    g = (g + 255) // 2
    b = (b + 255) // 2
    return f'rgba({r}, {g}, {b}, {alpha})'

def get_hex_color(name):
    """获取不透明的HEX颜色用于线条"""
    if name == 'Cash': return '#95a5a6'
    hash_obj = hashlib.md5(name.encode())
    hex_dig = hash_obj.hexdigest()
    return f"#{hex_dig[:6]}"

def metric_html(label, value, sub="", color="#2c3e50"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """

# ==========================================
# 2. 数据层
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
    start_str = '20190101' # 下载多一点数据用于计算初始指标
    price_dict = {}
    name_map = {}
    
    etf_list = get_all_etf_list()
    
    for code in codes_list:
        name = code
        if code in PRESET_ETFS:
            name = PRESET_ETFS[code]
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
# 3. 策略与优化内核
# ==========================================

def calculate_momentum(data, lookback, smooth):
    mom = data.pct_change(lookback)
    if smooth > 1:
        mom = mom.rolling(smooth).mean()
    return mom

def fast_backtest_vectorized(daily_ret, mom_df, threshold, min_holding=1, cost_rate=0.0001, allow_cash=True):
    # 向量化回测：
    # 注意：传入的 mom_df 和 daily_ret 已经是切片后的时间段
    # 但 mom_df 是基于全历史计算的，所以第一行就有值
    # 但是，我们必须 shift(1) 来避免未来函数
    
    signal_mom = mom_df.shift(1)
    
    # [关键修复]：由于 shift(1) 会导致切片后的第一天变成 NaN，导致第一天无法交易
    # 主程序也是 signal_mom = sliced_mom.shift(1)，所以主程序第一天也是不交易的
    # 因此，只要数据源一致（都是基于全历史计算mom再切片），两边的逻辑就是对齐的。
    # 唯一的微小差异是第一天的收益归属，但可以忽略不计。
    
    n_days, _ = daily_ret.shape
    p_ret = daily_ret.values
    p_mom = signal_mom.values
    strategy_ret = np.zeros(n_days)
    curr_idx = -2 
    trade_count = 0
    days_held = 0 
    
    for i in range(n_days):
        if curr_idx != -2: days_held += 1
        row_mom = p_mom[i]
        
        # 如果整行都是NaN（比如第一天），直接跳过，保持空仓
        if np.isnan(row_mom).all(): continue
            
        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)
        best_idx = np.argmax(clean_mom)
        best_val = clean_mom[best_idx]
        target_idx = curr_idx
        
        if allow_cash and best_val < 0:
            target_idx = -1
        else:
            if curr_idx == -2:
                if best_val > -np.inf: target_idx = best_idx
            elif curr_idx == -1:
                if best_val > 0 or (not allow_cash): target_idx = best_idx
            else:
                if days_held >= min_holding:
                    curr_val = clean_mom[curr_idx]
                    if best_idx != curr_idx:
                        if best_val > curr_val + threshold:
                            target_idx = best_idx
                    else:
                        target_idx = curr_idx
                        
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
    max_dd = ((equity_curve - cummax) / cummax).min()
    return total_ret, max_dd, equity_curve, trade_count

def optimize_parameters_3d(raw_data, mask, allow_cash, min_holding):
    # [关键修复]：接收 raw_data（全量数据）和 mask（切片掩码）
    # 这样可以在计算 momentum 时利用切片前的历史数据，避免“冷启动”偏差。
    
    # Lookback: 20 ~ 25, 步长 1
    # Smooth: 1 ~ 5, 步长 1
    # Threshold: 0 ~ 0.01, 步长 0.001
    
    lookbacks = range(20, 26, 1)         
    smooths = range(1, 6, 1)            
    thresholds = np.arange(0.0, 0.011, 0.001) 
    
    # 1. 基于全量数据计算收益率
    daily_ret_all = raw_data.pct_change().fillna(0)
    
    results = []
    total_iters = len(lookbacks) * len(smooths) * len(thresholds)
    my_bar = st.progress(0, text=f"正在进行精度三维扫描 (0/{total_iters})...")
    
    count = 0
    start_time = time.time()
    
    for lb in lookbacks:
        for sm in smooths:
            # 2. 基于全量数据计算动量（关键！）
            # 这样2021-01-01那天的动量值是基于2020年数据算出来的，不是NaN
            mom_all = calculate_momentum(raw_data, lb, sm)
            
            # 3. 此时再进行切片，传入回测函数
            sub_ret = daily_ret_all.loc[mask]
            sub_mom = mom_all.loc[mask]
            
            for th in thresholds:
                ret, dd, _, trades = fast_backtest_vectorized(
                    sub_ret, sub_mom, th, 
                    min_holding=min_holding, cost_rate=TRANSACTION_COST, allow_cash=allow_cash
                )
                score = ret / (abs(dd) + 0.1)
                results.append([lb, sm, th, ret, trades, dd, score])
                count += 1
        
        my_bar.progress(min(count / total_iters, 1.0))
                    
    my_bar.empty()
    st.toast(f"扫描完成！耗时 {time.time()-start_time:.1f} 秒", icon="✅")
    
    df_res = pd.DataFrame(results, columns=['Lookback', 'Smooth', 'Threshold', 'Return', 'Trades', 'MaxDD', 'Score'])
    df_res['Annual_Ret'] = (1 + df_res['Return']) ** (252 / len(sub_ret)) - 1
    return df_res

# ==========================================
# 4. 主程序 UI
# ==========================================

def main():
    if 'params' not in st.session_state:
        st.session_state.params = load_config()

    # --- 侧边栏配置区 ---
    with st.sidebar:
        st.title("🎛️ 策略控制台")
        
        # 1. 标的选择
        st.subheader("1. 标的池配置")
        all_etfs = get_all_etf_list()
        etf_options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES
        
        current_codes = st.session_state.params.get('selected_codes', DEFAULT_CODES)
        default_display = []
        if not all_etfs.empty:
            for c in current_codes:
                match = all_etfs[all_etfs['代码'] == c]
                if not match.empty: default_display.append(match.iloc[0]['display'])
                else: default_display.append(c)
        else:
            default_display = current_codes
            
        selected_display = st.multiselect(
            "选择核心资产 (Core Assets)", 
            etf_options, 
            default=[x for x in default_display if x in etf_options]
        )
        selected_codes_final = [x.split(" | ")[0] for x in selected_display]

        # 2. 参数表单
        with st.form("strategy_form"):
            st.divider()
            st.subheader("2. 资金与时间")
            
            c_d1, c_d2 = st.columns(2)
            p_start_date = c_d1.date_input("开始日期", datetime(2021, 1, 1))
            p_end_date = c_d2.date_input("结束日期", datetime.now())
            
            p_initial_capital = st.number_input("初始本金 (¥)", value=100000.0, step=10000.0)

            st.divider()
            st.subheader("3. 策略三维参数")
            
            c_p1, c_p2 = st.columns(2)
            p_lookback = c_p1.number_input("Lookback (周期)", 5, 120, st.session_state.params.get('lookback', 25))
            p_smooth = c_p2.number_input("Smooth (平滑)", 1, 30, st.session_state.params.get('smooth', 3))
            p_threshold = st.number_input("Threshold (阈值)", 0.0, 0.05, st.session_state.params.get('threshold', 0.005), step=0.001, format="%.3f")
            
            st.markdown("---")
            p_min_holding = st.number_input("最小持仓天数", 1, 30, st.session_state.params.get('min_holding', 3))
            p_allow_cash = st.checkbox("允许空仓 (Cash Protection)", value=st.session_state.params.get('allow_cash', True))
            
            st.markdown("### ")
            submitted = st.form_submit_button("🚀 确认修改并运行", type="primary")

    # --- 逻辑处理 ---
    if submitted or 'run_once' not in st.session_state:
        st.session_state.run_once = True
        current_params = {
            'lookback': p_lookback, 'smooth': p_smooth, 'threshold': p_threshold,
            'min_holding': p_min_holding, 'allow_cash': p_allow_cash, 'selected_codes': selected_codes_final
        }
        st.session_state.params = current_params
        save_config(current_params)
    
    run_codes = st.session_state.params['selected_codes']
    
    st.markdown("## 🚀 核心资产轮动策略终端 Pro")
    
    if not run_codes:
        st.warning("👈 请在侧边栏选择标的并点击【确认运行】")
        st.stop()
        
    end_date_str = p_end_date.strftime('%Y%m%d')
    start_date_ts = datetime.combine(p_start_date, datetime.min.time())
    end_date_ts = datetime.combine(p_end_date, datetime.min.time())

    with st.spinner("正在获取市场数据..."):
        raw_data, name_map = download_market_data(run_codes, end_date_str)
        
    if raw_data is None:
        st.error("无法获取数据，请检查网络或代码有效性。")
        st.stop()

    daily_ret_all = raw_data.pct_change().fillna(0)
    mom_all = calculate_momentum(raw_data, p_lookback, p_smooth)
    
    mask = (raw_data.index >= start_date_ts) & (raw_data.index <= end_date_ts)
    sliced_data = raw_data.loc[mask]
    
    if sliced_data.empty:
        st.error(f"所选时间段 {p_start_date} 至 {p_end_date} 无数据，请调整时间。")
        st.stop()
        
    sliced_mom = mom_all.loc[mask]
    sliced_ret = daily_ret_all.loc[mask]
    
    # === 策略回测逻辑 ===
    signal_mom = sliced_mom.shift(1)
    dates = sliced_ret.index
    
    cash = p_initial_capital
    share_val = 0.0
    curr_hold = None
    days_held = 0
    holdings_history = []
    total_assets_curve = []
    daily_details = []
    
    ordered_names = [name_map.get(c, c) for c in run_codes if c in name_map]
    
    def format_market_perf_html(row, ordered_keys, name_mapping):
        html_parts = []
        html_parts.append('<div class="log-container">')
        for name in ordered_keys:
            if name in row.index:
                val = row[name]
                line_color = get_hex_color(name)
                arrow = "▲" if val > 0 else "▼" if val < 0 else "-"
                val_class = "trend-up" if val > 0 else "trend-down" if val < 0 else "trend-flat"
                
                # 简化 Tag 结构
                html = f"""
                <div class="market-tag">
                    <span class="tag-name" style="border-left: 3px solid {line_color}">{name}</span>
                    <span class="tag-val {val_class}">{arrow}{abs(val):.2%}</span>
                </div>
                """
                html_parts.append(html)
        html_parts.append('</div>')
        return "".join(html_parts)

    for i, date in enumerate(dates):
        r_today = sliced_ret.loc[date]
        market_perf_html = format_market_perf_html(r_today, ordered_names, name_map)
        
        if curr_hold: days_held += 1
        
        row = signal_mom.loc[date]
        target = curr_hold
        
        if not row.isna().all():
            clean_row = row.fillna(-np.inf)
            best_asset = clean_row.idxmax()
            best_val = clean_row.max()
            
            if p_allow_cash and best_val < 0:
                target = 'Cash'
            else:
                if curr_hold is None or curr_hold == 'Cash':
                    target = best_asset
                else:
                    if days_held >= p_min_holding:
                        curr_val = clean_row.get(curr_hold, -np.inf)
                        if best_asset != curr_hold:
                            if best_val > curr_val + p_threshold:
                                target = best_asset
                        else:
                            target = curr_hold

        day_return = 0.0
        if curr_hold and curr_hold != 'Cash' and curr_hold in r_today:
            day_return = r_today[curr_hold]
            
        share_val = share_val * (1 + day_return)
        
        note_html = '<span class="op-badge-none">-</span>'
        if target != curr_hold:
            if curr_hold is not None:
                total_equity = share_val + cash
                cost = total_equity * TRANSACTION_COST
                if cash >= cost: cash -= cost
                else: share_val -= cost
                days_held = 0
                
                old = name_map.get(curr_hold, curr_hold) if curr_hold else "Cash"
                new = name_map.get(target, target) if target else "Cash"
                note_html = f'<span class="op-badge">🔄 {old} → {new}</span>'
            
            if target == 'Cash':
                cash += share_val
                share_val = 0.0
            else:
                total = share_val + cash
                share_val = total
                cash = 0.0
            
            curr_hold = target
            
        current_total = share_val + cash
        holdings_history.append(curr_hold if curr_hold else "Cash")
        total_assets_curve.append(current_total)
        
        display_hold = name_map.get(curr_hold, curr_hold) if curr_hold and curr_hold != 'Cash' else 'Cash'
        daily_details.append({
            "日期": date,
            "当前持仓": display_hold, # 去掉 <b> 以防止表格错位，样式已在CSS中加粗
            "日收益": day_return, 
            "总资产": current_total,
            "操作": note_html,
            "市场全景": market_perf_html
        })

    # === 结果整合 ===
    df_res = pd.DataFrame({
        '总资产': total_assets_curve,
        '持仓': holdings_history
    }, index=dates)
    
    df_res['净值'] = df_res['总资产'] / p_initial_capital
    bm_curve = (1 + sliced_ret.mean(axis=1)).cumprod() 
    
    total_ret = df_res['净值'].iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / len(dates)) - 1
    max_dd = ((df_res['净值'] - df_res['净值'].cummax()) / df_res['净值'].cummax()).min()
    
    # === UI 展示 ===
    last_h = holdings_history[-1]
    h_name = name_map.get(last_h, last_h) if last_h != 'Cash' else '🛡️ 空仓 (Cash)'
    
    col_sig, col_kpi = st.columns([1, 2])
    with col_sig:
        st.markdown(f"""
        <div class="signal-banner">
            <h3 style="margin:0">当前持仓: {h_name}</h3>
            <p style="margin:8px 0 0 0; opacity:0.9; font-size:0.95rem;">连续持仓: <b>{days_held}</b> 天</p>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(metric_html("总收益率", f"{total_ret:+.1%}", "Total Return", "#d62728"), unsafe_allow_html=True)
        k2.markdown(metric_html("年化收益", f"{ann_ret:+.1%}", "CAGR", "#d62728"), unsafe_allow_html=True)
        k3.markdown(metric_html("最大回撤", f"{max_dd:.1%}", "Max Drawdown", "#2ca02c"), unsafe_allow_html=True)
        k4.markdown(metric_html("当前资产", f"¥{current_total:,.0f}", "Asset", "#2c3e50"), unsafe_allow_html=True)

    # Tabs (恢复年度/月度回报 Tab)
    tab1, tab2, tab3, tab4 = st.tabs(["📈 综合走势对比", "📅 年度/月度回报", "🛠️ 3D参数优化引擎", "📝 交易日记"])
    
    with tab1:
        st.markdown("##### 策略 vs 基准 vs 标的走势 (归一化对比)")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_res.index, y=df_res['净值'], name="🤖 策略净值", 
            line=dict(color='#d62728', width=3), mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=bm_curve.index, y=bm_curve, name="📊 等权基准", 
            line=dict(color='gray', width=2, dash='dash'), visible='legendonly' 
        ))
        
        normalized_data = sliced_data / sliced_data.iloc[0]
        for col in normalized_data.columns:
            display_name = name_map.get(col, col)
            line_color = get_hex_color(display_name)
            fig.add_trace(go.Scatter(
                x=normalized_data.index, y=normalized_data[col],
                name=f"{display_name}",
                line=dict(width=1, color=line_color), opacity=0.6, visible='legendonly' 
            ))
            
        fig.update_layout(
            height=500, hovermode="x unified", xaxis_title="", 
            yaxis_title="归一化净值 (Start=1.0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # === 恢复的功能：年度/月度回报 ===
        st.markdown("##### 📅 年度盈亏统计")
        res_y = []
        years = df_res.index.year.unique()
        for y in years:
            d_sub = df_res[df_res.index.year == y]
            if d_sub.empty: continue
            y_ret = d_sub['净值'].iloc[-1] / d_sub['净值'].iloc[0] - 1
            # 基准
            b_start = bm_curve.loc[d_sub.index[0]]
            b_end = bm_curve.loc[d_sub.index[-1]]
            b_ret = b_end / b_start - 1
            res_y.append({"年份": y, "策略收益": y_ret, "基准收益": b_ret, "超额(Alpha)": y_ret - b_ret})
        
        if res_y:
            st.dataframe(pd.DataFrame(res_y).set_index("年份").style.format("{:+.2%}").background_gradient(subset=["超额(Alpha)"], cmap="RdYlGn", vmin=-0.2, vmax=0.2), use_container_width=True)

        st.markdown("##### 🗓️ 月度收益矩阵 (Heatmap)")
        # 计算月度收益
        df_nav = df_res['净值'].resample('ME').last() # Pandas 2.0+
        monthly_rets = df_nav.pct_change().fillna(0)
        
        monthly_data = []
        for date, val in monthly_rets.items():
            monthly_data.append({'Year': date.year, 'Month': date.month, 'Return': val})
            
        if monthly_data:
            df_month = pd.DataFrame(monthly_data)
            pivot_month = df_month.pivot(index='Year', columns='Month', values='Return')
            # 补全
            for m in range(1, 13):
                if m not in pivot_month.columns: pivot_month[m] = np.nan
            pivot_month = pivot_month.sort_index(ascending=False).sort_index(axis=1)
            
            fig_m = px.imshow(pivot_month, 
                              labels=dict(x="月份", y="年份", color="收益率"),
                              x=[f"{i}月" for i in range(1, 13)],
                              color_continuous_scale="RdYlGn", 
                              color_continuous_midpoint=0.0,
                              text_auto=".1%")
            fig_m.update_layout(height=400)
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("数据不足以生成月度矩阵")

    with tab3:
        st.markdown("#### 🛠️ 三维参数极限精度扫描")
        st.markdown("通过立体空间观察参数稳定性：**X轴(周期)** / **Y轴(平滑)** / **Z轴(阈值)**。")
        st.info("💡 提示：您选择了极细的步长 (1天)，计算量较大，请耐心等待 30-60秒。")
        
        # [修改]：传入 raw_data 和 mask，而不是 sliced_data
        if st.button("开始极限精度扫描"):
            opt_res = optimize_parameters_3d(raw_data, mask, p_allow_cash, p_min_holding)
            
            best_row = opt_res.loc[opt_res['Score'].idxmax()]
            
            c1, c2 = st.columns(2)
            c1.success(f"👑 最佳参数: Lookback={best_row['Lookback']}, Smooth={best_row['Smooth']}, Th={best_row['Threshold']:.3f}")
            c2.metric("最佳年化收益", f"{best_row['Annual_Ret']:.1%}", f"回撤: {best_row['MaxDD']:.1%}")
            
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=opt_res['Lookback'],
                y=opt_res['Smooth'],
                z=opt_res['Threshold'],
                mode='markers',
                marker=dict(
                    size=opt_res['Score'] * 6 + 1, 
                    color=opt_res['Annual_Ret'],  
                    colorscale='Plasma', 
                    opacity=0.7,
                    colorbar=dict(title="年化收益")
                ),
                hovertemplate =
                '<b>Lookback</b>: %{x}<br>'+
                '<b>Smooth</b>: %{y}<br>'+
                '<b>Threshold</b>: %{z:.3f}<br>'+
                '<b>Return</b>: %{marker.color:.1%}<br>'+
                '<extra></extra>'
            )])
            
            fig_3d.update_layout(
                scene = dict(
                    xaxis_title='Lookback (周期)',
                    yaxis_title='Smooth (平滑)',
                    zaxis_title='Threshold (阈值)'
                ),
                height=650,
                margin=dict(r=0, b=0, l=0, t=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    with tab4:
        # 交易日记 - 2.0 样式
        df_log = pd.DataFrame(daily_details)
        df_log['日期'] = df_log['日期'].dt.strftime('%Y-%m-%d')
        df_log['总资产'] = df_log['总资产'].apply(lambda x: f"<b>¥{x:,.0f}</b>")
        
        st.write(
            df_log.sort_values("日期", ascending=False).to_html(
                columns=["日期", "当前持仓", "操作", "总资产", "市场全景"],
                index=False,
                escape=False, 
                classes="dataframe"
            ),
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
