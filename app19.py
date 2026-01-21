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

# 核心标的池映射 (代码 -> 中文名)
# 修复：预设中文名称，确保即使无网络也能显示中文
CODE_NAME_MAP = {
    "518880": "黄金ETF (避险)",
    "588000": "科创50 (硬科技)", 
    "513100": "纳指100 (海外)",
    "510180": "上证180 (蓝筹)",
    "159915": "创业板指 (成长)",
    "510300": "沪深300 (大盘)",
    "510500": "中证500 (中盘)",
    "512890": "红利低波 (防御)",
    "513500": "标普500 (美股)",
    "512480": "半导体ETF (行业)",
    "512880": "证券ETF (Beta)",
    "600519": "贵州茅台 (白酒)",
    "000300": "沪深300指数",
}

DEFAULT_CODES = ["518880", "588000", "513100", "510180"]

DEFAULT_PARAMS = {
    'lookback': 25,
    'smooth': 3,
    'threshold': 0.005,
    'min_holding': 3,
    'allow_cash': True,
    'mom_method': 'Risk-Adjusted (稳健)', 
    'selected_codes': DEFAULT_CODES,
    'multiplier': 100.0,  # 新增：默认合约乘数/每手股数
}

def load_config():
    """从本地文件加载配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
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
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ==========================================
# 1. 页面配置 & CSS样式
# ==========================================
st.set_page_config(
    page_title="AlphaTarget | 核心资产轮动策略终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .metric-card {
        background-color: #ffffff; border: 1px solid #eaeaea; border-radius: 12px;
        padding: 20px 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.04); text-align: center; height: 100%;
    }
    .metric-label { color: #7f8c8d; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { color: #2c3e50; font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
    .metric-sub { font-size: 0.8rem; color: #95a5a6; margin-top: 6px; }
    .signal-banner {
        padding: 25px; border-radius: 12px; margin-bottom: 25px; color: white;
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
    }
    .total-asset-header { font-size: 2.2rem; font-weight: 800; color: #2c3e50; margin-bottom: 0.2rem; }
    .total-asset-sub { font-size: 1.1rem; color: #7f8c8d; font-weight: 500; }
    .opt-highlight { background-color: #e8f4f8; border-left: 4px solid #3498db; padding: 10px; border-radius: 4px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

TRANSACTION_COST = 0.0001  # 万分之一

# 辅助函数：生成指标卡片HTML
def metric_html(label, value, sub="", color="#2c3e50"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """

# ==========================================
# 2. 数据层 (增强版)
# ==========================================

@st.cache_data(ttl=3600*4)
def download_market_data(codes_list, end_date_str):
    """
    纯净历史数据下载，支持 ETF 和 股票 自动切换
    """
    start_str = '20180101' 
    price_dict = {}
    name_map = CODE_NAME_MAP.copy() # 使用预设映射作为基础
    
    # 尝试获取 ETF 列表以更新名称（非必须）
    try:
        etf_list = ak.fund_etf_spot_em()
    except:
        etf_list = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, code in enumerate(codes_list):
        status_text.text(f"正在下载: {code} ...")
        progress_bar.progress((i + 1) / len(codes_list))
        
        # 1. 尝试更新名称
        if code not in name_map:
            if not etf_list.empty:
                match = etf_list[etf_list['代码'] == code]
                if not match.empty:
                    name_map[code] = match.iloc[0]['名称']
            if code not in name_map:
                name_map[code] = code # 兜底

        # 2. 下载数据 (优先尝试 ETF 接口，失败尝试 股票 接口)
        df = pd.DataFrame()
        try:
            # 尝试 ETF
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
        except:
            pass
            
        if df.empty:
            try:
                # 尝试 股票
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
            except:
                pass

        if not df.empty:
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            # 统一列名
            if '收盘' in df.columns:
                price_dict[code] = df['收盘'].astype(float)
            elif '收盘价' in df.columns: # stock_zh_a_hist 返回的是 '收盘'，但也防止变动
                price_dict[code] = df['收盘价'].astype(float)
    
    progress_bar.empty()
    status_text.empty()

    if not price_dict:
        return None, None

    # 合并数据
    data = pd.concat(price_dict, axis=1).sort_index().ffill()
    data.dropna(how='all', inplace=True)
    
    # 至少要有一定数据量
    if len(data) < 20: return None, None
    return data, name_map

# ==========================================
# 3. 策略内核 (Vectorized Engine)
# ==========================================

def calculate_momentum(data, lookback, smooth, method='Classic (普通)'):
    if method == 'Classic (普通)':
        mom = data.pct_change(lookback)
    elif method == 'Risk-Adjusted (稳健)':
        ret = data.pct_change(lookback)
        vol = data.pct_change().rolling(lookback).std()
        mom = ret / (vol + 1e-9)
    elif method == 'MA Distance (趋势)':
        ma = data.rolling(lookback).mean()
        mom = (data / ma) - 1
    else:
        mom = data.pct_change(lookback)

    if smooth > 1:
        mom = mom.rolling(smooth).mean()
        
    return mom

def fast_backtest_vectorized(daily_ret, mom_df, threshold, min_holding=1, cost_rate=0.0001, allow_cash=True):
    # 这里的逻辑保持高效的向量化回测
    signal_mom = mom_df.shift(1)
    n_days, n_assets = daily_ret.shape
    p_ret = daily_ret.values
    p_mom = signal_mom.values
    
    strategy_ret = np.zeros(n_days)
    curr_idx = -2 
    trade_count = 0
    days_held = 0 
    
    # 缓存索引以加速
    cash_idx = -1
    no_pos_idx = -2
    
    for i in range(n_days):
        if curr_idx != no_pos_idx:
            days_held += 1
            
        row_mom = p_mom[i]
        # 如果当天没有动量数据（如上市前），跳过
        if np.isnan(row_mom).all(): continue
            
        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)
        best_idx = np.argmax(clean_mom)
        best_val = clean_mom[best_idx]
        target_idx = curr_idx
        
        # 1. 绝对动量检查 (Cash Check)
        if allow_cash and best_val < 0:
            target_idx = cash_idx
        else:
            # 2. 相对动量检查
            if curr_idx == no_pos_idx:
                if best_val > -np.inf: target_idx = best_idx
            elif curr_idx == cash_idx:
                if best_val > 0 or (not allow_cash): target_idx = best_idx
            else:
                # 持仓中
                if days_held >= min_holding:
                    curr_val = clean_mom[curr_idx]
                    if best_idx != curr_idx:
                        # 换仓阈值检查
                        if best_val > curr_val + threshold:
                            target_idx = best_idx
                    else:
                        target_idx = curr_idx
        
        # 3. 执行换仓
        if target_idx != curr_idx:
            if curr_idx != no_pos_idx:
                strategy_ret[i] -= cost_rate
                trade_count += 1
                days_held = 0
            curr_idx = target_idx
            
        # 4. 计算当日收益
        if curr_idx >= 0:
            strategy_ret[i] += p_ret[i, curr_idx]
            
    equity_curve = (1 + strategy_ret).cumprod()
    total_ret = equity_curve[-1] - 1
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    return total_ret, max_dd, equity_curve, trade_count

# ==========================================
# 4. 分析师工具箱 (指标计算)
# ==========================================

def calculate_pro_metrics(equity_curve, benchmark_curve, trade_count):
    if len(equity_curve) < 2: return {}
    s_eq = pd.Series(equity_curve)
    daily_ret = s_eq.pct_change().fillna(0)
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
    
    beta, alpha = 0.0, 0.0
    if HAS_SCIPY and len(benchmark_curve) == len(equity_curve):
        s_bm = pd.Series(benchmark_curve)
        bm_ret = s_bm.pct_change().fillna(0)
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
    methods = ['Classic (普通)', 'Risk-Adjusted (稳健)', 'MA Distance (趋势)']
    lookbacks = range(20, 31, 2) # 稍微减少步长以加快速度
    smooths = range(1, 6, 1)      
    thresholds = np.arange(0.0, 0.012, 0.002)
    
    daily_ret = data.pct_change().fillna(0)
    n_days = len(daily_ret) 
    results = []
    
    total_iters = len(methods) * len(lookbacks) * len(smooths) * len(thresholds)
    my_bar = st.progress(0, text="正在进行四维全景扫描...")
    
    idx = 0
    for method in methods:
        for lb in lookbacks:
            for sm in smooths:
                mom = calculate_momentum(data, lb, sm, method)
                for th in thresholds:
                    ret, dd, equity, count = fast_backtest_vectorized(
                        daily_ret, mom, th, 
                        min_holding=min_holding,
                        cost_rate=TRANSACTION_COST, 
                        allow_cash=allow_cash
                    )
                    
                    ann_ret = (1 + ret) ** (252 / n_days) - 1
                    if n_days > 1:
                        eq_s = pd.Series(equity)
                        d_r = eq_s.pct_change().fillna(0)
                        ann_vol = d_r.std() * np.sqrt(252)
                        sharpe = (ann_ret - 0.03) / (ann_vol + 1e-9)
                    else:
                        sharpe = 0.0
                    
                    ann_trades = count * (252 / n_days)
                    score = ret / (abs(dd) + 0.05)
                    
                    results.append([method, lb, sm, th, ret, ann_ret, count, ann_trades, dd, sharpe, score])
                    
                    idx += 1
                    if idx % 100 == 0:
                        my_bar.progress(min(idx / total_iters, 1.0))
                    
    my_bar.empty()
    df_res = pd.DataFrame(results, columns=['方法', '周期', '平滑', '阈值', '累计收益', '年化收益', '调仓次数', '年化调仓', '最大回撤', '夏普比率', '得分'])
    return df_res

# ==========================================
# 5. 主程序 UI
# ==========================================

def main():
    if 'params' not in st.session_state:
        saved_config = load_config()
        st.session_state.params = saved_config

    if 'opt_results' not in st.session_state:
        st.session_state.opt_results = None

    with st.sidebar:
        st.title("🎛️ 策略控制台")
        
        # --- 1. 资产与数据 ---
        st.subheader("1. 资产池配置")
        
        # 修复：构建带中文名称的选项列表 (代码 | 名称)
        # 优先使用配置中的 selected_codes
        current_selection_codes = st.session_state.params.get('selected_codes', DEFAULT_CODES)
        
        # 合并默认池和当前选择，去重
        all_possible_codes = list(set(list(CODE_NAME_MAP.keys()) + current_selection_codes))
        
        # 生成显示用的 Label 列表
        options_map = {} # "Code | Name" -> "Code"
        display_options = []
        
        for code in all_possible_codes:
            name = CODE_NAME_MAP.get(code, "未知标的")
            label = f"{code} | {name}"
            options_map[label] = code
            display_options.append(label)
            
        display_options.sort() # 排序
        
        # 找出当前选中的 label
        default_labels = []
        for code in current_selection_codes:
            name = CODE_NAME_MAP.get(code, "未知标的")
            default_labels.append(f"{code} | {name}")
            
        # 核心修复：带搜索功能的 Multiselect
        selected_labels = st.multiselect(
            "核心标的池 (支持搜索中文/代码)", 
            display_options, 
            default=[lbl for lbl in default_labels if lbl in display_options]
        )
        
        # 反向解析出代码
        selected_codes = [options_map[lbl] for lbl in selected_labels]
        
        st.divider()
        st.subheader("2. 资金管理")
        
        date_mode = st.radio("回测区间", ["全历史", "自定义"], index=0)
        
        start_date_input = datetime(2020, 1, 1)
        end_date_input = datetime.now()
        
        if date_mode == "自定义":
            c1, c2 = st.columns(2)
            start_date_input = c1.date_input("Start", datetime(2020, 1, 1))
            end_date_input = c2.date_input("End", datetime.now())

        invest_mode = st.radio("投资模式", ["一次性投入 (Lump Sum)", "定期定额 (SIP)"], index=0)
        
        initial_capital = 100000.0
        sip_amount = 0.0
        sip_freq = "None"
        
        if invest_mode == "一次性投入 (Lump Sum)":
            initial_capital = st.number_input("初始本金 (¥)", value=100000.0, step=10000.0)
        else:
            c1, c2 = st.columns(2)
            initial_capital = c1.number_input("初始底仓", value=10000.0, step=1000.0)
            sip_amount = c2.number_input("定投金额", value=2000.0, step=500.0)
            sip_freq = st.selectbox("定投频率", ["每月 (Monthly)", "每周 (Weekly)"], index=0)

        # 修复：增加合约乘数设置，满足“乘以100”的需求
        p_multiplier = st.number_input(
            "合约乘数/每手股数 (Multiplier)", 
            value=st.session_state.params.get('multiplier', 100.0), 
            min_value=1.0, 
            step=10.0,
            help="用于计算每日盈亏金额。ETF/股票通常为100，期货根据合约设定。"
        )

        st.divider()
        
        # --- 3. 策略参数 ---
        with st.form(key='settings_form'):
            st.subheader("3. 策略内核参数")
            
            mom_options = ['Classic (普通)', 'Risk-Adjusted (稳健)', 'MA Distance (趋势)']
            default_mom = st.session_state.params.get('mom_method', 'Risk-Adjusted (稳健)')
            if default_mom not in mom_options: default_mom = 'Classic (普通)'
            
            p_method = st.selectbox("动量计算逻辑", mom_options, index=mom_options.index(default_mom))
            
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                p_lookback = st.number_input("动量周期", min_value=5, max_value=120, value=st.session_state.params.get('lookback', 25), step=1)
            with c_p2:
                p_smooth = st.number_input("平滑窗口", min_value=1, max_value=60, value=st.session_state.params.get('smooth', 3), step=1)
                
            p_threshold = st.number_input("换仓阈值", 0.0, 0.05, st.session_state.params.get('threshold', 0.005), step=0.001, format="%.3f")
            
            st.markdown("---")
            st.markdown("**🛑 风控参数**")
            p_min_holding = st.number_input("最小持仓天数", min_value=1, max_value=60, value=st.session_state.params.get('min_holding', 3), step=1)
            p_allow_cash = st.checkbox("启用绝对动量避险 (Cash Protection)", value=st.session_state.params.get('allow_cash', True))
            
            submit_btn = st.form_submit_button("🚀 确认并运行 (Run Analysis)")

        if submit_btn:
            current_params = {
                'lookback': p_lookback, 'smooth': p_smooth, 'threshold': p_threshold,
                'min_holding': p_min_holding, 'allow_cash': p_allow_cash, 'selected_codes': selected_codes,
                'mom_method': p_method, 'multiplier': p_multiplier
            }
            if current_params != st.session_state.params:
                st.session_state.params = current_params
                save_config(current_params)
        
        if st.button("🔄 重置默认配置"):
            st.session_state.params = DEFAULT_PARAMS.copy()
            save_config(DEFAULT_PARAMS)
            st.rerun()

    # 日期逻辑处理，修复 datetime 兼容性
    def to_datetime(d):
        if isinstance(d, datetime): return d
        return datetime.combine(d, datetime.min.time())

    start_date = to_datetime(start_date_input)
    end_date = to_datetime(end_date_input)

    st.markdown("## 🚀 核心资产轮动策略终端 (Pro Ver.)")
    
    if not selected_codes:
        st.warning("请在侧边栏选择标的。")
        st.stop()
        
    with st.spinner("正在加载历史行情数据 (支持 ETF 和 股票)..."):
        raw_data, name_map = download_market_data(selected_codes, end_date.strftime('%Y%m%d'))
        
    if raw_data is None:
        st.error("数据不足或下载失败。请检查代码是否正确。")
        st.stop()

    daily_ret_all = raw_data.pct_change().fillna(0)
    mom_method_curr = st.session_state.params.get('mom_method', 'Classic (普通)')
    mom_all = calculate_momentum(raw_data, p_lookback, p_smooth, mom_method_curr)
    
    mask = (raw_data.index >= start_date) & (raw_data.index <= end_date)
    sliced_data = raw_data.loc[mask]
    sliced_mom = mom_all.loc[mask] 
    sliced_ret = daily_ret_all.loc[mask]
    
    if sliced_data.empty:
        st.error("所选区间内无数据，请调整日期。")
        st.stop()

    signal_mom = sliced_mom.shift(1)
    dates = sliced_ret.index
    
    # === 详细回测循环 (用于生成日志) ===
    cash = initial_capital
    share_val = 0.0
    curr_hold = None
    days_held = 0
    current_hold_start_val = 0.0 
    
    holdings_history = []
    total_assets_curve = []
    total_invested_curve = []
    total_invested = initial_capital
    trade_count_real = 0
    daily_details = [] 
    last_sip_date = dates[0]
    
    for i, date in enumerate(dates):
        r_today = sliced_ret.loc[date]
        
        # A. 定投
        if invest_mode == "定期定额 (SIP)" and i > 0:
            is_sip_day = False
            if sip_freq.startswith("每月"):
                if date.month != last_sip_date.month: is_sip_day = True
            elif sip_freq.startswith("每周"):
                if date.weekday() == 0 and last_sip_date.weekday() != 0: is_sip_day = True
            
            if is_sip_day:
                cash += sip_amount
                total_invested += sip_amount
                last_sip_date = date

        # B. 信号生成
        if curr_hold is not None: days_held += 1
        row = signal_mom.loc[date]
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
                            if best_score > curr_score + p_threshold: target = best_asset
                    else:
                        target = curr_hold

        day_return = 0.0
        if curr_hold and curr_hold != 'Cash' and curr_hold in r_today:
            day_return = r_today[curr_hold]
        
        share_val = share_val * (1 + day_return)
        
        temp_segment_ret = 0.0
        if curr_hold and curr_hold != 'Cash' and current_hold_start_val > 0:
            temp_segment_ret = (share_val / current_hold_start_val) - 1
            
        log_hold = curr_hold
        log_days = days_held
        log_ret = temp_segment_ret
        note = ""

        # C. 交易执行
        if target != curr_hold:
            if curr_hold is not None:
                total_equity = share_val + cash
                cost = total_equity * TRANSACTION_COST
                if cash >= cost: cash -= cost
                else: share_val -= cost
                trade_count_real += 1
                days_held = 0
                
                old_name = name_map.get(curr_hold, curr_hold) if curr_hold else "Cash"
                new_name = name_map.get(target, target) if target else "Cash"
                note = f"调仓: {old_name} -> {new_name}"
                
            if target == 'Cash':
                cash += share_val
                share_val = 0.0
            else:
                total = share_val + cash
                share_val = total
                cash = 0.0
                current_hold_start_val = total
                
            curr_hold = target
            
        holdings_history.append(target if target else "Cash")
        current_total = share_val + cash
        total_assets_curve.append(current_total)
        total_invested_curve.append(total_invested)
        
        hold_name_display = name_map.get(log_hold, log_hold) if log_hold and log_hold != 'Cash' else 'Cash'
        
        # 修复：计算当日盈亏 (金额)
        prev_total = total_assets_curve[-2] if len(total_assets_curve) > 1 else initial_capital
        daily_pnl = current_total - prev_total
        
        daily_record = {
            "日期": date.strftime('%Y-%m-%d'),
            "当前持仓": hold_name_display,
            "持仓天数": log_days if log_hold != 'Cash' else 0,
            # 修复：收益率乘以 100 以百分比显示
            "段内收益(%)": (log_ret * 100) if log_hold != 'Cash' else 0.0,
            "当日盈亏(¥)": daily_pnl, # 修复：新增当日盈亏
            "操作": note,
            "总资产": current_total,
        }
        
        for code, val in r_today.items():
            col_name = name_map.get(code, code)
            # 这里的原始涨跌幅也乘以100方便阅读
            daily_record[col_name] = val * 100 
            
        daily_details.append(daily_record)

    df_res = pd.DataFrame({
        '总资产': total_assets_curve,
        '投入本金': total_invested_curve,
        '持仓': holdings_history
    }, index=dates)
    
    # 使用向量引擎计算净值曲线 (用于图表，速度更快)
    _, _, nav_series, _ = fast_backtest_vectorized(
        sliced_ret, sliced_mom, p_threshold, 
        min_holding=p_min_holding, cost_rate=TRANSACTION_COST, allow_cash=p_allow_cash
    )
    df_res['策略净值'] = nav_series
    bm_curve = (1 + sliced_ret.mean(axis=1)).cumprod()
    
    # 信号栏
    latest_mom = mom_all.iloc[-1].dropna().sort_values(ascending=False)
    last_hold = holdings_history[-1]
    
    col_sig1, col_sig2 = st.columns([2, 1])
    with col_sig1:
        hold_name = name_map.get(last_hold, last_hold) if last_hold != 'Cash' else '🛡️ 空仓避险 (Cash)'
        lock_msg = f"(已持仓 {days_held} 天)" if last_hold != 'Cash' else ""
        if days_held < p_min_holding and last_hold != 'Cash': lock_msg += " 🔒 **锁定中**"
        
        data_last_date = raw_data.index[-1].strftime('%Y-%m-%d')
        
        st.markdown(f"""
        <div class="signal-banner">
            <h3 style="margin:0">📌 当前持仓: {hold_name}</h3>
            <div style="margin-top:5px; font-size: 0.9rem">
                逻辑: {mom_method_curr} | 最小持仓: {p_min_holding} 天 {lock_msg} | 数据截止: {data_last_date}
            </div>
        </div>""", unsafe_allow_html=True)
        
    with col_sig2:
        st.markdown("**🏆 实时排名 (Top 3)**")
        for i, (asset, score) in enumerate(latest_mom.head(3).items()):
            display_name = name_map.get(asset, asset)
            st.markdown(f"{i+1}. **{display_name}**: `{score*100:.2f}`") # 分数也乘100好看

    # === 优化引擎 (4D) ===
    with st.expander("🛠️ 策略参数优化引擎 (4D Smart Optimizer)", expanded=False):
        opt_source = st.radio(
            "优化数据源 (Data Source for Optimization)", 
            ["当前选定区间 (Selected Range)", "全历史数据 (Full History)"],
            index=0,
            horizontal=True
        )
        
        if st.button("运行全参数扫描 (Method/L/S/T)"):
            data_to_opt = sliced_data if opt_source.startswith("当前") else raw_data
            with st.spinner(f"正在基于 [{opt_source}] 进行四维全景扫描 (约 2000+ 次回测)..."):
                opt_df = optimize_parameters(data_to_opt, p_allow_cash, p_min_holding)
                st.session_state.opt_results = opt_df 
        
        if st.session_state.opt_results is not None:
            opt_df = st.session_state.opt_results
            
            best_ret_idx = opt_df['累计收益'].idxmax()
            best_r = opt_df.loc[best_ret_idx]
            
            best_sharpe_idx = opt_df['夏普比率'].idxmax()
            best_s = opt_df.loc[best_sharpe_idx]
            
            def apply_params(row_data):
                new_params = st.session_state.params.copy()
                new_params['lookback'] = int(row_data['周期'])
                new_params['smooth'] = int(row_data['平滑'])
                new_params['threshold'] = float(row_data['阈值'])
                new_params['mom_method'] = row_data['方法']
                st.session_state.params = new_params
                save_config(new_params)
                st.toast("✅ 参数已更新并保存！正在重新回测...", icon="💾")
                time.sleep(1)
                st.rerun()

            c1, c2 = st.columns(2)
            # 简写 helper
            def short_method(m): return m.split(" ")[0]

            with c1:
                st.markdown(f'<div class="opt-highlight">🔥 <b>收益优先</b></div>', unsafe_allow_html=True)
                p_str = f"{short_method(best_r['方法'])}/L{int(best_r['周期'])}/S{int(best_r['平滑'])}/T{best_r['阈值']:.3f}"
                st.write(f"**年化:** `{best_r['年化收益']:.1%}`")
                st.write(f"**夏普:** `{best_r['夏普比率']:.2f}`")
                st.caption(f"配置: {p_str}")
                if st.button("💾 应用 (收益)", key="btn_apply_ret"):
                    apply_params(best_r)

            with c2:
                st.markdown(f'<div class="opt-highlight">💎 <b>夏普优先</b></div>', unsafe_allow_html=True)
                p_str_s = f"{short_method(best_s['方法'])}/L{int(best_s['周期'])}/S{int(best_s['平滑'])}/T{best_s['阈值']:.3f}"
                st.write(f"**年化:** `{best_s['年化收益']:.1%}`")
                st.write(f"**夏普:** `{best_s['夏普比率']:.2f}`")
                st.caption(f"配置: {p_str_s}")
                if st.button("💾 应用 (夏普)", key="btn_apply_sharpe"):
                    apply_params(best_s)

    # 报表
    account_ret = df_res['总资产'].iloc[-1] / df_res['投入本金'].iloc[-1] - 1
    account_profit = df_res['总资产'].iloc[-1] - df_res['投入本金'].iloc[-1]
    metrics = calculate_pro_metrics(df_res['策略净值'].values, bm_curve.values, trade_count_real)
    
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div class="total-asset-header">¥{df_res['总资产'].iloc[-1]:,.0f}</div>
        <div class="total-asset-sub">投入本金: ¥{df_res['投入本金'].iloc[-1]:,.0f} | <span style="color: {'#d62728' if account_profit > 0 else 'green'}">总盈亏: {account_profit:+,.0f} ({account_ret:+.2%})</span></div>
    </div>""", unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.markdown(metric_html("累计收益", f"{metrics.get('Total Return',0):.1%}", "", "#c0392b"), unsafe_allow_html=True)
    with m2: st.markdown(metric_html("年化收益", f"{metrics.get('CAGR',0):.1%}", "", "#c0392b"), unsafe_allow_html=True)
    with m3: st.markdown(metric_html("最大回撤", f"{metrics.get('Max Drawdown',0):.1%}", "", "#27ae60"), unsafe_allow_html=True)
    with m4: st.markdown(metric_html("夏普比率", f"{metrics.get('Sharpe Ratio',0):.2f}", "", "#2c3e50"), unsafe_allow_html=True)
    with m5: st.markdown(metric_html("交易次数", f"{trade_count_real}", "", "#2c3e50"), unsafe_allow_html=True)
    with m6: st.markdown(metric_html("合约/每手", f"{int(p_multiplier)}", "Multiplier", "#95a5a6"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 综合图表", "📅 年度回报", "📝 交易日记 (优化)"])
    
    # ---------------- TAB 1: 综合图表 ----------------
    with tab1:
        st.caption("📉 标的走势叠加 (Asset Overlays)")
        all_assets = sliced_data.columns.tolist()
        
        # 转换 options 显示名称
        overlay_options = [f"{c} | {name_map.get(c,c)}" for c in all_assets]
        overlay_map = {f"{c} | {name_map.get(c,c)}": c for c in all_assets}
        
        overlay_selected = st.multiselect("选择要对比的底层资产", overlay_options, default=[])
        overlay_codes = [overlay_map[o] for o in overlay_selected]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.update_layout(template="plotly_white", height=600)
        
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['策略净值'], name="策略净值", line=dict(color='#c0392b', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_res.index, y=bm_curve, name="基准(等权)", line=dict(color='#95a5a6', dash='dash', width=1.5)), row=1, col=1)
        
        for code in overlay_codes:
            # 归一化
            series = sliced_data[code]
            normalized = series / series.iloc[0]
            fig.add_trace(go.Scatter(x=series.index, y=normalized, name=name_map.get(code, code), opacity=0.5), row=1, col=1)

        # 区域图
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['总资产'], name="总资产", fill='tozeroy', line=dict(color='#2980b9')), row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 2: 年度回报 ----------------
    with tab2:
        df_ret = pd.DataFrame(df_res['策略净值'])
        df_ret['Year'] = df_ret.index.year
        yearly_ret = df_ret.groupby('Year')['策略净值'].apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
        
        fig_bar = px.bar(x=yearly_ret.index, y=yearly_ret.values, 
                        title="年度收益率", labels={'x':'年份', 'y':'收益率'},
                        text_auto='.1%', color=yearly_ret.values, color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- TAB 3: 交易日记 (修复版) ----------------
    with tab3:
        st.markdown("### 📋 每日持仓详情 (Daily Log)")
        df_log = pd.DataFrame(daily_details)
        if not df_log.empty:
            # 格式化显示
            cols = ["日期", "当前持仓", "总资产", "当日盈亏(¥)", "段内收益(%)", "持仓天数", "操作"]
            # 把剩余的列（具体标的涨跌幅）加到后面
            extra_cols = [c for c in df_log.columns if c not in cols]
            final_cols = cols + extra_cols
            
            st.dataframe(
                df_log[final_cols].style.format({
                    "总资产": "{:,.2f}",
                    "当日盈亏(¥)": "{:+,.2f}",
                    "段内收益(%)": "{:+.2f}%",
                    **{c: "{:+.2f}%" for c in extra_cols} # 其他涨跌幅列也格式化为 %
                }).background_gradient(subset=["当日盈亏(¥)"], cmap="RdYlGn"),
                use_container_width=True,
                height=500
            )
        else:
            st.info("暂无交易记录")

if __name__ == "__main__":
    main()
