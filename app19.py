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
    .stApp { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    
    /* 指标卡片 */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #eaeaea;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        text-align: center;
        height: 100%;
    }
    .metric-label { color: #7f8c8d; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin-bottom: 5px; }
    .metric-value { color: #2c3e50; font-size: 1.5rem; font-weight: 700; }
    .metric-sub { font-size: 0.75rem; color: #95a5a6; margin-top: 4px; }

    /* 信号横幅 */
    .signal-banner {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        box-shadow: 0 4px 10px rgba(44, 62, 80, 0.2);
    }
    
    /* 交易日记标签样式 */
    .asset-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 500;
        margin-right: 5px;
        margin-bottom: 2px;
        color: #333;
        border: 1px solid rgba(0,0,0,0.05);
    }
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
    # 此处保持原有逻辑，为节省篇幅略去重复注释，逻辑与之前一致
    signal_mom = mom_df.shift(1)
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

def optimize_parameters_3d(data, allow_cash, min_holding):
    # === 三维全参数扫描 ===
    lookbacks = range(15, 60, 5)        # 维度1
    smooths = range(1, 10, 2)           # 维度2
    thresholds = np.arange(0.0, 0.012, 0.002) # 维度3
    
    daily_ret = data.pct_change().fillna(0)
    n_days = len(daily_ret)
    results = []
    
    total_iters = len(lookbacks) * len(smooths) * len(thresholds)
    my_bar = st.progress(0, text=f"正在进行三维空间参数扫描 (0/{total_iters})...")
    
    count = 0
    for lb in lookbacks:
        for sm in smooths:
            mom = calculate_momentum(data, lb, sm)
            for th in thresholds:
                ret, dd, _, trades = fast_backtest_vectorized(
                    daily_ret, mom, th, 
                    min_holding=min_holding, cost_rate=TRANSACTION_COST, allow_cash=allow_cash
                )
                # 简单打分：收益 / (|最大回撤| + 0.1)
                score = ret / (abs(dd) + 0.1)
                results.append([lb, sm, th, ret, trades, dd, score])
                count += 1
                if count % 20 == 0:
                    my_bar.progress(min(count / total_iters, 1.0))
                    
    my_bar.empty()
    df_res = pd.DataFrame(results, columns=['Lookback', 'Smooth', 'Threshold', 'Return', 'Trades', 'MaxDD', 'Score'])
    df_res['Annual_Ret'] = (1 + df_res['Return']) ** (252 / n_days) - 1
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
        
        # 1. 标的选择 (放在Form外面，因为需要动态交互)
        st.subheader("1. 标的池配置")
        all_etfs = get_all_etf_list()
        etf_options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES
        
        # 恢复上次选择
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

        # 2. 参数表单 (Form: 只有点击提交才运行)
        with st.form("strategy_form"):
            st.divider()
            st.subheader("2. 资金与时间")
            
            # 时间设置：固定起始 2021-01-01
            c_d1, c_d2 = st.columns(2)
            # 默认从2021-01-01开始
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
            # 提交按钮
            submitted = st.form_submit_button("🚀 确认修改并运行", type="primary")

    # --- 逻辑处理 ---
    # 如果是第一次加载，或者点击了提交按钮，则更新参数并运行
    if submitted or 'run_once' not in st.session_state:
        st.session_state.run_once = True
        current_params = {
            'lookback': p_lookback, 'smooth': p_smooth, 'threshold': p_threshold,
            'min_holding': p_min_holding, 'allow_cash': p_allow_cash, 'selected_codes': selected_codes_final
        }
        st.session_state.params = current_params
        save_config(current_params)
    
    # 获取当前生效的参数
    run_codes = st.session_state.params['selected_codes']
    
    # 页面主体
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

    # 数据切片
    daily_ret_all = raw_data.pct_change().fillna(0)
    # 使用当前参数计算动量
    mom_all = calculate_momentum(raw_data, p_lookback, p_smooth)
    
    mask = (raw_data.index >= start_date_ts) & (raw_data.index <= end_date_ts)
    sliced_data = raw_data.loc[mask]
    
    if sliced_data.empty:
        st.error(f"所选时间段 {p_start_date} 至 {p_end_date} 无数据，请调整时间。")
        st.stop()
        
    sliced_mom = mom_all.loc[mask]
    sliced_ret = daily_ret_all.loc[mask]
    
    # === 策略回测逻辑 (无SIP) ===
    signal_mom = sliced_mom.shift(1)
    dates = sliced_ret.index
    
    cash = p_initial_capital
    share_val = 0.0
    curr_hold = None
    days_held = 0
    holdings_history = []
    total_assets_curve = []
    daily_details = []
    
    # 生成按顺序的列名列表（用于日记固定顺序显示）
    # 优先使用用户选择的顺序
    ordered_names = [name_map.get(c, c) for c in run_codes if c in name_map]
    
    def format_market_perf_html(row, ordered_keys, name_mapping):
        html_parts = []
        for name in ordered_keys:
            if name in row.index:
                val = row[name]
                color_bg = get_color_from_name(name, alpha=0.15)
                # 涨跌幅颜色
                val_color = "#d62728" if val > 0 else "#2ca02c"
                html = f"""
                <span class="asset-tag" style="background-color:{color_bg};">
                    {name} <span style="color:{val_color};font-weight:bold;">{val:+.2%}</span>
                </span>
                """
                html_parts.append(html)
        return "".join(html_parts)

    for i, date in enumerate(dates):
        # 1. 市场表现 HTML 生成
        r_today = sliced_ret.loc[date]
        market_perf_html = format_market_perf_html(r_today, ordered_names, name_map)
        
        # 2. 策略逻辑
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

        # 3. 净值更新
        day_return = 0.0
        if curr_hold and curr_hold != 'Cash' and curr_hold in r_today:
            day_return = r_today[curr_hold]
            
        share_val = share_val * (1 + day_return)
        
        # 4. 调仓执行
        note = ""
        if target != curr_hold:
            if curr_hold is not None:
                # 卖出成本
                total_equity = share_val + cash
                cost = total_equity * TRANSACTION_COST
                if cash >= cost: cash -= cost
                else: share_val -= cost
                days_held = 0
                
                old = name_map.get(curr_hold, curr_hold) if curr_hold else "Cash"
                new = name_map.get(target, target) if target else "Cash"
                note = f"🔄 {old} -> {new}"
            
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
        
        # 记录日记
        display_hold = name_map.get(curr_hold, curr_hold) if curr_hold and curr_hold != 'Cash' else 'Cash'
        daily_details.append({
            "日期": date,
            "当前持仓": display_hold,
            "日收益": day_return, # 仅持仓资产的当日涨跌
            "总资产": current_total,
            "操作": note,
            "市场全景": market_perf_html
        })

    # === 结果整合 ===
    df_res = pd.DataFrame({
        '总资产': total_assets_curve,
        '持仓': holdings_history
    }, index=dates)
    
    df_res['净值'] = df_res['总资产'] / p_initial_capital
    bm_curve = (1 + sliced_ret.mean(axis=1)).cumprod() # 等权基准
    
    # 统计指标
    total_ret = df_res['净值'].iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / len(dates)) - 1
    max_dd = ((df_res['净值'] - df_res['净值'].cummax()) / df_res['净值'].cummax()).min()
    
    # === UI 展示 ===
    
    # 信号横幅
    last_h = holdings_history[-1]
    h_name = name_map.get(last_h, last_h) if last_h != 'Cash' else '🛡️ 空仓 (Cash)'
    
    col_sig, col_kpi = st.columns([1, 2])
    with col_sig:
        st.markdown(f"""
        <div class="signal-banner">
            <h3 style="margin:0">当前持仓: {h_name}</h3>
            <p style="margin:5px 0 0 0; opacity:0.9">连续持仓: {days_held} 天</p>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(metric_html("总收益率", f"{total_ret:+.1%}", "Total Return", "#d62728"), unsafe_allow_html=True)
        k2.markdown(metric_html("年化收益", f"{ann_ret:+.1%}", "CAGR", "#d62728"), unsafe_allow_html=True)
        k3.markdown(metric_html("最大回撤", f"{max_dd:.1%}", "Max Drawdown", "#2ca02c"), unsafe_allow_html=True)
        k4.markdown(metric_html("当前资产", f"¥{current_total:,.0f}", "Asset", "#2c3e50"), unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 综合走势对比", "🛠️ 3D参数优化引擎", "📝 交易日记"])
    
    with tab1:
        st.markdown("##### 策略 vs 基准 vs 标的走势 (归一化对比)")
        fig = go.Figure()
        
        # 1. 策略曲线 (加粗)
        fig.add_trace(go.Scatter(
            x=df_res.index, y=df_res['净值'], 
            name="🤖 策略净值", 
            line=dict(color='#d62728', width=3),
            mode='lines'
        ))
        
        # 2. 基准曲线
        fig.add_trace(go.Scatter(
            x=bm_curve.index, y=bm_curve, 
            name="📊 等权基准", 
            line=dict(color='gray', width=2, dash='dash'),
            visible='legendonly' # 默认隐藏，点击显示
        ))
        
        # 3. 所有标的曲线 (归一化)
        # 将起点设为1以便比较
        normalized_data = sliced_data / sliced_data.iloc[0]
        
        for col in normalized_data.columns:
            display_name = name_map.get(col, col)
            line_color = get_hex_color(display_name)
            fig.add_trace(go.Scatter(
                x=normalized_data.index, y=normalized_data[col],
                name=f"{display_name}",
                line=dict(width=1, color=line_color),
                opacity=0.6,
                visible='legendonly' # 默认隐藏，不喧宾夺主，用户自己点
            ))
            
        fig.update_layout(
            height=500, 
            hovermode="x unified",
            xaxis_title="", 
            yaxis_title="归一化净值 (Start=1.0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.markdown("#### 🛠️ 三维参数全景扫描 (Lookback x Smooth x Threshold)")
        st.info("提示：点击下方按钮开始遍历。点越大/颜色越深代表得分越高。图表可拖动旋转。")
        
        if st.button("开始3D参数寻优"):
            opt_res = optimize_parameters_3d(sliced_data, p_allow_cash, p_min_holding)
            
            # 找到最佳
            best_row = opt_res.loc[opt_res['Score'].idxmax()]
            
            c1, c2 = st.columns(2)
            c1.success(f"最佳参数组合: Lookback={best_row['Lookback']}, Smooth={best_row['Smooth']}, Th={best_row['Threshold']:.3f}")
            c2.metric("最佳年化收益", f"{best_row['Annual_Ret']:.1%}")
            
            # 3D 散点图
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=opt_res['Lookback'],
                y=opt_res['Smooth'],
                z=opt_res['Threshold'],
                mode='markers',
                marker=dict(
                    size=opt_res['Score'] * 5 + 2, # 分数越高点越大
                    color=opt_res['Annual_Ret'],   # 颜色代表收益率
                    colorscale='Viridis',
                    opacity=0.8,
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
                height=600,
                margin=dict(r=0, b=0, l=0, t=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    with tab3:
        # 交易日记
        df_log = pd.DataFrame(daily_details)
        df_log['日期'] = df_log['日期'].dt.strftime('%Y-%m-%d')
        
        # HTML 渲染
        st.write(
            df_log.sort_values("日期", ascending=False).to_html(
                columns=["日期", "当前持仓", "操作", "总资产", "市场全景"],
                index=False,
                escape=False, # 允许HTML渲染
                classes="dataframe"
            ),
            unsafe_allow_html=True
        )
        st.markdown("""
        <style>
        table.dataframe { width: 100%; text-align: left; border-collapse: collapse; }
        table.dataframe th { background-color: #f0f2f6; padding: 10px; font-size: 14px; }
        table.dataframe td { padding: 8px; border-bottom: 1px solid #eee; font-size: 13px; vertical-align: middle; }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
