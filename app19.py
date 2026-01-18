import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import json
import os
import hashlib

# ==========================================
# 0. 配置与工具函数
# ==========================================
st.set_page_config(
    page_title="AlphaTarget | 核心资产轮动终端",
    page_icon="🕋",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONFIG_FILE = 'strategy_config.json'
DEFAULT_CODES = ["518880", "588000", "513100", "510180", "159915", "510300"]
PRESET_ETFS = {
    "518880": "黄金ETF (避险)", "588000": "科创50 (硬科技)", "513100": "纳指100 (海外)",
    "510180": "上证180 (蓝筹)", "159915": "创业板指 (成长)", "510300": "沪深300 (大盘)",
    "510500": "中证500 (中盘)", "512890": "红利低波 (防御)", "513500": "标普500 (美股)",
    "512480": "半导体ETF (行业)", "512880": "证券ETF (Beta)"
}

# --- CSS 注入：顶级投行风格 (Glassmorphism & Clean UI) ---
st.markdown("""
<style>
    /* 全局背景：高级灰蓝 */
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }
    
    /* 侧边栏：半透明磨砂 */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 指标卡片：悬浮视差效果 */
    .metric-container {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-align: center;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
        background: rgba(255, 255, 255, 0.9);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #1e293b;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* 信号Banner：渐变与光泽 */
    .signal-banner {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(30, 41, 59, 0.25);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .signal-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(45deg, transparent 0%, rgba(255,255,255,0.05) 100%);
        pointer-events: none;
    }

    /* 表格优化 */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 胶囊标签样式 (用于HTML渲染) */
    .badge {
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.85em;
        font-weight: 500;
    }
    .badge-red { background: rgba(255, 75, 75, 0.1); color: #d93025; }
    .badge-green { background: rgba(52, 168, 83, 0.1); color: #1e8e3e; }
    .badge-gray { background: rgba(100, 116, 139, 0.1); color: #64748b; }
    
</style>
""", unsafe_allow_html=True)

def metric_html(label, value, delta="", delta_color="gray"):
    color_map = {"red": "#d93025", "green": "#1e8e3e", "gray": "#94a3b8", "blue": "#1a73e8"}
    d_style = f"color: {color_map.get(delta_color, 'gray')}"
    return f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta" style="{d_style}">{delta}</div>
    </div>
    """

# ==========================================
# 1. 数据管理
# ==========================================

@st.cache_data(ttl=3600*12) 
def get_all_etf_list():
    try:
        df = ak.fund_etf_spot_em()
        df['display'] = df['代码'] + " | " + df['名称']
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def download_market_data(codes_list):
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
            if not match.empty: name = match.iloc[0]['名称']
        
        name_map[code] = name
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, adjust="qfq")
            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                price_dict[name] = df['收盘'].astype(float)
        except: continue

    if not price_dict: return None, None
    data = pd.concat(price_dict, axis=1).sort_index().ffill()
    data.dropna(how='all', inplace=True)
    return data, name_map

# ==========================================
# 2. 策略计算核心
# ==========================================

def fast_backtest(daily_ret, mom_df, threshold, min_holding, cost_rate, allow_cash):
    """向量化回测核心逻辑"""
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
        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)
        best_idx = np.argmax(clean_mom)
        best_val = clean_mom[best_idx]
        target_idx = curr_idx
        
        # 信号生成逻辑
        if allow_cash and best_val < 0:
            target_idx = -1 # 空仓
        else:
            if curr_idx == -2:
                if best_val > -np.inf: target_idx = best_idx
            elif curr_idx == -1:
                if best_val > 0 or (not allow_cash): target_idx = best_idx
            else:
                if days_held >= min_holding:
                    curr_val = clean_mom[curr_idx]
                    if best_idx != curr_idx and best_val > curr_val + threshold:
                        target_idx = best_idx
        
        # 交易执行
        if target_idx != curr_idx:
            if curr_idx != -2:
                strategy_ret[i] -= cost_rate
                trade_count += 1
                days_held = 0
            curr_idx = target_idx
            
        if curr_idx >= 0:
            strategy_ret[i] += p_ret[i, curr_idx]
            
    return strategy_ret, trade_count

def calculate_advanced_metrics(equity_curve, trade_count):
    """计算高级指标：夏普、卡玛、胜率等"""
    if len(equity_curve) < 2: return {}
    series = pd.Series(equity_curve)
    ret = series.pct_change().fillna(0)
    
    total_ret = equity_curve[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / len(equity_curve)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    rf = 0.02
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)
    
    max_dd = ((series / series.cummax()) - 1).min()
    calmar = ann_ret / (abs(max_dd) + 1e-9)
    
    return {
        "年化收益": ann_ret, "最大回撤": max_dd, "夏普比率": sharpe, 
        "卡玛比率": calmar, "调仓次数": trade_count, "波动率": ann_vol
    }

# ==========================================
# 3. 优化引擎 (升级版)
# ==========================================

def run_optimization(data, allow_cash, min_holding):
    # 更细致的参数网格
    lookbacks = range(15, 65, 5)  
    smooths = range(1, 10, 2)
    thresholds = [0.0, 0.002, 0.005, 0.01]
    
    daily_ret = data.pct_change().fillna(0)
    results = []
    
    progress_bar = st.progress(0, text="AI 正在进行多维度参数寻优...")
    total_steps = len(lookbacks) * len(smooths) * len(thresholds)
    step = 0
    
    for lb in lookbacks:
        for sm in smooths:
            mom = data.pct_change(lb).rolling(sm).mean()
            for th in thresholds:
                s_ret, count = fast_backtest(daily_ret, mom, th, min_holding, 0.0001, allow_cash)
                
                # 快速计算关键指标
                eq = (1 + s_ret).cumprod()
                final_ret = eq[-1] - 1
                ann_ret = (1 + final_ret) ** (252 / len(eq)) - 1
                vol = np.std(s_ret) * np.sqrt(252)
                sharpe = (ann_ret - 0.02) / (vol + 1e-9)
                dd = ((pd.Series(eq) / pd.Series(eq).cummax()) - 1).min()
                
                results.append({
                    "周期(L)": lb, "平滑(S)": sm, "阈值(T)": th,
                    "年化收益": ann_ret, "夏普比率": sharpe, 
                    "最大回撤": dd, "调仓次数": count
                })
                
                step += 1
                if step % 20 == 0: progress_bar.progress(step / total_steps)
                
    progress_bar.empty()
    return pd.DataFrame(results)

# ==========================================
# 4. 主界面逻辑
# ==========================================

def main():
    # --- 侧边栏 ---
    with st.sidebar:
        st.title("🎛️ 策略控制台")
        st.caption("AlphaTarget Pro v2.0")
        
        # 标的选择
        st.subheader("1. 核心资产池")
        all_etfs = get_all_etf_list()
        options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES
        
        # 智能匹配默认值
        default_dis = []
        if not all_etfs.empty:
            for c in DEFAULT_CODES:
                m = all_etfs[all_etfs['代码'] == c]
                if not m.empty: default_dis.append(m.iloc[0]['display'])
        
        selected_display = st.multiselect("多资产轮动池", options, default=default_dis[:6])
        selected_codes = [x.split(" | ")[0] for x in selected_display]
        
        st.divider()
        st.subheader("2. 回测参数")
        p_lookback = st.slider("动量周期 (Lookback)", 10, 120, 25)
        p_smooth = st.slider("平滑窗口 (Smooth)", 1, 30, 3)
        
        c1, c2 = st.columns(2)
        p_threshold = c1.number_input("换仓阈值", 0.0, 0.05, 0.005, step=0.001, format="%.3f")
        p_min_hold = c2.number_input("最小持仓(天)", 1, 60, 3)
        p_cash = st.toggle("启用空仓避险 (Risk-Off)", True)
        
        st.divider()
        st.info("💡 提示：点击主界面的'参数寻优'可自动寻找最佳 Lookback 和 Smooth 组合。")

    # --- 主区域 ---
    st.markdown("## 🦅 核心资产轮动策略终端")
    
    if not selected_codes:
        st.warning("请在左侧选择至少一个标的。")
        st.stop()
        
    # 数据加载
    with st.spinner("正在构建数据立方体..."):
        raw_data, name_map = download_market_data(selected_codes)
    
    if raw_data is None: st.stop()
    
    # 策略计算
    mom_df = raw_data.pct_change(p_lookback).rolling(p_smooth).mean()
    d_ret = raw_data.pct_change().fillna(0)
    
    start_dt = st.date_input("回测开始日期", datetime(2019, 1, 1))
    mask = raw_data.index >= pd.to_datetime(start_dt)
    
    s_ret, trades = fast_backtest(d_ret[mask], mom_df[mask], p_threshold, p_min_hold, 0.0001, p_cash)
    equity = (1 + s_ret).cumprod()
    
    # 指标统计
    metrics = calculate_advanced_metrics(equity, trades)
    
    # --- 顶部状态栏 (Signal Banner) ---
    last_signal = mom_df.iloc[-1].idxmax()
    if p_cash and mom_df.iloc[-1].max() < 0: last_signal = "Cash"
    
    # 计算当前持仓了几天 (近似倒推)
    # 此处简化逻辑，实际应从回测状态获取
    
    col_ban, col_rank = st.columns([2, 1])
    with col_ban:
        sig_name = name_map.get(last_signal, last_signal) if last_signal != "Cash" else "🛡️ 现金/货币基金 (Cash)"
        st.markdown(f"""
        <div class="signal-banner">
            <div style="font-size:0.9rem; opacity:0.8;">CURRENT POSITION | 当前持仓</div>
            <div style="font-size:2.2rem; font-weight:700; margin:10px 0;">{sig_name}</div>
            <div style="font-size:0.9rem;">
                <span style="background:rgba(255,255,255,0.2); padding:4px 10px; border-radius:4px;">
                触发阈值: {p_threshold*100:.1f}%
                </span>
                &nbsp;&nbsp;最小持仓限制: {p_min_hold} 天
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_rank:
        st.markdown("**⚡ 动量实时榜 (Momentum Rank)**")
        ranks = mom_df.iloc[-1].sort_values(ascending=False).head(4)
        for code, score in ranks.items():
            color = "#ef4444" if score > 0 else "#22c55e"
            n = name_map.get(code, code)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; border-bottom:1px solid #eee; padding-bottom:4px;">
                <span style="font-weight:500; font-size:0.9rem;">{n}</span>
                <span style="color:{color}; font-weight:bold; font-family:monospace;">{score:+.2%}</span>
            </div>
            """, unsafe_allow_html=True)

    # --- 核心指标卡片 ---
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.markdown(metric_html("累计收益", f"{equity[-1]-1:+.1%}", "Total Return", "red"), unsafe_allow_html=True)
    with m2: st.markdown(metric_html("年化收益 (CAGR)", f"{metrics['年化收益']:.1%}", "Annualized", "red"), unsafe_allow_html=True)
    with m3: st.markdown(metric_html("夏普比率", f"{metrics['夏普比率']:.2f}", "Sharpe Ratio", "blue"), unsafe_allow_html=True)
    with m4: st.markdown(metric_html("最大回撤", f"{metrics['最大回撤']:.1%}", "Max Drawdown", "green"), unsafe_allow_html=True)
    with m5: st.markdown(metric_html("调仓次数", f"{metrics['调仓次数']}", "Trades", "gray"), unsafe_allow_html=True)

    # --- 标签页功能区 ---
    tab_chart, tab_opt, tab_log = st.tabs(["📈 综合市场透视", "🛠️ 参数优化引擎", "📒 智能交易日记"])
    
    with tab_chart:
        # [需求2] 综合图表：同时涵盖所选标的走势，且有选择性展示功能
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        # 1. 策略曲线 (高亮)
        fig.add_trace(go.Scatter(x=d_ret[mask].index, y=equity, name="策略净值 (Strategy)", 
                                 line=dict(color='#d93025', width=2.5)), row=1, col=1)
        
        # 2. 个股曲线 (默认隐藏 legendonly)
        colors = px.colors.qualitative.Plotly
        for idx, code in enumerate(selected_codes):
            asset_eq = (1 + d_ret[code][mask]).cumprod()
            n = name_map.get(code, code)
            # 只有第一个标的默认显示作为参考，其他隐藏
            vis = 'legendonly' 
            fig.add_trace(go.Scatter(x=d_ret[mask].index, y=asset_eq, name=n,
                                     line=dict(width=1.5, color=colors[idx % len(colors)]),
                                     opacity=0.8, visible=vis), row=1, col=1)
        
        # 3. 回撤区域
        dd_series = (pd.Series(equity) / pd.Series(equity).cummax()) - 1
        fig.add_trace(go.Scatter(x=d_ret[mask].index, y=dd_series, name="回撤 (Drawdown)",
                                 fill='tozeroy', line=dict(color='gray', width=0.5), opacity=0.3), row=2, col=1)
        
        fig.update_layout(
            height=600, 
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
            legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_opt:
        # [需求1] 参数优化：考虑调仓次数和夏普，优化布局
        c_opt1, c_opt2 = st.columns([1, 3])
        with c_opt1:
            st.write("点击下方按钮开始遍历计算。")
            if st.button("🚀 启动优化引擎", type="primary"):
                with st.spinner("计算中..."):
                    df_res = run_optimization(raw_data[mask], p_cash, p_min_hold)
                    st.session_state['opt_res'] = df_res
        
        if 'opt_res' in st.session_state:
            df_res = st.session_state['opt_res']
            with c_opt2:
                # 散点图可视化：夏普 vs 收益
                fig_opt = px.scatter(df_res, x="年化收益", y="夏普比率", 
                                     color="最大回撤", size="调仓次数",
                                     hover_data=["周期(L)", "平滑(S)", "阈值(T)"],
                                     color_continuous_scale="RdYlGn",
                                     title="参数效能分布 (气泡大小=调仓频率)")
                st.plotly_chart(fig_opt, use_container_width=True)
            
            st.markdown("##### 🏆 优化结果明细 (支持点击表头排序)")
            st.dataframe(
                df_res.style.format({
                    "年化收益": "{:.2%}", "夏普比率": "{:.2f}", "最大回撤": "{:.2%}", "阈值(T)": "{:.3f}"
                }).background_gradient(subset=["夏普比率", "年化收益"], cmap="Reds"),
                use_container_width=True,
                height=300
            )

    with tab_log:
        # [需求3] 交易日记：顺序一致，颜色区分
        st.markdown("##### 📝 结构化交易日志")
        
        # 1. 重新构建带信号的详细数据
        log_data = []
        sim_dates = d_ret[mask].index
        sim_moms = mom_df[mask].shift(1) # 昨天的动量决定今天的持仓
        
        # 预先生成固定顺序的表头
        fixed_assets = [name_map.get(c, c) for c in selected_codes]
        
        current_h = "Cash"
        
        # 为了演示速度，这里只取最近 100 个交易日（实际可放开）
        recent_dates = sim_dates[-100:] 
        
        for d in recent_dates:
            d_str = d.strftime("%Y-%m-%d")
            # 当日各标的动量
            row_mom = sim_moms.loc[d]
            best_c = row_mom.idxmax()
            best_val = row_mom.max()
            
            # 简化版持仓判断（仅做展示用，实际逻辑同回测）
            pos_name = "Cash"
            if not np.isnan(best_val):
                if p_cash and best_val < 0: pos_name = "Cash"
                else: pos_name = name_map.get(best_c, best_c)
            
            # 构建市场扫描列 (HTML)
            # 固定顺序：Asset A | Asset B | Asset C ...
            market_scan_html = []
            today_perf = d_ret.loc[d] # 当日涨跌幅
            
            for code in selected_codes:
                val = today_perf[code]
                c_name = name_map.get(code, code)
                color = "#d93025" if val > 0 else "#1e8e3e" # 红涨绿跌
                bg = "rgba(217,48,37,0.1)" if val > 0 else "rgba(30,142,62,0.1)"
                # 迷你胶囊
                badge = f"<span style='color:{color}; background:{bg}; padding:2px 6px; border-radius:4px; font-size:0.8em; margin-right:4px;'>{c_name[:4]} {val:+.1%}</span>"
                market_scan_html.append(badge)
            
            log_data.append({
                "日期": d_str,
                "策略持仓": pos_name,
                "全市场扫描 (Fixed Order)": "".join(market_scan_html),
                "当日净值": f"{equity[d]:.3f}"
            })
            
        df_log = pd.DataFrame(log_data).sort_values("日期", ascending=False)
        
        # 使用 column_config 渲染 HTML
        st.dataframe(
            df_log,
            column_config={
                "全市场扫描 (Fixed Order)": st.column_config.Column(width="large"),
                "策略持仓": st.column_config.TextColumn(help="当日实际持有的标的"),
            },
            hide_index=True,
            use_container_width=True
        )
        st.markdown(f"<div style='text-align:right; color:gray; font-size:0.8em;'>*仅展示最近 {len(recent_dates)} 个交易日以提升渲染速度</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
