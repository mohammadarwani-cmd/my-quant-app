import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==========================================
# 1. 投行级页面配置 & CSS样式
# ==========================================
st.set_page_config(
    page_title="AlphaTarget | 核心资产轮动策略终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入专业金融终端风格CSS
st.markdown("""
<style>
    /* 全局字体与背景 */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Roboto', 'Helvetica Neue', sans-serif;
    }
    
    /* 关键指标卡片 */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #212529;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #adb5bd;
    }
    
    /* 信号横幅 */
    .signal-banner {
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: white;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.2);
    }
    
    /* 优化报告容器 */
    .opt-container {
        border: 1px solid #d1d9e6;
        background-color: #fcfcfc;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 默认标的池 (按用户要求更新)
DEFAULT_CODES = ["518880", "588000", "513100", "510180"]

# 预置ETF映射表 (代码 -> 名称)
PRESET_ETFS = {
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
    "512880": "证券ETF (Beta)"
}

COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

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
def download_market_data(codes_list):
    """
    下载并清洗数据，处理不同起点的对齐问题
    """
    now = datetime.now()
    if now.hour >= 15:
        target_date = now
    else:
        target_date = now - timedelta(days=1)
        
    start_str = '20150101' # 稍微放宽数据起点以涵盖更多周期
    end_str = target_date.strftime('%Y%m%d')
    
    price_dict = {}
    name_map = {}
    
    # 获取名称映射
    etf_list = get_all_etf_list()
    
    for code in codes_list:
        # 尝试从预置或在线列表获取名称
        name = code
        if code in PRESET_ETFS:
            name = PRESET_ETFS[code].split(" ")[0]
        elif not etf_list.empty:
            match = etf_list[etf_list['代码'] == code]
            if not match.empty:
                name = match.iloc[0]['名称']
        
        name_map[code] = name
        
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                price_dict[name] = df['收盘'].astype(float)
        except Exception as e:
            st.error(f"代码 {code} 数据获取失败: {str(e)}")
            continue

    if not price_dict:
        return None, None

    # 对齐数据，向前填充 (FFill) 处理停牌，丢弃全空行
    data = pd.concat(price_dict, axis=1).sort_index().ffill()
    data.dropna(how='all', inplace=True)
    
    # 再次清洗，确保至少有一定长度的数据
    if len(data) < 20:
        return None, None
        
    return data, name_map

# ==========================================
# 3. 策略内核 (Strategy Core)
# ==========================================

def calculate_momentum(data, lookback, smooth):
    """
    计算动量因子
    逻辑：ROC (Rate of Change) + MA平滑
    """
    mom = data.pct_change(lookback)
    if smooth > 1:
        mom = mom.rolling(smooth).mean()
    return mom

def fast_backtest(daily_ret, mom_df, threshold):
    """
    向量化回测加速版 (用于参数遍历)
    """
    # 信号生成: 昨天收盘后的动量决定今天的持仓
    # shift(1) 代表用T-1的数据在T日交易
    signal_mom = mom_df.shift(1)
    
    n_days, n_assets = daily_ret.shape
    p_ret = daily_ret.values
    p_mom = signal_mom.values
    
    # 初始化
    strategy_ret = np.zeros(n_days)
    curr_idx = -1 # -1表示空仓
    
    # 遍历每一天 (由于路径依赖，难以完全向量化，使用Numba或Cython会更快，这里用原生Python优化循环)
    # 为了性能，这里简化逻辑：仅计算每日收益率，不记录详细持仓
    
    for i in range(n_days):
        row_mom = p_mom[i]
        
        # 检查是否全为NaN (比如刚开始几天)
        if np.isnan(row_mom).all():
            continue
            
        # 找到动量最大的索引
        # 处理NaN: 将NaN设为负无穷，避免选中
        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)
        best_idx = np.argmax(clean_mom)
        best_val = clean_mom[best_idx]
        
        # 如果当前无持仓，直接买入第一名
        if curr_idx == -1:
            if best_val > -np.inf: # 确保有效
                curr_idx = best_idx
        else:
            # 换仓判定
            curr_val = clean_mom[curr_idx]
            # 只有当 新的最佳得分 > 当前持仓得分 + 阈值 时才换仓
            if best_idx != curr_idx:
                if best_val > curr_val + threshold:
                    curr_idx = best_idx
        
        if curr_idx != -1:
            strategy_ret[i] = p_ret[i, curr_idx]
            
    # 计算累计净值
    equity_curve = (1 + strategy_ret).cumprod()
    
    # 计算核心指标
    total_ret = equity_curve[-1] - 1
    
    # 最大回撤
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    
    return total_ret, max_dd, equity_curve

# ==========================================
# 4. 分析师工具箱 (Analyst Toolkit)
# ==========================================

def calculate_pro_metrics(equity_curve):
    """
    计算投行级策略指标
    """
    if len(equity_curve) < 2:
        return {}
        
    # 日收益率
    daily_ret = pd.Series(equity_curve).pct_change().fillna(0)
    
    # 1. 基础收益
    total_ret = equity_curve[-1] - 1
    
    # 2. 年化收益 (假设252个交易日)
    days = len(equity_curve)
    ann_ret = (1 + total_ret) ** (252 / days) - 1
    
    # 3. 年化波动率
    ann_vol = daily_ret.std() * np.sqrt(252)
    
    # 4. 夏普比率 (无风险利率设为3%)
    rf = 0.03
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)
    
    # 5. 最大回撤
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    
    # 6. 卡玛比率 (收益回撤比)
    calmar = ann_ret / (abs(max_dd) + 1e-9)
    
    return {
        "Total Return": total_ret,
        "CAGR": ann_ret,
        "Volatility": ann_vol,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar
    }

def optimize_parameters(data):
    """
    参数网格搜索引擎
    """
    # 缩小搜索范围以保证响应速度，但足够捕捉特征
    lookbacks = range(10, 35, 2) # 10到34，步长2
    smooths = [1, 3, 5, 8]
    thresholds = [0.0, 0.005, 0.01]
    
    daily_ret = data.pct_change().fillna(0)
    results = []
    
    total_iters = len(lookbacks) * len(smooths) * len(thresholds)
    progress_bar = st.progress(0)
    idx = 0
    
    for lb in lookbacks:
        for sm in smooths:
            # 预计算Momentum
            mom = calculate_momentum(data, lb, sm)
            for th in thresholds:
                ret, dd, _ = fast_backtest(daily_ret, mom, th)
                # 分数逻辑：卡玛比率权重高
                score = ret / (abs(dd) + 0.01) 
                results.append([lb, sm, th, ret, dd, score])
                
                idx += 1
                if idx % 10 == 0:
                    progress_bar.progress(idx / total_iters)
                    
    progress_bar.empty()
    df_res = pd.DataFrame(results, columns=['周期', '平滑', '阈值', '累计收益', '最大回撤', '得分'])
    return df_res

# ==========================================
# 5. 主程序 UI
# ==========================================

def main():
    # Session State 初始化
    if 'params' not in st.session_state:
        st.session_state.params = {'lookback': 20, 'smooth': 3, 'threshold': 0.005}

    # --- 侧边栏：参数与配置 ---
    with st.sidebar:
        st.title("🎛️ 策略控制台")
        
        st.subheader("1. 资产池配置")
        # 默认选中逻辑
        all_etfs = get_all_etf_list()
        if not all_etfs.empty:
            options = all_etfs['display'].tolist()
            defaults = [o for o in options if o.split(" | ")[0] in DEFAULT_CODES]
        else:
            options = DEFAULT_CODES
            defaults = DEFAULT_CODES
            
        selected_display = st.multiselect("核心标的池 (Universe)", options, default=defaults)
        selected_codes = [x.split(" | ")[0] for x in selected_display]
        
        st.divider()
        
        st.subheader("2. 策略参数 (当前)")
        p_lookback = st.slider("动量周期 (Lookback)", 5, 60, st.session_state.params['lookback'])
        p_smooth = st.slider("平滑窗口 (Smooth)", 1, 10, st.session_state.params['smooth'])
        p_threshold = st.number_input("换仓阈值 (Threshold)", 0.0, 0.05, st.session_state.params['threshold'], step=0.001, format="%.3f")
        
        # 更新Session
        st.session_state.params.update({'lookback': p_lookback, 'smooth': p_smooth, 'threshold': p_threshold})
        
        st.info("💡 分析师提示：\n较高的阈值可以减少震荡市的磨损，但可能导致信号滞后。建议结合波动率设定。")

    # --- 主界面 ---
    st.markdown("## 🚀 核心资产轮动策略终端 (AlphaTarget Pro)")
    
    if not selected_codes:
        st.warning("请在左侧选择至少一个标的。")
        st.stop()
        
    # 1. 数据加载
    with st.spinner("正在接入市场数据终端..."):
        data, name_map = download_market_data(selected_codes)
        
    if data is None:
        st.error("数据获取失败，请检查网络或代码有效性。")
        st.stop()

    # 2. 计算当前策略
    daily_ret = data.pct_change().fillna(0)
    mom_current = calculate_momentum(data, p_lookback, p_smooth)
    total_ret, max_dd, equity = fast_backtest(daily_ret, mom_current, p_threshold)
    
    # 构建详细回测结果 (用于绘图和信号)
    # 重跑一遍逻辑以获取持仓细节 (fast_backtest为了速度只返回了曲线)
    signal_mom = mom_current.shift(1)
    holdings = []
    capital = 1.0
    curve = []
    curr_hold = None
    
    dates = daily_ret.index
    for i, date in enumerate(dates):
        row = signal_mom.loc[date]
        r_today = daily_ret.loc[date]
        
        target = curr_hold
        if not row.isna().all():
            best_asset = row.idxmax()
            best_score = row.max()
            
            if curr_hold is None:
                target = best_asset
            else:
                curr_score = row[curr_hold]
                if pd.isna(curr_score): # 持仓退市或无数据
                    target = best_asset
                elif best_asset != curr_hold and best_score > curr_score + p_threshold:
                    target = best_asset
        
        # 计算净值
        ret = 0.0
        if target and target in r_today:
            ret = r_today[target]
            
        capital *= (1 + ret)
        curve.append(capital)
        holdings.append(target if target else "Cash")
        curr_hold = target
        
    df_res = pd.DataFrame({
        '总资产': curve,
        '持仓': holdings
    }, index=dates)
    
    # 3. 今日信号面板 (Dashboard)
    latest_date = data.index[-1]
    last_hold = holdings[-1]
    
    # 提取今日动量排名
    latest_mom = mom_current.iloc[-1].sort_values(ascending=False)
    
    # 构建信号卡片
    col_sig1, col_sig2 = st.columns([2, 1])
    
    with col_sig1:
        st.markdown(f"""
        <div class="signal-banner">
            <h3 style="margin:0">📌 当前持仓建议: {name_map.get(last_hold, last_hold) if last_hold != 'Cash' else '空仓观望'}</h3>
            <div style="margin-top:10px; opacity:0.9">
                数据截止: {latest_date.strftime('%Y-%m-%d')} | 策略周期: {p_lookback}日 | 阈值: {p_threshold:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_sig2:
        # 展示前三名动量
        st.markdown("**🏆 实时动量排名 (Top 3)**")
        for i, (asset, score) in enumerate(latest_mom.head(3).items()):
            display_name = name_map.get(asset, asset)
            st.markdown(f"{i+1}. **{display_name}**: `{score:.2%}`")

    # 4. 参优引擎 (Optimization Engine)
    with st.expander("🛠️ 策略参数优化引擎 (Backtest Optimizer)", expanded=False):
        st.markdown("通过网格搜索 (Grid Search) 遍历周期、平滑和阈值组合，寻找夏普比率与卡玛比率的最佳平衡点。")
        if st.button("开始参数寻优计算"):
            with st.spinner("AI正在进行多维参数空间遍历..."):
                opt_df = optimize_parameters(data)
                
                # 找到最佳
                best_ret_row = opt_df.loc[opt_df['累计收益'].idxmax()]
                best_calmar_row = opt_df.loc[opt_df['得分'].idxmax()]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🔥 进攻型组合 (最高收益)")
                    st.code(f"周期: {int(best_ret_row['周期'])}\n平滑: {int(best_ret_row['平滑'])}\n阈值: {best_ret_row['阈值']:.3f}\n\n累计收益: {best_ret_row['累计收益']:.2%}")
                    if st.button("应用进攻参数"):
                        st.session_state.params['lookback'] = int(best_ret_row['周期'])
                        st.session_state.params['smooth'] = int(best_ret_row['平滑'])
                        st.session_state.params['threshold'] = float(best_ret_row['阈值'])
                        st.rerun()
                        
                with c2:
                    st.markdown("#### 🛡️ 防御型组合 (最佳风报比)")
                    st.code(f"周期: {int(best_calmar_row['周期'])}\n平滑: {int(best_calmar_row['平滑'])}\n阈值: {best_calmar_row['阈值']:.3f}\n\n收益回撤比: {best_calmar_row['得分']:.2f}")
                    if st.button("应用稳健参数"):
                        st.session_state.params['lookback'] = int(best_calmar_row['周期'])
                        st.session_state.params['smooth'] = int(best_calmar_row['平滑'])
                        st.session_state.params['threshold'] = float(best_calmar_row['阈值'])
                        st.rerun()

    st.divider()
    
    # 5. 专业级回测报告
    st.subheader("📊 深度回测分析 (Analyst Report)")
    
    # 计算指标
    metrics = calculate_pro_metrics(df_res['总资产'].values)
    
    # 指标展示行
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    def metric_html(label, value, sub="", color="black"):
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """
    
    with m1: st.markdown(metric_html("累计收益", f"{metrics['Total Return']:.1%}", "Total Return", "#d62728"), unsafe_allow_html=True)
    with m2: st.markdown(metric_html("年化收益 (CAGR)", f"{metrics['CAGR']:.1%}", "Annualized"), unsafe_allow_html=True)
    with m3: st.markdown(metric_html("夏普比率", f"{metrics['Sharpe Ratio']:.2f}", "Risk Adjusted", "#1f77b4"), unsafe_allow_html=True)
    with m4: st.markdown(metric_html("卡玛比率", f"{metrics['Calmar Ratio']:.2f}", "Ret/MaxDD"), unsafe_allow_html=True)
    with m5: st.markdown(metric_html("最大回撤", f"{metrics['Max Drawdown']:.1%}", "Max Drawdown", "green"), unsafe_allow_html=True)
    with m6: st.markdown(metric_html("年化波动", f"{metrics['Volatility']:.1%}", "Volatility"), unsafe_allow_html=True)

    # 图表区
    tab_curve, tab_corr, tab_dd = st.tabs(["📈 净值与持仓", "🔗 资产相关性矩阵", "📉 动态回撤分析"])
    
    with tab_curve:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        
        # 净值曲线
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['总资产'], name="策略净值", line=dict(color='#1e3c72', width=2)), row=1, col=1)
        
        # 基准曲线 (简单等权)
        benchmark = (daily_ret.mean(axis=1) + 1).cumprod()
        benchmark = benchmark / benchmark.iloc[0]
        fig.add_trace(go.Scatter(x=df_res.index, y=benchmark, name="等权基准", line=dict(color='#adb5bd', width=1, dash='dash')), row=1, col=1)
        
        # 持仓色块
        # 将持仓转换为数值以便绘图
        codes = list(name_map.keys())
        # 创建颜色映射
        color_map = {c: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, c in enumerate(codes)}
        
        # 简化持仓显示，避免由于频繁换仓导致的渲染卡顿
        # 使用甘特图思想
        df_res['group'] = (df_res['持仓'] != df_res['持仓'].shift()).cumsum()
        for g, grp in df_res.groupby('group'):
            asset = grp['持仓'].iloc[0]
            start = grp.index[0]
            end = grp.index[-1]
            if asset in name_map: # 只绘制有效持仓
                c_code = asset
                c_name = name_map[asset]
                color = color_map.get(c_code, '#333')
                
                fig.add_trace(go.Scatter(
                    x=[start, end], y=[1, 1],
                    mode='lines',
                    line=dict(color=color, width=15),
                    name=c_name,
                    legendgroup="pos",
                    showlegend=False,
                    hovertemplate=f"持仓: {c_name}<br>{start.date()} ~ {end.date()}"
                ), row=2, col=1)
        
        fig.update_layout(height=500, margin=dict(t=20, b=20, l=40, r=40), hovermode="x unified")
        fig.update_yaxes(title="净值", row=1, col=1)
        fig.update_yaxes(showticklabels=False, title="持仓分布", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_corr:
        st.markdown("**为何关注相关性？** 核心资产轮动的精髓在于标的之间的低相关性。如果所有标的都高度正相关，轮动将失效。理想情况下，标的间应呈现低相关或负相关。")
        corr_matrix = data.pct_change().corr()
        # 将列名替换为中文名称
        corr_matrix.columns = [name_map.get(c, c) for c in corr_matrix.columns]
        corr_matrix.index = [name_map.get(c, c) for c in corr_matrix.index]
        
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with tab_dd:
        # 动态回撤图
        dd_series = (df_res['总资产'] - df_res['总资产'].cummax()) / df_res['总资产'].cummax()
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', line=dict(color='darkred', width=1), name="回撤"))
        fig_dd.update_layout(title="历史回撤幅度监控", yaxis_tickformat='.1%', height=400)
        st.plotly_chart(fig_dd, use_container_width=True)

if __name__ == "__main__":
    main()
