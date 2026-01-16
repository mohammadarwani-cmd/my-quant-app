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
    
    /* 年份收益表格样式 */
    .dataframe {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# 默认标的池
DEFAULT_CODES = ["518880", "588000", "513100", "510180"]

# 预置ETF映射表
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
    下载全量数据，后续在内存中进行时间切片
    """
    now = datetime.now()
    if now.hour >= 15:
        target_date = now
    else:
        target_date = now - timedelta(days=1)
        
    start_str = '20150101' 
    end_str = target_date.strftime('%Y%m%d')
    
    price_dict = {}
    name_map = {}
    
    # 获取名称映射
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
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                price_dict[name] = df['收盘'].astype(float)
        except Exception as e:
            # st.error(f"代码 {code} 数据获取失败: {str(e)}") # 减少报错干扰
            continue

    if not price_dict:
        return None, None

    # 对齐数据
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

def fast_backtest_vectorized(daily_ret, mom_df, threshold):
    """
    向量化快速回测 (仅用于参数优化的净值计算，假设一次性投入)
    """
    signal_mom = mom_df.shift(1)
    
    n_days, n_assets = daily_ret.shape
    p_ret = daily_ret.values
    p_mom = signal_mom.values
    
    strategy_ret = np.zeros(n_days)
    curr_idx = -1 
    
    for i in range(n_days):
        row_mom = p_mom[i]
        if np.isnan(row_mom).all(): continue
            
        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)
        best_idx = np.argmax(clean_mom)
        best_val = clean_mom[best_idx]
        
        if curr_idx == -1:
            if best_val > -np.inf: curr_idx = best_idx
        else:
            curr_val = clean_mom[curr_idx]
            if best_idx != curr_idx:
                if best_val > curr_val + threshold:
                    curr_idx = best_idx
        
        if curr_idx != -1:
            strategy_ret[i] = p_ret[i, curr_idx]
            
    equity_curve = (1 + strategy_ret).cumprod()
    total_ret = equity_curve[-1] - 1
    
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    
    return total_ret, max_dd, equity_curve

# ==========================================
# 4. 分析师工具箱 (Analyst Toolkit)
# ==========================================

def calculate_pro_metrics(equity_curve, days_count):
    """
    计算投行级策略指标
    """
    if len(equity_curve) < 2: return {}
    
    # 日收益率
    daily_ret = pd.Series(equity_curve).pct_change().fillna(0)
    
    # 1. 基础收益
    total_ret = equity_curve[-1] / equity_curve[0] - 1
    
    # 2. 年化收益
    if days_count == 0: days_count = len(equity_curve)
    ann_ret = (1 + total_ret) ** (252 / days_count) - 1
    
    # 3. 年化波动率
    ann_vol = daily_ret.std() * np.sqrt(252)
    
    # 4. 夏普比率
    rf = 0.03
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)
    
    # 5. 最大回撤
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    
    # 6. 卡玛比率
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
    参数网格搜索引擎 (基于Lump Sum模式进行快速筛选)
    """
    lookbacks = range(10, 35, 2)
    smooths = [1, 3, 5, 8]
    thresholds = [0.0, 0.005, 0.01]
    
    daily_ret = data.pct_change().fillna(0)
    results = []
    
    total_iters = len(lookbacks) * len(smooths) * len(thresholds)
    progress_bar = st.progress(0)
    idx = 0
    
    for lb in lookbacks:
        for sm in smooths:
            mom = calculate_momentum(data, lb, sm)
            for th in thresholds:
                # 优化时使用向量化回测，速度最快
                ret, dd, _ = fast_backtest_vectorized(daily_ret, mom, th)
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
    if 'params' not in st.session_state:
        st.session_state.params = {'lookback': 20, 'smooth': 3, 'threshold': 0.005}

    # --- 侧边栏 ---
    with st.sidebar:
        st.title("🎛️ 策略控制台")
        
        # 1. 资产池
        st.subheader("1. 资产池配置")
        all_etfs = get_all_etf_list()
        options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES
        defaults = [o for o in options if o.split(" | ")[0] in DEFAULT_CODES] if not all_etfs.empty else DEFAULT_CODES
        selected_display = st.multiselect("核心标的池", options, default=defaults)
        selected_codes = [x.split(" | ")[0] for x in selected_display]
        
        st.divider()
        
        # 2. 资金管理实验室 (新增)
        st.subheader("2. 资金管理实验室")
        
        # A. 时间段选择
        date_mode = st.radio("回测区间", ["全历史 (2015至今)", "自定义区间"], index=0)
        start_date = datetime(2015, 1, 1)
        end_date = datetime.now()
        
        if date_mode == "自定义区间":
            c1, c2 = st.columns(2)
            start_date = c1.date_input("开始日期", datetime(2019, 1, 1))
            end_date = c2.date_input("结束日期", datetime.now())
            # 转换为datetime
            start_date = datetime.combine(start_date, datetime.min.time())
            end_date = datetime.combine(end_date, datetime.min.time())

        # B. 投资模式
        invest_mode = st.radio("投资模式", ["一次性投入 (Lump Sum)", "定期定额 (SIP)"], index=0)
        
        initial_capital = 100000.0
        sip_amount = 0.0
        sip_freq = "None"
        
        if invest_mode == "一次性投入 (Lump Sum)":
            initial_capital = st.number_input("初始本金", value=100000.0, step=10000.0)
        else:
            c1, c2 = st.columns(2)
            initial_capital = c1.number_input("初始底仓", value=10000.0, step=1000.0, help="开始时投入的第一笔资金")
            sip_amount = c2.number_input("定投金额", value=2000.0, step=500.0)
            sip_freq = st.selectbox("定投频率", ["每月 (Monthly)", "每周 (Weekly)"], index=0)

        st.divider()
        
        # 3. 策略参数
        st.subheader("3. 策略内核参数")
        p_lookback = st.slider("动量周期", 5, 60, st.session_state.params['lookback'])
        p_smooth = st.slider("平滑窗口", 1, 10, st.session_state.params['smooth'])
        p_threshold = st.number_input("换仓阈值", 0.0, 0.05, st.session_state.params['threshold'], step=0.001, format="%.3f")
        
        st.session_state.params.update({'lookback': p_lookback, 'smooth': p_smooth, 'threshold': p_threshold})

    # --- 主界面 ---
    st.markdown("## 🚀 核心资产轮动策略终端 (AlphaTarget Pro)")
    
    if not selected_codes:
        st.warning("请在左侧选择至少一个标的。")
        st.stop()
        
    # 1. 数据加载
    with st.spinner("正在构建数据立方体..."):
        # 下载全量数据 (为了计算指标，需要比选定开始时间更早的数据)
        raw_data, name_map = download_market_data(selected_codes)
        
    if raw_data is None:
        st.error("数据获取失败，请检查网络或代码有效性。")
        st.stop()

    # 2. 策略计算 (含定投逻辑)
    # 先计算全量动量，防止切片导致开头无数据
    daily_ret_all = raw_data.pct_change().fillna(0)
    mom_all = calculate_momentum(raw_data, p_lookback, p_smooth)
    
    # 时间切片：根据用户选择截取回测段
    # 注意：需要确保切片后的数据包含 start_date
    mask = (raw_data.index >= start_date) & (raw_data.index <= end_date)
    # 如果筛选后为空，提示
    if not mask.any():
        st.error("选定区间内无有效交易数据，请调整日期。")
        st.stop()
        
    sliced_data = raw_data.loc[mask]
    sliced_mom = mom_all.loc[mask] # 动量已经算好，直接切片即可
    sliced_ret = daily_ret_all.loc[mask]
    
    # 详细逐日回测循环 (支持定投)
    signal_mom = sliced_mom.shift(1) # T-1日的信号
    
    dates = sliced_ret.index
    holdings = []
    
    # 资金账户
    cash = initial_capital
    share_val = 0.0
    total_assets_curve = []
    total_invested_curve = [] # 记录投入本金(计算定投收益率用)
    total_invested = initial_capital
    
    curr_hold = None # 当前持有的资产代码
    
    # 定投辅助
    last_sip_date = dates[0]
    
    for i, date in enumerate(dates):
        # --- 1. 定投逻辑 ---
        # 简单判断：如果是定投模式，检查日期间隔
        if invest_mode == "定期定额 (SIP)" and i > 0:
            is_sip_day = False
            if sip_freq.startswith("每月"):
                # 如果月份变了，且今天是该月较早的日子
                if date.month != last_sip_date.month:
                    is_sip_day = True
            elif sip_freq.startswith("每周"):
                # 简单处理：每隔5个交易日(近似)
                # 更精确的是判断 date.weekday()，假设每周一(0)定投
                if date.weekday() == 0 and last_sip_date.weekday() != 0: 
                    is_sip_day = True
            
            if is_sip_day:
                cash += sip_amount
                total_invested += sip_amount
                last_sip_date = date
        
        # --- 2. 信号与换仓逻辑 ---
        row = signal_mom.loc[date]
        r_today = sliced_ret.loc[date]
        
        target = curr_hold
        
        # 只有当有有效信号时才尝试换仓
        if not row.isna().all():
            best_asset = row.idxmax()
            best_score = row.max()
            
            if curr_hold is None:
                # 空仓状态，有信号即买入
                target = best_asset
            else:
                curr_score = row.get(curr_hold, -np.inf) # 如果持仓数据缺失，得分极低，强制换仓
                
                # 换仓条件
                if best_asset != curr_hold:
                    if best_score > curr_score + p_threshold:
                        target = best_asset
        
        # --- 3. 结算当日收益 ---
        # 如果有持仓，计算持仓收益
        day_return = 0.0
        if curr_hold and curr_hold in r_today:
             day_return = r_today[curr_hold]
        
        # 更新资产: 资产 = 现金(未投入部分) + 证券市值 * (1+涨跌幅)
        # 这里为了简化模型，假设全仓买入/卖出，即 Cash 和 MarketValue 互转
        
        # 模拟逻辑：
        # 如果发生换仓 (curr_hold != target):
        # 1. 卖出 curr_hold (MarketValue -> Cash)
        # 2. 买入 target (Cash -> MarketValue)
        # 如果不换仓:
        # MarketValue 随涨跌幅波动
        # Cash (定投刚进来的钱) 暂时不动，或者并在今天买入？
        # 假设：定投的钱在当天收盘前也会买入当前的 target
        
        # 修正逻辑：始终维持满仓 target (除去不可抗力)
        # 现有总权益
        equity_before_ret = cash + share_val
        
        # 经过一天的波动
        # 注意：day_return 是今天的涨跌，作用于昨天的 share_val
        # 但如果是定投日，cash增加了，这部分cash今天能吃到涨跌吗？通常按收盘价买入，吃不到当天的。
        
        share_val = share_val * (1 + day_return)
        
        # 这里的 cash 主要是定投刚进来的钱，或者刚卖出后的钱
        # 我们假设所有 cash 都会在当天收盘转为 target (如果是有效target)
        if target:
            # 资金入场
            share_val += cash 
            cash = 0.0
        
        total_equity = share_val + cash
        
        total_assets_curve.append(total_equity)
        total_invested_curve.append(total_invested)
        holdings.append(target if target else "Cash")
        curr_hold = target

    # 结果集整理
    df_res = pd.DataFrame({
        '总资产': total_assets_curve,
        '投入本金': total_invested_curve,
        '持仓': holdings,
        '日收益率': sliced_ret.mean(axis=1) # 仅作参考，实际收益率需按净值算
    }, index=dates)
    
    # 重新计算真实的策略日收益率 (基于净值)
    df_res['策略日收益'] = df_res['总资产'].pct_change().fillna(0)
    # 修正定投日的收益率失真：资产增加是因为充值，不是因为涨跌
    # 当日净值变动 = (今日总资产 - 昨日总资产 - 今日定投额) / 昨日总资产
    s_diff = df_res['总资产'] - df_res['总资产'].shift(1).fillna(initial_capital)
    
    # === 引入单位净值计算 (Unit NAV) 用于评估策略能力 ===
    units = initial_capital # 初始份额
    nav_curve = []
    curr_nav = 1.0
    
    # 快速获取 NAV 曲线 (即该区间的一次性满仓回测结果)
    # 利用之前的向量化函数，针对 sliced 数据跑一次作为基准策略表现
    _, _, nav_series = fast_backtest_vectorized(sliced_ret, sliced_mom, p_threshold)
    df_res['策略净值'] = nav_series
    
    # 3. 今日信号面板
    latest_date = sliced_data.index[-1]
    last_hold = holdings[-1]
    latest_mom = mom_all.iloc[-1].sort_values(ascending=False) # 使用mom_all获取最新
    
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
        st.markdown("**🏆 实时动量排名**")
        for i, (asset, score) in enumerate(latest_mom.head(3).items()):
            display_name = name_map.get(asset, asset)
            st.markdown(f"{i+1}. **{display_name}**: `{score:.2%}`")

    # 4. 优化引擎 (提示仅支持Lump Sum逻辑)
    with st.expander("🛠️ 策略参数优化引擎", expanded=False):
        st.info("注：参数优化基于全历史数据的“一次性投入”模式进行计算，以寻找普适性最佳参数。")
        if st.button("运行参数寻优"):
            with st.spinner("AI正在遍历参数矩阵..."):
                opt_df = optimize_parameters(raw_data)
                best_ret = opt_df.loc[opt_df['累计收益'].idxmax()]
                best_calmar = opt_df.loc[opt_df['得分'].idxmax()]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.code(f"🔥 进攻型 (Ret {best_ret['累计收益']:.1%})\nLookback: {int(best_ret['周期'])}, Smooth: {int(best_ret['平滑'])}, Thres: {best_ret['阈值']:.3f}")
                with c2:
                    st.code(f"🛡️ 防御型 (Score {best_ret['得分']:.2f})\nLookback: {int(best_calmar['周期'])}, Smooth: {int(best_calmar['平滑'])}, Thres: {best_calmar['阈值']:.3f}")

    st.divider()
    
    # 5. 核心报表区
    st.subheader("📊 账户深度分析")
    
    # 核心指标计算
    # 账户总收益率 (MWR近似)
    account_ret = df_res['总资产'].iloc[-1] / df_res['投入本金'].iloc[-1] - 1
    account_profit = df_res['总资产'].iloc[-1] - df_res['投入本金'].iloc[-1]
    
    # 策略表现指标 (基于净值)
    strat_metrics = calculate_pro_metrics(df_res['策略净值'].values, len(df_res))
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.markdown(metric_html("账户总资产", f"¥{df_res['总资产'].iloc[-1]:,.0f}", f"本金: ¥{df_res['投入本金'].iloc[-1]:,.0f}"), unsafe_allow_html=True)
    with m2: st.markdown(metric_html("账户累计收益", f"{account_ret:+.2%}", f"盈亏: ¥{account_profit:+,.0f}", color="#d62728" if account_profit>0 else "green"), unsafe_allow_html=True)
    with m3: st.markdown(metric_html("策略年化 (CAGR)", f"{strat_metrics.get('CAGR',0):.1%}", "Time Weighted"), unsafe_allow_html=True)
    with m4: st.markdown(metric_html("最大回撤", f"{strat_metrics.get('Max Drawdown',0):.1%}", "策略风险"), unsafe_allow_html=True)
    with m5: st.markdown(metric_html("夏普比率", f"{strat_metrics.get('Sharpe Ratio',0):.2f}", "风险调整后收益"), unsafe_allow_html=True)

    # 图表区
    tab_curve, tab_year, tab_daily, tab_dd = st.tabs(["📈 资产曲线", "📅 年度回报表", "📝 每日交易日记", "📉 风险分析"])
    
    with tab_curve:
        fig = go.Figure()
        # 账户资产
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['总资产'], name="账户总资产", line=dict(color='#1e3c72', width=2)))
        # 投入本金线
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['投入本金'], name="投入本金", line=dict(color='#adb5bd', dash='dash')))
        
        # 持仓背景
        # 简化版：仅显示主颜色条
        fig.update_layout(height=450, hovermode="x unified", title="账户资产增长曲线 (Asset Growth)")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_year:
        # 计算分年度收益 (基于策略净值，代表策略本身能力)
        # 也可以计算账户年度盈亏
        
        # 1. 策略年度表现
        yearly_nav = df_res['策略净值'].resample('Y').last()
        yearly_ret = yearly_nav.pct_change()
        # 补上第一年：第一年年底净值 / 1 - 1 (如果第一年就是起点)
        # 或者: (year_end / year_start) - 1
        
        res_y = []
        years = df_res.index.year.unique()
        for y in years:
            d_sub = df_res[df_res.index.year == y]
            start_nav = d_sub['策略净值'].iloc[0]
            end_nav = d_sub['策略净值'].iloc[-1]
            y_ret = end_nav / start_nav - 1
            
            # 账户当年盈亏
            start_asset = d_sub['总资产'].iloc[0]
            end_asset = d_sub['总资产'].iloc[-1]
            # 扣除当年投入
            net_inflow = d_sub['投入本金'].iloc[-1] - d_sub['投入本金'].iloc[0]
            y_profit = end_asset - start_asset - net_inflow
            
            res_y.append({
                "年份": y,
                "策略收益率": y_ret,
                "账户当年盈亏": y_profit
            })
            
        df_year = pd.DataFrame(res_y).set_index("年份")
        
        # 热力图样式
        st.markdown("#### 分年度表现 (Yearly Performance)")
        st.dataframe(
            df_year.style.format({
                "策略收益率": "{:+.2%}",
                "账户当年盈亏": "{:+,.0f}"
            }).background_gradient(subset=["策略收益率"], cmap="RdYlGn", vmin=-0.3, vmax=0.3),
            use_container_width=True
        )
        
    with tab_daily:
        st.markdown("#### 每日交易详细记录")
        # 格式化显示
        show_df = df_res[['总资产', '投入本金', '持仓', '策略净值']].copy()
        show_df['持仓名称'] = show_df['持仓'].map(lambda x: name_map.get(x, x))
        show_df = show_df.sort_index(ascending=False)
        st.dataframe(
            show_df.style.format({
                "总资产": "{:,.2f}",
                "投入本金": "{:,.2f}",
                "策略净值": "{:.4f}"
            }), 
            use_container_width=True,
            height=400
        )

    with tab_dd:
        dd_series = (df_res['策略净值'] - df_res['策略净值'].cummax()) / df_res['策略净值'].cummax()
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', line=dict(color='darkred', width=1), name="回撤"))
        fig_dd.update_layout(title="策略历史回撤 (Drawdown)", yaxis_tickformat='.1%', height=400)
        st.plotly_chart(fig_dd, use_container_width=True)

def metric_html(label, value, sub="", color="black"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """

if __name__ == "__main__":
    main()
