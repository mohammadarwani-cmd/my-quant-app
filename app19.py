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



# 安全导入 scipy，防止未安装导致程序崩溃

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

    'allow_cash': True,

    'selected_codes': DEFAULT_CODES

}



def load_config():

    """从本地文件加载配置，如果不存在则使用默认值"""

    if os.path.exists(CONFIG_FILE):

        try:

            with open(CONFIG_FILE, 'r') as f:

                saved_config = json.load(f)

                # 确保加载的配置包含所有必要的键（合并默认值，防止旧版配置缺失新键）

                config = DEFAULT_PARAMS.copy()

                config.update(saved_config)

                return config

        except Exception as e:

            # 文件损坏等情况，回退到默认

            return DEFAULT_PARAMS.copy()

    return DEFAULT_PARAMS.copy()



def save_config(config):

    """保存配置到本地文件"""

    try:

        with open(CONFIG_FILE, 'w') as f:

            json.dump(config, f)

    except Exception as e:

        pass # 忽略保存错误，避免中断程序



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

    

    /* 优化表格样式 */

    .dataframe {

        font-size: 13px !important;

    }

    

    /* 总资产大标题 */

    .total-asset-header {

        font-size: 2rem;

        font-weight: bold;

        color: #1e3c72;

        margin-bottom: 0.5rem;

    }

    .total-asset-sub {

        font-size: 1rem;

        color: #666;

    }

</style>

""", unsafe_allow_html=True)



# 全局常量配置

TRANSACTION_COST = 0.0001  # 万分之一



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

def download_market_data(codes_list, end_date_str):

    """

    数据下载核心

    """

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

        except Exception as e:

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



def fast_backtest_vectorized(daily_ret, mom_df, threshold, cost_rate=0.0001, allow_cash=True):

    """

    向量化快速回测 (含交易成本 & 绝对动量避险)

    """

    signal_mom = mom_df.shift(1)

    

    n_days, n_assets = daily_ret.shape

    p_ret = daily_ret.values

    p_mom = signal_mom.values

    

    strategy_ret = np.zeros(n_days)

    curr_idx = -2 # -2: 初始状态, -1: 空仓(Cash), 0~N: 资产索引

    

    trade_count = 0

    

    for i in range(n_days):

        row_mom = p_mom[i]

        

        # 跳过无效数据

        if np.isnan(row_mom).all(): 

            continue

            

        # 处理NaN为负无穷

        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)

        

        # 1. 找到最好的资产

        best_idx = np.argmax(clean_mom)

        best_val = clean_mom[best_idx]

        

        target_idx = curr_idx

        

        # 2. 绝对动量判定 (避险) - 可通过 allow_cash 开关控制

        if allow_cash and best_val < 0:

            target_idx = -1 # 强制空仓

        else:

            # 3. 相对动量判定 (轮动)

            if curr_idx == -2: # 刚启动

                if best_val > -np.inf: 

                    target_idx = best_idx

            elif curr_idx == -1: # 当前空仓

                if best_val > 0: # 只有大于0才入场 (如果是避险模式)；非避险模式下只要有信号就入

                    target_idx = best_idx

                elif not allow_cash: # 即使小于0，如果关闭避险，也要买

                    target_idx = best_idx

            else: # 当前持有资产

                # 如果持仓资产数据缺失(退市/停牌)，强制换到best

                curr_val = clean_mom[curr_idx]

                if best_idx != curr_idx:

                    if best_val > curr_val + threshold:

                        target_idx = best_idx

        

        # 4. 结算收益与成本

        r_day = 0.0

        

        # 计算交易成本

        if target_idx != curr_idx:

            # 只要仓位变动(包括开仓、平仓、换仓)，都扣除一次成本

            # 初始建仓也扣

            if curr_idx != -2:

                r_day -= cost_rate

                trade_count += 1

            curr_idx = target_idx

            

        # 计算持仓收益

        if curr_idx >= 0:

            r_day += p_ret[i, curr_idx]

        # else: 空仓收益为0

            

        strategy_ret[i] = r_day

            

    equity_curve = (1 + strategy_ret).cumprod()

    total_ret = equity_curve[-1] - 1

    

    cummax = np.maximum.accumulate(equity_curve)

    drawdown = (equity_curve - cummax) / cummax

    max_dd = drawdown.min()

    

    return total_ret, max_dd, equity_curve, trade_count



# ==========================================

# 4. 分析师工具箱 (Analyst Toolkit)

# ==========================================



def calculate_pro_metrics(equity_curve, benchmark_curve, trade_count):

    """

    计算投行级策略指标 (含Alpha/Beta)

    """

    if len(equity_curve) < 2: return {}

    

    # 转换为Series

    s_eq = pd.Series(equity_curve)

    s_bm = pd.Series(benchmark_curve) if len(benchmark_curve) == len(equity_curve) else None

    

    # 日收益率

    daily_ret = s_eq.pct_change().fillna(0)

    bm_ret = s_bm.pct_change().fillna(0) if s_bm is not None else None

    

    days = len(equity_curve)

    

    # 1. 基础收益

    total_ret = equity_curve[-1] - 1

    

    # 2. 年化收益

    ann_ret = (1 + total_ret) ** (252 / days) - 1

    

    # 3. 年化波动率

    ann_vol = daily_ret.std() * np.sqrt(252)

    

    # 4. 夏普比率 (无风险利率=3%)

    rf = 0.03

    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)

    

    # 5. 最大回撤

    cummax = np.maximum.accumulate(equity_curve)

    drawdown = (equity_curve - cummax) / cummax

    max_dd = drawdown.min()

    

    # 6. 卡玛比率

    calmar = ann_ret / (abs(max_dd) + 1e-9)

    

    # 7. Alpha & Beta (相对于等权基准)

    alpha, beta = 0.0, 0.0

    if HAS_SCIPY and bm_ret is not None and len(bm_ret) > 10:

        try:

            # 线性回归

            slope, intercept, _, _, _ = stats.linregress(bm_ret.values[1:], daily_ret.values[1:])

            beta = slope

            # Alpha需要年化: (日Alpha * 252)

            alpha = intercept * 252

        except:

            pass

            

    return {

        "Total Return": total_ret,

        "CAGR": ann_ret,

        "Volatility": ann_vol,

        "Max Drawdown": max_dd,

        "Sharpe Ratio": sharpe,

        "Calmar Ratio": calmar,

        "Alpha": alpha,

        "Beta": beta,

        "Trades": trade_count

    }



def optimize_parameters(data, allow_cash):

    """

    优化引擎 v2.0

    """

    lookbacks = range(20, 31, 1)

    smooths = range(1, 6, 1)    

    thresholds = np.arange(0.0, 0.013, 0.001) 

    

    daily_ret = data.pct_change().fillna(0)

    n_days = len(daily_ret) 

    results = []

    

    total_iters = len(lookbacks) * len(smooths) * len(thresholds)

    

    progress_text = f"多维参数空间遍历中 (含交易摩擦, 避险={'开启' if allow_cash else '关闭'})..."

    my_bar = st.progress(0, text=progress_text)

    

    idx = 0

    

    for lb in lookbacks:

        for sm in smooths:

            mom = calculate_momentum(data, lb, sm)

            for th in thresholds:

                ret, dd, _, count = fast_backtest_vectorized(daily_ret, mom, th, cost_rate=TRANSACTION_COST, allow_cash=allow_cash)

                ann_ret = (1 + ret) ** (252 / n_days) - 1

                score = ret / (abs(dd) + 0.05)

                results.append([lb, sm, th, ret, ann_ret, dd, count, score])

                

                idx += 1

                if idx % 100 == 0:

                    my_bar.progress(min(idx / total_iters, 1.0), text=f"{progress_text} {idx}/{total_iters}")

                    

    my_bar.empty()

    df_res = pd.DataFrame(results, columns=['周期', '平滑', '阈值', '累计收益', '年化收益', '最大回撤', '调仓次数', '得分'])

    return df_res



# ==========================================

# 5. 主程序 UI

# ==========================================



def main():

    # 1. 状态初始化 (优先加载本地保存的配置)

    if 'params' not in st.session_state:

        saved_config = load_config()

        st.session_state.params = saved_config



    # --- 侧边栏 ---

    with st.sidebar:

        st.title("🎛️ 策略控制台")

        

        st.subheader("1. 资产池配置")

        all_etfs = get_all_etf_list()

        

        # 处理选中项的默认值 (需确保在选项列表中)

        options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES

        

        # 从session_state或默认配置中获取已选代码

        current_selection_codes = st.session_state.params.get('selected_codes', DEFAULT_CODES)

        

        # 将代码转换为显示名称 (Options)

        default_display = []

        if not all_etfs.empty:

            for code in current_selection_codes:

                match = all_etfs[all_etfs['代码'] == code]

                if not match.empty:

                    default_display.append(match.iloc[0]['display'])

                else:

                    # 如果找不到对应显示名称，尝试保留代码(可能是手动输入的或过期的)

                    # 这里的逻辑主要是为了兼容。

                    # 简单起见，如果options里有包含该代码的，就选上

                    for opt in options:

                        if opt.startswith(code):

                            default_display.append(opt)

                            break

        else:

            default_display = current_selection_codes



        # 过滤掉不在options里的默认值，防止报错

        valid_defaults = [x for x in default_display if x in options]



        selected_display = st.multiselect("核心标的池", options, default=valid_defaults)

        selected_codes = [x.split(" | ")[0] for x in selected_display]

        

        st.divider()

        

        st.subheader("2. 资金管理实验室")

        

        date_mode = st.radio("回测区间", ["全历史 (2015至今)", "自定义区间"], index=0)

        start_date = datetime(2015, 1, 1)

        end_date = datetime.now()

        

        if date_mode == "自定义区间":

            c1, c2 = st.columns(2)

            start_date = c1.date_input("开始日期", datetime(2019, 1, 1))

            end_date = c2.date_input("结束日期", datetime.now())

            start_date = datetime.combine(start_date, datetime.min.time())

            end_date = datetime.combine(end_date, datetime.min.time())



        invest_mode = st.radio("投资模式", ["一次性投入 (Lump Sum)", "定期定额 (SIP)"], index=0)

        

        initial_capital = 100000.0

        sip_amount = 0.0

        sip_freq = "None"

        

        if invest_mode == "一次性投入 (Lump Sum)":

            initial_capital = st.number_input("初始本金", value=100000.0, step=10000.0)

        else:

            c1, c2 = st.columns(2)

            initial_capital = c1.number_input("初始底仓", value=10000.0, step=1000.0)

            sip_amount = c2.number_input("定投金额", value=2000.0, step=500.0)

            sip_freq = st.selectbox("定投频率", ["每月 (Monthly)", "每周 (Weekly)"], index=0)



        st.divider()

        

        st.subheader("3. 策略内核参数")

        # 使用 session_state 中的值作为控件默认值

        p_lookback = st.slider("动量周期 (Lookback)", 5, 60, st.session_state.params['lookback'])

        p_smooth = st.slider("平滑窗口 (Smooth)", 1, 10, st.session_state.params['smooth'])

        p_threshold = st.number_input("换仓阈值 (Threshold)", 0.0, 0.05, st.session_state.params['threshold'], step=0.001, format="%.3f")

        

        # 空仓避险开关

        p_allow_cash = st.checkbox("启用绝对动量避险 (Cash Protection)", 

                                   value=st.session_state.params.get('allow_cash', True),

                                   help="开启: 当最佳标的动量 < 0 时，全仓转为现金避险。\n关闭: 始终持有相对动量最高的标的，即使它在下跌。")

        

        st.caption(f"ℹ️ 当前交易费率设定: {TRANSACTION_COST*10000:.0f}‱ (万一)")

        

        # 实时更新 session_state 并自动保存到本地

        current_params = {

            'lookback': p_lookback, 

            'smooth': p_smooth, 

            'threshold': p_threshold,

            'allow_cash': p_allow_cash,

            'selected_codes': selected_codes

        }

        

        # 检查是否发生变化，有变化则保存

        if current_params != st.session_state.params:

            st.session_state.params = current_params

            save_config(current_params)

        

        st.divider()

        

        # 重置按钮

        if st.button("🔄 恢复默认设置 (Reset)", use_container_width=True):

            # 恢复默认配置

            default_conf = DEFAULT_PARAMS.copy()

            st.session_state.params = default_conf

            save_config(default_conf)

            st.rerun()



    # --- 主界面 ---

    st.markdown("## 🚀 核心资产轮动策略终端 (AlphaTarget Pro)")

    

    if not selected_codes:

        st.warning("请在左侧选择至少一个标的。")

        st.stop()

        

    # 1. 数据加载

    utc_now = datetime.now(timezone.utc)

    beijing_now = utc_now + timedelta(hours=8)

    

    if beijing_now.hour >= 15:

        target_date = beijing_now

        status_msg = f"当前北京时间 {beijing_now.strftime('%H:%M')} (已收盘)，获取截至今日数据"

    else:

        target_date = beijing_now - timedelta(days=1)

        status_msg = f"当前北京时间 {beijing_now.strftime('%H:%M')} (盘中)，获取截至昨日数据"

    

    end_date_str = target_date.strftime('%Y%m%d')



    with st.spinner("正在接入市场数据终端 (Smart-Link)..."):

        raw_data, name_map = download_market_data(selected_codes, end_date_str)

        

    st.toast(status_msg, icon="🕒")

        

    if raw_data is None:

        st.error("数据获取失败，请检查网络或代码有效性。")

        st.stop()



    # 2. 策略计算

    daily_ret_all = raw_data.pct_change().fillna(0)

    mom_all = calculate_momentum(raw_data, p_lookback, p_smooth)

    

    mask = (raw_data.index >= start_date) & (raw_data.index <= end_date)

    if not mask.any():

        st.error("选定区间内无有效交易数据，请调整日期。")

        st.stop()

        

    sliced_data = raw_data.loc[mask]

    sliced_mom = mom_all.loc[mask] 

    sliced_ret = daily_ret_all.loc[mask]

    

    # === 详细逐日回测循环 (含成本与避险) ===

    signal_mom = sliced_mom.shift(1) # T-1

    

    dates = sliced_ret.index

    holdings = []

    

    cash = initial_capital

    share_val = 0.0

    total_assets_curve = []

    total_invested_curve = []

    total_invested = initial_capital

    

    curr_hold = None # None表示初始状态

    trade_count_real = 0 # 实际调仓次数统计

    

    last_sip_date = dates[0]

    

    for i, date in enumerate(dates):

        # A. 定投逻辑

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

        

        # B. 信号与换仓逻辑

        row = signal_mom.loc[date]

        r_today = sliced_ret.loc[date]

        

        target = curr_hold

        is_trade_today = False

        

        if not row.isna().all():

            clean_row = row.fillna(-np.inf)

            best_asset = clean_row.idxmax()

            best_score = clean_row.max()

            

            # --- 绝对动量避险逻辑 (受 p_allow_cash 控制) ---

            if p_allow_cash and best_score < 0:

                target = 'Cash'

            else:

                # 相对动量轮动

                if curr_hold is None or curr_hold == 'Cash':

                    target = best_asset # 空仓转多仓 (或者刚开始)

                else:

                    curr_score = clean_row.get(curr_hold, -np.inf)

                    if best_asset != curr_hold:

                        if best_score > curr_score + p_threshold:

                            target = best_asset

        

        # C. 结算与成本扣除

        

        # 1. 计算日内持仓涨跌

        day_return = 0.0

        if curr_hold and curr_hold != 'Cash' and curr_hold in r_today:

             day_return = r_today[curr_hold]

        

        share_val = share_val * (1 + day_return)

        

        # 2. 执行换仓

        if target != curr_hold:

            # 发生交易 (包括 资产A->资产B, 资产->Cash, Cash->资产)

            if curr_hold is not None: 

                # 扣除成本 (基于当前总权益)

                total_equity_temp = share_val + cash

                cost = total_equity_temp * TRANSACTION_COST

                

                # 简单处理：从现金或市值中扣除

                if cash >= cost:

                    cash -= cost

                else:

                    share_val -= cost

                

                trade_count_real += 1

                is_trade_today = True



            # 资产转移逻辑

            if target == 'Cash':

                # 卖出所有变为现金

                cash += share_val

                share_val = 0.0

            else:

                # 变为特定资产

                total_money = share_val + cash

                share_val = total_money

                cash = 0.0

                

            curr_hold = target



        total_equity = share_val + cash

        

        total_assets_curve.append(total_equity)

        total_invested_curve.append(total_invested)

        holdings.append(target if target else "Cash")



    # 结果集

    df_res = pd.DataFrame({

        '总资产': total_assets_curve,

        '投入本金': total_invested_curve,

        '持仓': holdings,

    }, index=dates)

    

    # 格式化市场表现

    def format_market_perf(row, n_map):

        items = []

        sorted_items = row.sort_values(ascending=False)

        for code, val in sorted_items.items():

            name = n_map.get(code, code).split("(")[0]

            items.append(f"{name}: {val:+.2%}")

        return " | ".join(items)



    df_res['全市场表现'] = sliced_ret.apply(lambda r: format_market_perf(r, name_map), axis=1)

    df_res['策略日收益'] = df_res['总资产'].pct_change().fillna(0)

    

    # === 策略净值 (Unit NAV) 计算 - 含成本 & 避险状态传入 ===

    _, _, nav_series, _ = fast_backtest_vectorized(sliced_ret, sliced_mom, p_threshold, cost_rate=TRANSACTION_COST, allow_cash=p_allow_cash)

    df_res['策略净值'] = nav_series

    

    # === 计算 Benchmark (等权策略) ===

    # 简单构建一个不择时、等权持有的基准

    bm_daily_ret = sliced_ret.mean(axis=1)

    bm_curve = (1 + bm_daily_ret).cumprod()

    

    # 3. 今日信号面板

    latest_date = sliced_data.index[-1]

    last_hold = holdings[-1]

    latest_mom = mom_all.iloc[-1].dropna().sort_values(ascending=False)

    

    col_sig1, col_sig2 = st.columns([2, 1])

    with col_sig1:

        hold_name = name_map.get(last_hold, last_hold) if last_hold != 'Cash' else '🛡️ 空仓避险 (Cash)'

        mode_str = "开启" if p_allow_cash else "关闭"

        st.markdown(f"""

        <div class="signal-banner">

            <h3 style="margin:0">📌 当前持仓建议: {hold_name}</h3>

            <div style="margin-top:10px; opacity:0.9">

                数据截止: {latest_date.strftime('%Y-%m-%d')} | 避险模式: {mode_str} | 交易费率: 万一

            </div>

        </div>

        """, unsafe_allow_html=True)

    with col_sig2:

        st.markdown("**🏆 实时动量排名**")

        if not latest_mom.empty:

            top_score = latest_mom.iloc[0]

            if p_allow_cash and top_score < 0:

                 st.error(f"⚠️ 全线转弱 (最高 {top_score:.2%} < 0) -> 避险中")

            elif not p_allow_cash and top_score < 0:

                 st.warning(f"⚠️ 全线转弱 (最高 {top_score:.2%} < 0) -> 强制持有")

                 

            for i, (asset, score) in enumerate(latest_mom.head(3).items()):

                display_name = name_map.get(asset, asset)

                icon = "🔴" if score < 0 else "🟢"

                st.markdown(f"{i+1}. {icon} **{display_name}**: `{score:.2%}`")



    # 4. 优化引擎 (v2.1 含热力图)

    with st.expander("🛠️ 策略参数优化引擎 (Smart Optimizer)", expanded=False):

        c_opt1, c_opt2 = st.columns([1, 2])

        with c_opt1:

            opt_mode = st.radio("优化数据源", ["全历史数据", "当前选定区间"], index=0)

        

        data_to_opt = raw_data if opt_mode == "全历史数据" else sliced_data

        

        if st.button("运行参数寻优"):

            t0 = time.time()

            with st.spinner(f"正在进行多维参数回测 (避险模式={'开启' if p_allow_cash else '关闭'})..."):

                # 传入当前 UI 选择的 allow_cash 状态

                opt_df = optimize_parameters(data_to_opt, allow_cash=p_allow_cash)

                best_ret = opt_df.loc[opt_df['累计收益'].idxmax()]

                best_calmar = opt_df.loc[opt_df['得分'].idxmax()]

            

            st.success(f"✅ 优化完成 ({time.time()-t0:.1f}s)")

            

            c1, c2, c3 = st.columns([1, 1, 2])

            with c1:

                st.info("🔥 进攻型参数")

                st.write(f"Lookback: {int(best_ret['周期'])}")

                st.write(f"Threshold: {best_ret['阈值']:.3f}")

                st.metric("年化收益 (CAGR)", f"{best_ret['年化收益']:.1%}", f"累计: {best_ret['累计收益']:.1%}")

            with c2:

                st.success("🛡️ 防御型参数")

                st.write(f"Lookback: {int(best_calmar['周期'])}")

                st.write(f"Threshold: {best_calmar['阈值']:.3f}")

                st.metric("年化收益 (CAGR)", f"{best_calmar['年化收益']:.1%}", f"回撤: {best_calmar['最大回撤']:.1%}")

            

            with c3:

                st.markdown("**🌡️ 参数热力图 (周期 vs 阈值)**")

                # 聚合数据画热力图

                pivot_df = opt_df.pivot_table(index='阈值', columns='周期', values='得分', aggfunc='mean')

                fig_heat = px.imshow(pivot_df, labels=dict(x="Lookback", y="Threshold", color="Score"),

                                   color_continuous_scale="RdBu_r", aspect="auto", origin='lower')

                fig_heat.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))

                st.plotly_chart(fig_heat, use_container_width=True)



    st.divider()

    

    # 5. 核心报表区

    st.subheader("📊 账户深度分析 (Analyst Report)")

    

    # 核心指标

    account_ret = df_res['总资产'].iloc[-1] / df_res['投入本金'].iloc[-1] - 1

    account_profit = df_res['总资产'].iloc[-1] - df_res['投入本金'].iloc[-1]

    

    # 策略vs基准

    strat_metrics = calculate_pro_metrics(df_res['策略净值'].values, bm_curve.values, trade_count_real)

    

    # 展示总资产标题

    st.markdown(f"""

    <div style="margin-bottom: 20px;">

        <div class="total-asset-header">¥{df_res['总资产'].iloc[-1]:,.0f}</div>

        <div class="total-asset-sub">

            投入本金: ¥{df_res['投入本金'].iloc[-1]:,.0f} | 

            <span style="color: {'#d62728' if account_profit > 0 else 'green'}">

                总盈亏: {account_profit:+,.0f} ({account_ret:+.2%})

            </span>

        </div>

    </div>

    """, unsafe_allow_html=True)



    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1: st.markdown(metric_html("年化收益 (CAGR)", f"{strat_metrics.get('CAGR',0):.1%}", f"累计: {strat_metrics.get('Total Return',0):.1%}", "#d62728"), unsafe_allow_html=True)

    with m2: st.markdown(metric_html("最大回撤", f"{strat_metrics.get('Max Drawdown',0):.1%}", "历史极值", "green"), unsafe_allow_html=True)

    with m3: st.markdown(metric_html("夏普比率", f"{strat_metrics.get('Sharpe Ratio',0):.2f}", "风险调整后"), unsafe_allow_html=True)

    with m4: st.markdown(metric_html("策略Alpha", f"{strat_metrics.get('Alpha',0):+.1%}", "超额收益"), unsafe_allow_html=True)

    with m5: st.markdown(metric_html("策略Beta", f"{strat_metrics.get('Beta',0):.2f}", "市场敏感度"), unsafe_allow_html=True)

    with m6: st.markdown(metric_html("总交易次数", f"{trade_count_real}", "换手频率"), unsafe_allow_html=True)



    # 图表区

    tab_curve, tab_year, tab_daily, tab_dd = st.tabs(["📈 净值对比", "📅 年度回报", "📝 交易日记", "📉 风险透视"])

    

    with tab_curve:

        fig = go.Figure()

        # 策略净值

        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['策略净值'], name="策略净值 (Cost Adjusted)", line=dict(color='#d62728', width=2)))

        # 基准净值

        fig.add_trace(go.Scatter(x=df_res.index, y=bm_curve, name="等权基准 (Benchmark)", line=dict(color='#adb5bd', dash='dash')))

        

        # 标记空仓区域

        cash_mask = df_res['持仓'] == 'Cash'

        if cash_mask.any():

            cash_dates = df_res[cash_mask].index

            cash_vals = df_res.loc[cash_mask, '策略净值']

            fig.add_trace(go.Scatter(x=cash_dates, y=cash_vals, mode='markers', name="空仓避险", marker=dict(color='green', size=4, symbol='circle')))



        fig.update_layout(height=450, hovermode="x unified", title="策略 vs 基准 (Net Value)")

        st.plotly_chart(fig, use_container_width=True)

        

    with tab_year:

        res_y = []

        years = df_res.index.year.unique()

        for y in years:

            d_sub = df_res[df_res.index.year == y]

            start_nav = d_sub['策略净值'].iloc[0]

            end_nav = d_sub['策略净值'].iloc[-1]

            y_ret = end_nav / start_nav - 1

            

            # 基准同期

            b_start = bm_curve.loc[d_sub.index[0]]

            b_end = bm_curve.loc[d_sub.index[-1]]

            b_ret = b_end / b_start - 1

            

            res_y.append({

                "年份": y,

                "策略收益": y_ret,

                "基准收益": b_ret,

                "超额(Alpha)": y_ret - b_ret

            })

            

        df_year = pd.DataFrame(res_y).set_index("年份")

        st.markdown("#### 年度超额收益表")

        st.dataframe(

            df_year.style.format("{:+.2%}").background_gradient(subset=["超额(Alpha)"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),

            use_container_width=True

        )

        

    with tab_daily:

        show_df = df_res[['总资产', '投入本金', '持仓', '全市场表现']].copy()

        show_df['持仓名称'] = show_df['持仓'].map(lambda x: name_map.get(x, x))

        show_df = show_df.sort_index(ascending=False)

        st.dataframe(

            show_df.style.format({"总资产": "{:,.2f}", "投入本金": "{:,.2f}"}), 

            use_container_width=True, height=400

        )



    with tab_dd:

        dd_series = (df_res['策略净值'] - df_res['策略净值'].cummax()) / df_res['策略净值'].cummax()

        fig_dd = go.Figure()

        fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', line=dict(color='darkred', width=1), name="回撤"))

        fig_dd.update_layout(title="策略动态回撤", yaxis_tickformat='.1%', height=400)

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
