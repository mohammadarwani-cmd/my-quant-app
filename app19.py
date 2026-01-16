import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import functools

# ==========================================
# 1. 页面配置 & CSS
# ==========================================
st.set_page_config(
    page_title="Alpha Dash | 现金流定投实战版",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .status-card { background-color: #f8fafc; border-left: 5px solid #10b981; padding: 12px; border-radius: 6px; margin-bottom: 20px; }
    .holding-tag { background: #064e3b; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 金融计算工具 (免 SciPy 版)
# ==========================================

def xirr(cashflows):
    """
    计算内部收益率 XIRR
    cashflows: list of tuples (date, amount)
    """
    if not cashflows or len(cashflows) < 2:
        return np.nan
        
    d0 = cashflows[0][0]
    years = np.array([(d - d0).days / 365.0 for d, a in cashflows])
    amounts = np.array([a for d, a in cashflows])
    
    rate = 0.1
    for _ in range(25):
        t = 1 + rate
        if t <= 0: return np.nan
        npv = np.sum(amounts / (t ** years))
        npv_der = np.sum(-years * amounts / (t ** (years + 1)))
        if npv_der == 0: break
        new_rate = rate - npv / npv_der
        if abs(new_rate - rate) < 1e-6:
            return new_rate
        rate = new_rate
    return rate if -1 < rate < 10 else np.nan

# ==========================================
# 3. 数据引擎
# ==========================================

@st.cache_data(ttl=3600*6)
def get_all_etf_list():
    try:
        df = ak.fund_etf_spot_em()
        df['display'] = df['代码'] + " | " + df['名称']
        return df[['代码', '名称', 'display']]
    except:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def load_data(assets_config, start_date, end_date):
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    price_dict = {}
    
    # 强制获取基准
    try:
        bm_df = ak.fund_etf_hist_em(symbol="510300", period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        if not bm_df.empty:
            bm_df['日期'] = pd.to_datetime(bm_df['日期'])
            price_dict['基准(300)'] = bm_df.set_index('日期')['收盘']
    except:
        st.error("无法获取基准行情数据")

    for code, conf in assets_config.items():
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                name = conf['name'] + "(轮动)" if code == "510300" else conf['name']
                price_dict[name] = df.set_index('日期')['收盘']
        except:
            continue
            
    if not price_dict: return None
    return pd.concat(price_dict, axis=1).sort_index().ffill().dropna(how='all')

# ==========================================
# 4. 现金流回测内核
# ==========================================

def run_cashflow_backtest(data, lb, sm, th, mh, init_cash, invest_amt, invest_freq):
    asset_cols = [c for c in data.columns if c != '基准(300)']
    prices = data[asset_cols]
    rets = prices.pct_change().values
    
    # 计算动量
    mom = prices.pct_change(lb)
    if sm > 1: mom = mom.rolling(sm).mean()
    moms = mom.shift(1).fillna(-999).values
    dates = data.index
    
    n_days = len(dates)
    cap_strat = init_cash
    cap_bm = init_cash
    current_invested = init_cash
    
    strat_curve = np.zeros(n_days)
    bm_curve = np.zeros(n_days)
    invested_curve = np.zeros(n_days)
    holdings = [None] * n_days
    
    cashflows_strat = [(dates[0], -init_cash)]
    cashflows_bm = [(dates[0], -init_cash)]
    
    curr_idx = -1
    days_held = 0
    
    for t in range(n_days):
        # 1. 处理定投
        if t > 0 and t % invest_freq == 0:
            cap_strat += invest_amt
            cap_bm += invest_amt
            current_invested += invest_amt
            cashflows_strat.append((dates[t], -invest_amt))
            cashflows_bm.append((dates[t], -invest_amt))
            
        # 2. 策略轮动逻辑
        m_row = moms[t]
        best_idx = np.argmax(m_row)
        best_val = m_row[best_idx]
        
        if curr_idx == -1:
            if best_val > -900: curr_idx = best_idx
        else:
            days_held += 1
            if best_idx != curr_idx and best_val > m_row[curr_idx] + th:
                if days_held >= mh:
                    curr_idx = best_idx
                    days_held = 0
        
        # 3. 资产演变
        if curr_idx != -1:
            r = rets[t, curr_idx]
            if not np.isnan(r): cap_strat *= (1 + r)
            holdings[t] = asset_cols[curr_idx]
        
        bm_r = data['基准(300)'].pct_change().iloc[t]
        if not np.isnan(bm_r): cap_bm *= (1 + bm_r)
        
        strat_curve[t] = cap_strat
        bm_curve[t] = cap_bm
        invested_curve[t] = current_invested

    cf_s = cashflows_strat + [(dates[-1], cap_strat)]
    cf_b = cashflows_bm + [(dates[-1], cap_bm)]
    
    s_xirr = xirr(cf_s)
    b_xirr = xirr(cf_b)
    
    res = pd.DataFrame({
        '资产总值': strat_curve,
        '基准总值': bm_curve,
        '当前持仓': holdings,
        '累计投入': invested_curve,
        '当日盈亏比': (strat_curve - invested_curve) / invested_curve
    }, index=dates)
    
    return res, s_xirr, b_xirr

# ==========================================
# 5. UI 渲染
# ==========================================

def main():
    st.sidebar.header("⏳ 回测时间跨度")
    col_s, col_e = st.sidebar.columns(2)
    with col_s:
        start_date = st.date_input("开始日期", datetime(2020, 1, 1))
    with col_e:
        end_date = st.date_input("结束日期", datetime.now())

    st.sidebar.header("💰 现金流设置")
    init_cash = st.sidebar.number_input("初始投入金额 (元)", 10000, 10000000, 500000, step=10000)
    invest_amt = st.sidebar.number_input("定期定投金额 (元)", 0, 1000000, 10000, step=1000)
    invest_freq = st.sidebar.slider("定投频率 (交易日间隔)", 1, 60, 20)
    
    st.sidebar.header("⚙️ 策略参数")
    all_etf = get_all_etf_list()
    default_codes = ["513100", "518880", "588000", "159941"]
    
    if not all_etf.empty:
        options = all_etf['display'].tolist()
        defaults = [o for o in options if any(c in o for c in default_codes)]
        selected = st.sidebar.multiselect("资产池", options, default=defaults)
    else:
        selected = []
    
    asset_dict = {s.split(" | ")[0]: {'name': s.split(" | ")[1].replace("ETF","")} for s in selected}
    
    lb = st.sidebar.slider("动量观察期", 5, 60, 20)
    sm = st.sidebar.slider("平滑期", 1, 10, 3)
    th = st.sidebar.slider("换仓阈值", 0.0, 0.05, 0.005, 0.001)

    st.title("⚖️ Alpha Dash | 现金流定投增强版")
    
    if not asset_dict:
        st.info("请在左侧选择资产并设置金额。")
        return

    if start_date >= end_date:
        st.error("开始日期必须早于结束日期")
        return

    with st.spinner("获取历史行情并计算..."):
        data = load_data(asset_dict, start_date, end_date)
        if data is None or data.empty:
            st.warning("所选时间段内没有足够的数据。")
            return
            
        res_df, s_xirr, b_xirr = run_cashflow_backtest(data, lb, sm, th, 2, init_cash, invest_amt, invest_freq)

    # 核心指标看板
    total_invested = res_df['累计投入'].iloc[-1]
    final_val = res_df['资产总值'].iloc[-1]
    total_profit_pct = (final_val - total_invested) / total_invested
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("期末总资产", f"¥{final_val:,.0f}")
    m2.metric("累计投入本金", f"¥{total_invested:,.0f}")
    m3.metric("累计盈亏比", f"{total_profit_pct:+.2%}")
    m4.metric("策略 XIRR (年化)", f"{s_xirr:.2%}" if not np.isnan(s_xirr) else "N/A", 
              delta=f"{s_xirr-b_xirr:+.2%}" if not np.isnan(s_xirr) and not np.isnan(b_xirr) else None)

    # 绘图区域
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['资产总值'], name="策略资产", line=dict(color='#2563eb', width=2)))
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['基准总值'], name="基准(300)", line=dict(color='#94a3b8', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['累计投入'], name="累计本金", fill='tozeroy', line=dict(color='rgba(200, 200, 200, 0.2)', width=0)))
    
    fig.update_layout(title=f"资金演变图 ({start_date} 至 {end_date})", template="plotly_white", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 详情数据选项卡
    t1, t2 = st.tabs(["🔎 每日操作记录与细节", "📜 定投及收益概览"])
    with t1:
        st.markdown("##### 每日详细账单")
        # 格式化输出，方便阅读
        display_df = res_df.copy()
        display_df['当日盈亏比'] = display_df['当日盈亏比'].map('{:+.2%}'.format)
        display_df['资产总值'] = display_df['资产总值'].map('{:,.0f}'.format)
        display_df['累计投入'] = display_df['累计投入'].map('{:,.0f}'.format)
        st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)
        
    with t2:
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.write(f"**回测统计:**")
            st.write(f"- 交易日数: {len(res_df)}")
            st.write(f"- 定投次数: {len(res_df)//invest_freq} 次")
            st.write(f"- 初始本金权重: {init_cash/total_invested:.1%}")
        with col_info2:
            st.write(f"**当前状态:**")
            curr_h = res_df['当前持仓'].iloc[-1]
            st.write(f"- 最新持仓: {curr_h if curr_h else '空仓'}")
            st.write(f"- 基准年化 XIRR: {b_xirr:.2%}")

if __name__ == "__main__":
    main()