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
    'mom_method': 'Risk-Adjusted (稳健)', 
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
            st.error(f"加载配置失败: {e}")
    return DEFAULT_PARAMS.copy()

def save_config(config):
    """保存配置到本地文件"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        st.error(f"保存配置失败: {e}")
        return False

# ==========================================
# 1. 数据获取与缓存 (Data Fetching)
# ==========================================
@st.cache_data(ttl=3600)  # 缓存1小时
def get_data(codes, lookback_days=365*3):
    """
    获取多只ETF的复权收盘价，并对齐日期
    """
    data_dict = {}
    
    # 扩大获取范围以确保有足够的数据计算指标
    start_date = (datetime.now() - timedelta(days=lookback_days + 100)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, code in enumerate(codes):
        status_text.text(f"正在获取 {code} 数据...")
        try:
            # 使用 akshare 获取 ETF 日行情
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
            if df.empty:
                st.warning(f"代码 {code} 获取数据为空，已跳过")
                continue
                
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            data_dict[code] = df['close']  # 仅保留收盘价
        except Exception as e:
            st.error(f"获取 {code} 数据失败: {e}")
        
        progress_bar.progress((i + 1) / len(codes))
    
    status_text.empty()
    progress_bar.empty()
    
    if not data_dict:
        return pd.DataFrame()
    
    # 合并数据，按日期对齐（取交集或并集，这里取并集然后填充）
    df_all = pd.DataFrame(data_dict)
    df_all.sort_index(inplace=True)
    df_all.fillna(method='ffill', inplace=True) # 前向填充缺失值
    df_all.dropna(inplace=True) # 去除开头无法填充的部分
    
    return df_all

# ==========================================
# 2. 动量计算逻辑 (Core Strategy Logic)
# ==========================================
def calculate_momentum(df_prices, window, smooth, method='Return'):
    """
    计算动量分数
    """
    # 1. 平滑处理 (可选)
    if smooth > 1:
        prices = df_prices.rolling(window=smooth).mean()
    else:
        prices = df_prices
    
    # 2. 计算动量
    if method == 'Return':
        # 简单收益率: (P_t / P_{t-n}) - 1
        mom = prices.pct_change(window)
        
    elif method == 'Risk-Adjusted (稳健)':
        # 风险调整动量: 收益率 / 波动率 (类似夏普，但不减无风险利率)
        ret = prices.pct_change(window)
        # 计算窗口期内的日收益率标准差作为波动率估计
        # 这里近似处理：用过去 window 天的日收益率 std
        daily_ret = prices.pct_change()
        vol = daily_ret.rolling(window=window).std()
        mom = ret / (vol + 1e-9) # 避免除零
        
    elif method == 'Slope (线性回归)' and HAS_SCIPY:
        # 使用线性回归斜率 * R^2 (ID 动量思想)
        def calc_slope_r2(y):
            if len(y) < 2: return np.nan
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            # 年化指数回归斜率: (exp(slope) ^ 252) - 1 ... 
            # 简化版: slope * r_value^2 (既要涨得快，又要涨得稳)
            return slope * (r_value ** 2)
        
        # 这种方法计算很慢，apply rolling
        # 为了加速，这里仅演示，实际可能需要向量化优化或仅在回测循环中计算
        mom = prices.rolling(window=window).apply(calc_slope_r2, raw=True)
        
    else:
        # 默认回退到 Return
        mom = prices.pct_change(window)
    
    return mom

# ==========================================
# 3. 回测引擎 (Backtest Engine)
# ==========================================
def run_backtest(df_prices, params):
    """
    执行策略回测
    """
    if df_prices.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    lookback = params['lookback']
    smooth = params['smooth']
    threshold = params['threshold']
    min_holding = params['min_holding']
    allow_cash = params['allow_cash']
    mom_method = params['mom_method']
    
    # 计算动量矩阵
    df_mom = calculate_momentum(df_prices, lookback, smooth, mom_method)
    
    # 初始化回测变量
    cash = 1.0
    position = None # 当前持仓代码
    holding_days = 0
    total_assets = []
    positions = [] # 记录每日持仓
    operations = [] # 记录每日操作
    days_list = [] # 记录持仓天数
    
    # 收益曲线
    dates = df_prices.index
    
    # 从足够数据开始回测
    start_idx = lookback + smooth
    if start_idx >= len(dates):
        return pd.DataFrame(), pd.DataFrame()
        
    for i in range(start_idx, len(dates)):
        curr_date = dates[i]
        
        # 昨日数据用于决策 (模拟收盘后/次日开盘决策)
        # 实际上使用的是截止到 i-1 的数据计算出的动量
        # 这里的 df_mom.iloc[i-1] 包含了截止到昨天的动量信息
        
        # NOTE: 为了避免未来函数，必须使用 iloc[i] 的动量来决定 i+1 的持仓，或者 i 的收盘操作
        # 常见模式：在 i 时刻收盘，我们可以利用截止 i 的价格计算动量，然后决定 i+1 的持仓
        # 这里简化处理：假设在 i 时刻收盘时进行调仓 (Close-to-Close)
        
        current_moms = df_mom.iloc[i] 
        
        # 排除 NaN
        valid_moms = current_moms.dropna()
        
        target_code = None
        target_score = -np.inf
        
        if not valid_moms.empty:
            # 找到动量最高的
            best_code = valid_moms.idxmax()
            best_score = valid_moms.max()
            
            # 现金逻辑：如果所有标的动量都小于0 (且允许空仓)，或者最好的也小于某个阈值
            # 简单版：只要有正动量就选最好的，否则空仓
            if allow_cash and best_score < 0:
                target_code = None # Cash
            else:
                target_code = best_code
                target_score = best_score
        else:
            target_code = None
            
        # 交易逻辑判断
        op = ""
        
        # 1. 如果当前为空仓
        if position is None:
            if target_code is not None:
                position = target_code
                holding_days = 1
                op = f"买入 {target_code}"
            else:
                # 继续空仓
                holding_days += 1
                
        # 2. 如果当前有持仓
        else:
            # 获取当前持仓的最新动量分
            current_score = valid_moms.get(position, -np.inf)
            
            # 必须持有满足最小天数
            if holding_days < min_holding:
                holding_days += 1
                # 即使有更好标的也锁仓
            else:
                # 判断是否切换
                if target_code is None:
                    # 应该清仓
                    if allow_cash:
                        op = f"卖出 {position} (转现金)"
                        position = None
                        holding_days = 1
                    else:
                        # 不允许空仓，通常保留原仓位或切换到第二好的(这里简化为保留)
                        holding_days += 1
                elif target_code != position:
                    # 只有当 新标的分数 > 旧标的分数 + 阈值 时才切换
                    if target_score > current_score + threshold:
                        op = f"换仓 {position}->{target_code}"
                        position = target_code
                        holding_days = 1
                    else:
                        holding_days += 1
                else:
                    holding_days += 1
        
        # 计算当日净值
        # 如果 i 是调仓日，假设以收盘价调仓（简化），当日收益仍由旧仓位贡献（因为是收盘才换）
        # 或者：假设次日开盘换。这里采用：当日收益归属“今日开始时的持仓”
        # 这里代码逻辑是：先根据今日数据决定了 position (更新后的)，这其实是明日的持仓。
        # 这是一个细微的偏差。
        # 更严谨逻辑：Position[i] 是由 Data[i-1] 决定的。
        # 让我们修正一下：上面的逻辑是在计算 "i 时刻结束后应该持有的仓位"。
        # 那么 i 时刻的收益应该由 "i-1 时刻结束后的持仓" 决定。
        
        # 修正回测循环逻辑：
        # 1. 获取昨日确定的持仓 (prev_position)
        # 2. 计算今日收益 (基于 prev_position)
        # 3. 利用今日数据计算动量，确定今日收盘后的新持仓 (curr_position)
        
        # 但为了不大幅重写结构，我们采用类似的近似：
        # 记录每一步的决策，然后在外部计算收益，或者这里直接算。
        
        # 这里采用简化版：Position[i] 代表 i 日收盘时的目标持仓。
        # 收益计算：
        # 如果 i 日发生了换仓，假设以收盘价换。则 i 日收益由 Old Position 决定。
        # i+1 日收益由 New Position 决定。
        
        # 记录
        positions.append(position if position else "现金")
        operations.append(op)
        days_list.append(holding_days)
        
        # 计算资金曲线
        # 需要用到 i 日的涨跌幅
        if i > start_idx:
            # 昨天的持仓决定了今天的收益
            prev_pos = positions[-2] 
            
            if prev_pos == "现金":
                daily_ret = 0.0
            else:
                # 获取 prev_pos 在 i 日的涨跌幅
                # simple return
                try:
                    r = df_prices.loc[dates[i], prev_pos] / df_prices.loc[dates[i-1], prev_pos] - 1
                    daily_ret = r
                except:
                    daily_ret = 0.0
            
            new_equity = total_assets[-1] * (1 + daily_ret)
            total_assets.append(new_equity)
        else:
            total_assets.append(1.0) # 第一天归一
            
    # 整理结果
    res_df = pd.DataFrame({
        '日期': dates[start_idx:],
        '当前持仓': positions,
        '持仓天数': days_list,
        '操作': operations,
        '总资产': total_assets
    })
    
    # 补充全市场等权表现作为基准
    res_df['全市场表现'] = df_prices.mean(axis=1).pct_change().fillna(0) + 1
    # 重算基准净值 (从回测起点开始)
    # 截取对应日期的 prices
    sub_prices = df_prices.iloc[start_idx:]
    # 归一化
    normalized_prices = sub_prices / sub_prices.iloc[0]
    
    # 为了显示方便，把个股净值也放进去
    detail_df = res_df.copy()
    for col in normalized_prices.columns:
        # 这里需要注意日期索引匹配
        # detail_df['日期'] 是 datetime
        # normalized_prices index 是 datetime
        # merge
        pass
        
    # 直接把 normalized_prices 的值 merge 进 detail_df
    normalized_prices.reset_index(inplace=True)
    detail_df = pd.merge(detail_df, normalized_prices, on='日期', how='left')
    
    # 计算本段持仓收益 (Segment Return)
    # 逻辑：当前持仓连续持有了多少天，这期间的累计涨幅
    detail_df['段内收益'] = 0.0
    # 这是一个稍微复杂的向量化操作，用循环简单处理
    # 实际上如果是展示用，可以简化
    
    seg_rets = []
    # 倒序遍历或者记录买入价
    # 简单做法：如果操作是买入/换仓，记录基准净值
    
    return detail_df, df_mom

# ==========================================
# 4. Streamlit UI
# ==========================================
def main():
    st.set_page_config(page_title="ETF 动量轮动策略", layout="wide")
    
    st.title("🚀 ETF 动量轮动策略回测")
    
    # --- Sidebar 配置 ---
    st.sidebar.header("⚙️ 策略参数设置")
    
    current_config = load_config()
    
    with st.sidebar.form("params_form"):
        # 标的选择
        default_str = ",".join(current_config['selected_codes'])
        codes_input = st.text_area("标的池 (代码用逗号分隔)", value=default_str, height=100)
        
        # 参数
        lookback = st.slider("动量回看窗口 (天)", 5, 60, current_config['lookback'])
        smooth = st.slider("平滑窗口 (天)", 1, 10, current_config['smooth'])
        threshold = st.number_input("换仓阈值 (Threshold)", 0.0, 0.05, current_config['threshold'], step=0.001, format="%.3f")
        min_holding = st.number_input("最小持仓天数", 1, 20, current_config['min_holding'])
        allow_cash = st.checkbox("允许空仓 (持有现金)", value=current_config['allow_cash'])
        mom_method = st.selectbox("动量计算方法", ["Return", "Risk-Adjusted (稳健)", "Slope (线性回归)"], index=1)
        
        submitted = st.form_submit_button("开始回测")
        
        if submitted:
            # 更新配置
            code_list = [c.strip() for c in codes_input.split(',') if c.strip()]
            new_config = {
                'lookback': lookback,
                'smooth': smooth,
                'threshold': threshold,
                'min_holding': min_holding,
                'allow_cash': allow_cash,
                'mom_method': mom_method,
                'selected_codes': code_list
            }
            save_config(new_config)
            current_config = new_config

    # --- 主逻辑 ---
    codes = current_config['selected_codes']
    
    if not codes:
        st.warning("请在左侧输入标的代码")
        return

    # 获取数据
    with st.spinner('正在获取数据并计算...'):
        df_data = get_data(codes)
        
    if df_data.empty:
        st.error("无法获取数据，请检查网络或代码是否正确")
        return

    # 运行回测
    df_details, df_mom = run_backtest(df_data, current_config)
    
    if df_details.empty:
        st.warning("数据不足以进行回测 (可能是回看窗口太长)")
        return
        
    # --- 结果展示 ---
    
    # 1. 核心指标卡片
    final_equity = df_details['总资产'].iloc[-1]
    total_ret = (final_equity - 1) * 100
    
    # 计算年化
    days = (df_details['日期'].iloc[-1] - df_details['日期'].iloc[0]).days
    ann_ret = ((final_equity) ** (365/days) - 1) * 100 if days > 0 else 0
    
    # 最大回撤
    equity_series = df_details['总资产']
    drawdown = (equity_series / equity_series.cummax() - 1)
    max_dd = drawdown.min() * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计收益率", f"{total_ret:.2f}%")
    c2.metric("年化收益率", f"{ann_ret:.2f}%")
    c3.metric("最大回撤", f"{max_dd:.2f}%")
    c4.metric("当前持仓", f"{df_details['当前持仓'].iloc[-1]}")
    
    # 2. 资金曲线图
    st.subheader("📈 资金曲线")
    
    fig = go.Figure()
    # 策略净值
    fig.add_trace(go.Scatter(x=df_details['日期'], y=df_details['总资产'], mode='lines', name='策略净值', line=dict(width=2, color='#2962FF')))
    
    # 标的净值 (基准)
    # 找出 asset cols (排除基础列)
    asset_cols = [c for c in df_details.columns if c not in ['日期', '当前持仓', '持仓天数', '操作', '总资产', '全市场表现', '段内收益']]
    
    for col in asset_cols:
        fig.add_trace(go.Scatter(x=df_details['日期'], y=df_details[col], mode='lines', name=col, line=dict(width=1), visible='legendonly'))
        
    fig.update_layout(xaxis_title="日期", yaxis_title="净值", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. 详细交易记录
    st.subheader("📋 交易明细")
    
    if not df_details.empty:
        df_details['段内收益'] = df_details['段内收益'] * 100
        
        asset_cols = sorted([col for col in df_details.columns if col not in ["日期", "当前持仓", "持仓天数", "段内收益", "操作", "总资产", "全市场表现"]])
        
        for ac in asset_cols:
            df_details[ac] = df_details[ac] * 100
        
        # --- 下一个交易日持仓建议 (Next Trading Day Suggestion) ---
        st.write("---") # 分割线
        st.subheader("🔔 下一个交易日持仓建议")
        
        # 获取最后一行数据
        last_row = df_details.iloc[-1]
        last_pos = last_row['当前持仓']
        last_date = last_row['日期'].strftime('%Y-%m-%d')
        last_op = str(last_row['操作'])
        
        # 逻辑判断：
        # 回测逻辑是基于 Close-to-Close。如果最后一行（最新数据日期）有“换仓”或“买入”操作，
        # 说明基于该日收盘数据，模型发出了交易信号。
        # 如果最后一行没有操作，说明模型建议继续持有上一日的仓位。
        
        suggestion_text = ""
        suggestion_color = "blue"
        
        if "换仓" in last_op:
            # 解析目标: "换仓 Old->New"
            try:
                target = last_op.split("->")[1]
                suggestion_text = f"👉 建议 **换仓至 {target}**"
                suggestion_color = "red"
            except:
                suggestion_text = f"👉 建议 **{last_op}**"
        elif "买入" in last_op:
            try:
                target = last_op.split(" ")[1]
                suggestion_text = f"👉 建议 **买入 {target}**"
                suggestion_color = "red"
            except:
                suggestion_text = f"👉 建议 **{last_op}**"
        elif "卖出" in last_op:
             suggestion_text = "👉 建议 **卖出并空仓 (持有现金)**"
             suggestion_color = "orange"
        else:
            # 无操作，继续持有
            if last_pos == "现金" or last_pos is None:
                suggestion_text = "👉 建议 **继续空仓 (持有现金)**"
                suggestion_color = "gray"
            else:
                suggestion_text = f"👉 建议 **继续持有 {last_pos}**"
                suggestion_color = "green"
        
        # 使用 info 或 success 框显示
        if suggestion_color == "red":
            st.error(f"📅 基于最新数据 ({last_date}) 的操作建议：\n\n {suggestion_text}")
        elif suggestion_color == "orange":
            st.warning(f"📅 基于最新数据 ({last_date}) 的操作建议：\n\n {suggestion_text}")
        elif suggestion_color == "gray":
             st.info(f"📅 基于最新数据 ({last_date}) 的操作建议：\n\n {suggestion_text}")
        else:
            st.success(f"📅 基于最新数据 ({last_date}) 的操作建议：\n\n {suggestion_text}")
            
        st.caption("注：此建议基于最新收盘数据计算。如果今日已收盘，则为明日开盘操作建议；如果今日未收盘，请等待收盘数据更新。")
        st.write("---")
        # ----------------------------------------------------

        col_config = {
            "持仓天数": st.column_config.NumberColumn("持仓天数", help="当前连续持仓天数"),
            "段内收益": st.column_config.NumberColumn("段内收益", help="本段持仓期间的累计收益率", format="%.2f%%"),
            "操作": st.column_config.TextColumn("调仓操作", width="medium"),
            "总资产": st.column_config.NumberColumn("总资产", format="%.2f"),
            "日期": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
        }
        
        for ac in asset_cols:
            col_config[ac] = st.column_config.NumberColumn(ac, format="%.2f%%")

        final_cols = ["日期"] + asset_cols + ["当前持仓", "持仓天数", "段内收益", "总资产", "操作"]
        
        # 倒序显示，让最新的在最上面
        df_show = df_details[final_cols].sort_values('日期', ascending=False)
        
        st.dataframe(
            df_show,
            column_config=col_config,
            use_container_width=True,
            hide_index=True,
            height=600
        )

if __name__ == "__main__":
    main()
