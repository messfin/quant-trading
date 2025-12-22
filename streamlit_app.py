import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
import base64
import statsmodels.api as sm
import os

# Import local modules
from strategies import STRATEGIES, PairTrading
from utils import calculate_stats, get_base_css
from ai_reporting import AIReportingLayer

# --- Page Config ---
st.set_page_config(page_title="ZMTech Lab v2", page_icon="🧬", layout="wide")
st.markdown(get_base_css(), unsafe_allow_html=True)

# --- Initialize AI Layer ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
ai_layer = AIReportingLayer(GOOGLE_API_KEY)

# --- Helpers ---
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(1)
        return df
    except:
        return None

def main():
    st.sidebar.title("🧬 ZMTech Quant Portal")
    
    if GOOGLE_API_KEY:
        st.sidebar.success("✅ AI connected: Gemini 2.x")
    else:
        st.sidebar.warning("⚠️ Gemini API Key Missing")

    lab_mode = st.sidebar.radio("Select Research Lab", 
                               ["🚀 Trading Strategy Lab", 
                                "⚖️ Statistical Arbitrage Lab", 
                                "🎲 Quantitative Risk Lab", 
                                "🌎 Macro Research Lab"])
    
    st.sidebar.divider()

    if lab_mode == "🚀 Trading Strategy Lab":
        render_trading_lab()
    elif lab_mode == "⚖️ Statistical Arbitrage Lab":
        render_pairs_lab()
    elif lab_mode == "🎲 Quantitative Risk Lab":
        render_risk_lab()
    elif lab_mode == "🌎 Macro Research Lab":
        render_macro_research_lab()

# ==========================================
# LAB 1: TRADING STRATEGY
# ==========================================
def render_trading_lab():
    st.header("🚀 Algo Trading Strategy Lab")
    
    strategy_name = st.sidebar.selectbox("Select Strategy", options=list(STRATEGIES.keys()))
    strategy = STRATEGIES[strategy_name]
    
    ticker = st.sidebar.text_input("Stock Ticker", value="NVDA").upper()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*3))
    with col2:
        ed_date = st.sidebar.date_input("End Date", value=datetime.now())
    
    capital = st.sidebar.number_input("Initial Capital ($)", value=10000)
    position_size = st.sidebar.number_input("Shares per Trade", value=100)
    
    # Params
    st.sidebar.subheader("Strategy Parameters")
    params = {}
    if strategy_name == "Awesome Oscillator":
        params['ma1'] = st.sidebar.slider("Short MA Period", 2, 20, 5)
        params['ma2'] = st.sidebar.slider("Long MA Period", 20, 100, 34)
    elif strategy_name == "MACD Oscillator":
        params['ma1'] = st.sidebar.slider("Fast Period", 5, 20, 12)
        params['ma2'] = st.sidebar.slider("Slow Period", 20, 50, 26)
    elif strategy_name == "Bollinger Bands Pattern":
        params['window'] = st.sidebar.slider("Window", 10, 50, 20)
    elif strategy_name == "RSI Strategy":
        params['n'] = st.sidebar.slider("RSI Period", 5, 30, 14)
    elif strategy_name == "Parabolic SAR":
        params['initial_af'] = st.sidebar.number_input("Initial AF", value=0.02)
        params['step_af'] = st.sidebar.number_input("Step AF", value=0.02)
    elif strategy_name == "Dual Thrust":
        params['rg'] = st.sidebar.slider("Range Lags", 1, 20, 5)
        params['k'] = st.sidebar.slider("K Factor", 0.1, 1.0, 0.5)

    if st.sidebar.button("📊 GENERATE ANALYSIS"):
        st.session_state.pop('ai_analysis', None)
        df = load_data(ticker, st_date, ed_date)
        if df is not None:
            with st.spinner("Backtesting..."):
                signals_df, portfolio_df = strategy.run_backtest(df, capital=capital, position_size=position_size, **params)
                stats = calculate_stats(portfolio_df, signals_df, st_date, ed_date, capital0=capital)
                st.session_state['trading_results'] = {'signals': signals_df, 'portfolio': portfolio_df, 'stats': stats, 'ticker': ticker, 'strategy': strategy_name}

    if 'trading_results' in st.session_state:
        res = st.session_state['trading_results']
        display_results(res['signals'], res['portfolio'], res['stats'], res['ticker'], res['strategy'], capital)
    else:
        st.info("Configure parameters and click Generate.")

# ==========================================
# LAB 2: STATISTICAL ARBITRAGE (PAIRS)
# ==========================================
def render_pairs_lab():
    st.header("⚖️ Statistical Arbitrage Lab")
    st.markdown("Pairs trading finds cointegrated assets and bets on mean reversion.")
    
    col1, col2 = st.sidebar.columns(2)
    with col1: t1 = st.sidebar.text_input("Ticker A", value="NVDA").upper()
    with col2: t2 = st.sidebar.text_input("Ticker B", value="AMD").upper()
    
    st_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
    ed_date = st.sidebar.date_input("End Date", value=datetime.now())
    bandwidth = st.sidebar.slider("Rolling Window (Days)", 60, 500, 250)
    
    if st.sidebar.button("🔬 RUN PAIR ANALYSIS"):
        df1 = load_data(t1, st_date, ed_date)
        df2 = load_data(t2, st_date, ed_date)
        
        if df1 is not None and df2 is not None:
            with st.spinner("Testing Cointegration..."):
                pair_strat = PairTrading()
                signals, portfolio = pair_strat.run_backtest(df1, df2, ticker1=t1, ticker2=t2, capital=20000, bandwidth=bandwidth)
                stats = calculate_stats(portfolio, signals, st_date, ed_date, capital0=20000)
                st.session_state['pair_results'] = {'signals': signals, 'portfolio': portfolio, 'stats': stats, 't1': t1, 't2': t2}

    if 'pair_results' in st.session_state:
        res = st.session_state['pair_results']
        st.subheader(f"Results: {res['t1']} vs {res['t2']}")
        
        # Performance
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAGR", f"{res['stats'].get('CAGR', 0):.2%}")
        m2.metric("Sharpe", f"{res['stats'].get('Sharpe Ratio', 0):.2f}")
        m3.metric("Max DD", f"{res['stats'].get('Max Drawdown', 0):.2%}")
        m4.metric("Total Return", f"{res['stats'].get('Total Return', 0):.2%}")
        
        tab1, tab2, tab3 = st.tabs(["💧 Equity Curve", "📋 Analysis & Data", "🤖 AI Research"])

        with tab1:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=res['portfolio'].index, y=res['portfolio']['total asset'],
                fill='tozeroy', name="Pair Portfolio",
                line=dict(color="#00bcd4", width=3),
                fillcolor="rgba(0, 188, 212, 0.1)"
            ))
            fig.update_layout(
                template="plotly_dark",
                height=500,
                title="Statistical Arbitrage Equity Curve",
                hovermode="x unified"
            )
            st.plotly_chart(fig, width="stretch")
        
        with tab2:
            st.dataframe(res['signals'].tail(100))

        with tab3:
            st.subheader("🤖 ZMTech AI Intelligence: Statistical Arbitrage")
            if st.button("🧠 GENERATE PAIR INTELLIGENCE REPORT"):
                with st.spinner("Analyzing Cointegration..."):
                    pair_ticker = f"{res['t1']} vs {res['t2']}"
                    # Provide recent spreads/signals summary
                    recent_signals = res['signals'].loc[res['signals']['signals1'] != 0].tail(10).to_string()
                    analysis = ai_layer.generate_analysis("Pair Trading", pair_ticker, res['stats'], recent_signals)
                    st.session_state['pair_ai_analysis'] = analysis
            
            if 'pair_ai_analysis' in st.session_state:
                st.markdown(st.session_state['pair_ai_analysis'])
                
                pdf_data = ai_layer.create_pdf_report("Pair Trading", f"{res['t1']}_{res['t2']}", res['stats'], st.session_state['pair_ai_analysis'])
                if pdf_data:
                    b64 = base64.b64encode(pdf_data).decode('utf-8')
                    display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800"></iframe>'
                    st.markdown(display, unsafe_allow_html=True)

# ==========================================
# LAB 3: QUANTITATIVE RISK (MONTE CARLO / VIX)
# ==========================================
def render_risk_lab():
    st.header("🎲 Quantitative Risk Lab")
    
    sub_mode = st.sidebar.selectbox("Risk Tool", ["Monte Carlo Prediction", "VIX Volatility Analysis"])
    
    if sub_mode == "Monte Carlo Prediction":
        ticker = st.sidebar.text_input("Ticker", value="SPY").upper()
        sims = st.sidebar.slider("Simulations", 10, 500, 100)
        horizon = st.sidebar.slider("Forecast Days", 10, 250, 60)
        
        if st.sidebar.button("🎲 RUN SIMULATION"):
            df = load_data(ticker, datetime.now()-timedelta(days=730), datetime.now())
            if df is not None:
                returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
                mu = returns.mean()
                sigma = returns.std()
                last_price = float(df['Close'].iloc[-1])
                
                sim_results = []
                for _ in range(sims):
                    prices = [last_price]
                    for _ in range(horizon):
                        drift = (mu - 0.5 * sigma**2)
                        diffusion = sigma * np.random.normal()
                        prices.append(prices[-1] * np.exp(drift + diffusion))
                    sim_results.append(prices)
                st.session_state['mc_results'] = {'sims': sim_results, 'ticker': ticker}

        if 'mc_results' in st.session_state:
            res = st.session_state['mc_results']
            import plotly.graph_objects as go
            fig = go.Figure()
            
            # Show a subset of simulations for performance if too many
            sims_to_plot = res['sims'][:100] 
            for i, s in enumerate(sims_to_plot):
                fig.add_trace(go.Scatter(y=s, mode='lines', line=dict(width=1, color='#00bcd4'), opacity=0.1, showlegend=False))
            
            mean_path = np.mean(res['sims'], axis=0)
            fig.add_trace(go.Scatter(y=mean_path, mode='lines', name='Expected Mean Path', line=dict(width=4, color='white')))
            
            fig.update_layout(
                template="plotly_dark",
                height=500,
                title=f"Monte Carlo Risk Projection: {res['ticker']}",
                xaxis_title="Days Forward",
                yaxis_title="Price ($)",
                hovermode="x"
            )
            st.plotly_chart(fig, width="stretch")
            
            final_prices = [s[-1] for s in res['sims']]
            prob_up = sum(1 for p in final_prices if p > res['sims'][0][0]) / len(res['sims'])
            st.success(f"Probability of Price Increase: **{prob_up:.2%}**")

    elif sub_mode == "VIX Volatility Analysis":
        st.subheader("VIX Analysis & Fear Index Indicators")
        if st.button("📈 FETCH VIX DATA"):
            vix = load_data("^VIX", datetime.now()-timedelta(days=1000), datetime.now())
            spy = load_data("SPY", datetime.now()-timedelta(days=1000), datetime.now())
            if vix is not None and spy is not None:
                st.session_state['vix_data'] = {'vix': vix, 'spy': spy}
        
        if 'vix_data' in st.session_state:
            v_data = st.session_state['vix_data']
            vix_df = v_data['vix']
            spy_df = v_data['spy']
            
            # Align data
            common_idx = vix_df.index.intersection(spy_df.index)
            vix_plot = vix_df.loc[common_idx]
            spy_plot = spy_df.loc[common_idx]

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # SPY Price
            fig.add_trace(
                go.Scatter(x=spy_plot.index, y=spy_plot['Close'], name="SPY Price", line=dict(color="#00bcd4", width=2)),
                secondary_y=False,
            )

            # VIX Index
            fig.add_trace(
                go.Scatter(x=vix_plot.index, y=vix_plot['Close'], name="VIX Index", line=dict(color="#ff9800", width=1.5), opacity=0.7),
                secondary_y=True,
            )

            # Add Thresholds
            fig.add_hline(y=20, line_dash="dash", line_color="yellow", annotation_text="Caution (20)", secondary_y=True)
            fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Panic (30)", secondary_y=True)

            fig.update_layout(
                title_text="Market Fear Gauge: SPY vs VIX",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            fig.update_yaxes(title_text="SPY Price ($)", secondary_y=False, gridcolor="#333")
            fig.update_yaxes(title_text="VIX Level", secondary_y=True, showgrid=False)

            st.plotly_chart(fig, width="stretch")
            
            # Additional Insight
            corr = spy_plot['Close'].pct_change().corr(vix_plot['Close'].pct_change())
            st.info(f"📊 30-Day Correlation (Returns): **{corr:.2f}** (Typically strongly negative)")

# ==========================================
# LAB 4: MACRO RESEARCH LAB (COMMODITIES/MAPS)
# ==========================================
def render_macro_research_lab():
    st.header("🌎 Macro Research Lab")
    
    macro_tool = st.sidebar.selectbox("Research Tool", ["Global Production Bubble Map", "Commodity-Currency Correlator"])
    
    if macro_tool == "Global Production Bubble Map":
        st.subheader("Global Iron Ore Production")
        csv_path = "Ore Money project/iron ore production/iron ore production bubble map.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            fig = px.scatter_geo(df, lat='latitude', lon='longitude', size='iron ore production',
                                 hover_name='region', color='iron ore production',
                                 projection="natural earth", title="Global Iron Ore Production (Wikipedia Data)",
                                 color_continuous_scale=px.colors.sequential.YlOrRd)
            fig.update_layout(template="plotly_dark", height=600)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df)
        else:
            st.error("Iron Ore data file not found.")

    elif macro_tool == "Commodity-Currency Correlator":
        st.subheader("Oil vs Canadian Dollar Correlation")
        csv_path = "Oil Money project/data/wcs crude cadaud.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            st.line_chart(df[['cad', 'wcs']])
            
            # Regression with cleaning
            valid_mask = np.isfinite(df['wcs']) & np.isfinite(df['cad'])
            df_clean = df[valid_mask].dropna()
            
            if not df_clean.empty:
                y = df_clean['cad']
                x = sm.add_constant(df_clean['wcs'])
                model = sm.OLS(y, x).fit()
                st.markdown(f"**Regression R-Squared:** `{model.rsquared:.4f}`")
            else:
                st.error("Correlation error: Not enough finite data points available.")
            
            import plotly.graph_objects as go
            fig_macro = go.Figure()
            fig_macro.add_trace(go.Scatter(
                x=df['wcs'], y=df['cad'], mode='markers',
                name='Data Points', marker=dict(color='#00bcd4', opacity=0.4)
            ))
            fig_macro.add_trace(go.Scatter(
                x=df['wcs'], y=model.predict(x), mode='lines',
                name='Reg Line', line=dict(color='red', width=2)
            ))
            fig_macro.update_layout(
                template="plotly_dark",
                title="WCS Crude vs CAD Correlation",
                xaxis_title="WCS Price",
                yaxis_title="CAD/USD Rate"
            )
            st.plotly_chart(fig_macro, width="stretch")
        else:
            st.error("Oil Money data file not found.")

# --- Shared UI ---
def display_results(signals_df, portfolio_df, stats, ticker, strategy_name, capital):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.subheader(f"Strategy: {strategy_name} on {ticker}")
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CAGR", f"{stats.get('CAGR', 0):.2%}")
    m2.metric("Sharpe", f"{stats.get('Sharpe Ratio', 0):.2f}")
    m3.metric("Max DD", f"{stats.get('Max Drawdown', 0):.2%}")
    m4.metric("Total Return", f"{stats.get('Total Return', 0):.2%}")

    tab1, tab2, tab3 = st.tabs(["💡 TradingView Chart", "💰 Performance Lab", "🤖 AI Research"])
    
    with tab1:
        # TradingView Style Candlestick + Signals
        fig = go.Figure()
        
        # Add Candlesticks (or Line if data is simple)
        if all(col in signals_df.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig.add_trace(go.Candlestick(
                x=signals_df.index,
                open=signals_df['Open'],
                high=signals_df['High'],
                low=signals_df['Low'],
                close=signals_df['Close'],
                name="Price Action",
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=signals_df.index, y=signals_df['Close'],
                name="Close Price", line=dict(color="#2962FF", width=2)
            ))

        # Add Signals
        buys = signals_df[signals_df['signals'] > 0]
        sells = signals_df[signals_df['signals'] < 0]

        fig.add_trace(go.Scatter(
            x=buys.index, y=buys['Close'],
            mode='markers', name='BUY Signal',
            marker=dict(symbol='triangle-up', size=12, color='#00ff00', line=dict(width=1, color='white'))
        ))

        fig.add_trace(go.Scatter(
            x=sells.index, y=sells['Close'],
            mode='markers', name='SELL Signal',
            marker=dict(symbol='triangle-down', size=12, color='#ff0000', line=dict(width=1, color='white'))
        ))

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=600,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width="stretch")
        
    with tab2:
        # Equity Curve Lab
        fig_asset = go.Figure()
        fig_asset.add_trace(go.Scatter(
            x=portfolio_df.index, y=portfolio_df['total asset'],
            fill='tozeroy', name="Equity Curve",
            line=dict(color="#00bcd4", width=3),
            fillcolor="rgba(0, 188, 212, 0.1)"
        ))
        
        # Benchmarking (Optional: Initial Capital line)
        fig_asset.add_hline(y=capital, line_dash="dash", line_color="gray", annotation_text="Initial Capital")

        fig_asset.update_layout(
            template="plotly_dark",
            height=500,
            title="Portfolio Equity Expansion",
            hovermode="x unified"
        )
        st.plotly_chart(fig_asset, width="stretch")

    with tab3:
        st.subheader("🤖 Strategy Intelligence Report")
        if st.button("🧠 GENERATE FULL REPORT"):
            with st.spinner("Gemini is crunching numbers..."):
                recent = signals_df.loc[signals_df['signals'] != 0].tail(10).to_string()
                analysis = ai_layer.generate_analysis(strategy_name, ticker, stats, recent)
                st.session_state['ai_analysis'] = analysis
        
        if 'ai_analysis' in st.session_state:
            st.markdown(st.session_state['ai_analysis'])
            
            pdf_data = ai_layer.create_pdf_report(strategy_name, ticker, stats, st.session_state['ai_analysis'])
            if pdf_data:
                b64 = base64.b64encode(pdf_data).decode('utf-8')
                display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800"></iframe>'
                st.markdown(display, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
