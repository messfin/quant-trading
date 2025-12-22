import pandas as pd
import numpy as np
import yfinance as yf
import scipy.integrate
import scipy.stats
import matplotlib.pyplot as plt
import streamlit as st

def mdd(series):
    """Calculate Maximum Drawdown."""
    minimum = 0
    for i in range(1, len(series)):
        prev_max = max(series[:i])
        if prev_max > 0:
            drawdown = (series[i] / prev_max - 1)
            if minimum > drawdown:
                minimum = drawdown
    return minimum

def omega(risk_free, degree_of_freedom, maximum, minimum):
    """Calculate Omega Ratio."""
    try:
        y = scipy.integrate.quad(lambda g: 1 - scipy.stats.t.cdf(g, degree_of_freedom), risk_free, maximum)
        x = scipy.integrate.quad(lambda g: scipy.stats.t.cdf(g, degree_of_freedom), minimum, risk_free)
        if x[0] == 0: return 0
        return y[0] / x[0]
    except:
        return 0

def sortino(risk_free, degree_of_freedom, growth_rate, minimum):
    """Calculate Sortino Ratio."""
    try:
        v = np.sqrt(np.abs(scipy.integrate.quad(lambda g: ((risk_free - g)**2) * scipy.stats.t.pdf(g, degree_of_freedom), risk_free, minimum)))
        if v[0] == 0: return 0
        return (growth_rate - risk_free) / v[0]
    except:
        return 0

def calculate_stats(portfolio_df, trading_signals, stdate, eddate, capital0=10000):
    """Comprehensive performance statistics."""
    if portfolio_df.empty or 'total asset' not in portfolio_df.columns:
        return pd.DataFrame()
        
    stats = {}
    
    # Basic returns
    returns = portfolio_df['return'].dropna()
    if len(returns) == 0:
        return pd.DataFrame()
        
    maximum = np.max(returns)
    minimum = np.min(returns)
    
    # Portfolio Growth Rate (Annualized CAGR)
    days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    if days <= 0: days = 1
    
    final_val = float(portfolio_df['total asset'].iloc[-1])
    total_ret = (final_val / capital0) - 1
    
    # CAGR handle negative/complex
    if final_val > 0:
        growth_rate = (final_val / capital0)**(365.0/days) - 1
    else:
        growth_rate = -1.0
        
    # Ensure any numerical artifacts are real
    if hasattr(growth_rate, "real"): growth_rate = float(growth_rate.real)
    
    std = returns.std() * np.sqrt(252) # Annualized Vol
    
    # Benchmark
    try:
        benchmark = yf.download('^GSPC', start=stdate, end=eddate, progress=False, auto_adjust=True)
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.get_level_values(0)
        return_of_benchmark = float(benchmark['Close'].iloc[-1] / benchmark['Open'].iloc[0] - 1)
        rate_of_benchmark = (return_of_benchmark + 1)**(365.0/days) - 1
    except:
        return_of_benchmark = 0
        rate_of_benchmark = 0

    stats['CAGR'] = growth_rate
    stats['Total Return'] = total_ret
    stats['Benchmark Return'] = return_of_benchmark
    stats['Sharpe Ratio'] = (growth_rate - rate_of_benchmark) / std if std != 0 else 0
    stats['Max Drawdown'] = mdd(portfolio_df['total asset'])
    stats['Calmar Ratio'] = growth_rate / stats['Max Drawdown'] if stats['Max Drawdown'] != 0 else 0
    stats['Omega Ratio'] = float(omega(rate_of_benchmark/252, len(trading_signals), maximum, minimum))
    stats['Sortino Ratio'] = float(sortino(rate_of_benchmark/252, len(trading_signals), growth_rate/252, minimum))
    
    # Trade counts
    if 'signals' in trading_signals.columns:
        stats['Longs'] = int(trading_signals['signals'].loc[trading_signals['signals'] == 1].count())
        stats['Shorts'] = int(trading_signals['signals'].loc[trading_signals['signals'] < 0].count())
        stats['Total Trades'] = stats['Longs'] + stats['Shorts']
        
        if stats['Total Trades'] > 0:
            stats['Profit per Trade'] = (portfolio_df['total asset'].iloc[-1] - capital0) / stats['Total Trades']
        else:
            stats['Profit per Trade'] = 0
            
    return pd.Series(stats)

def get_base_css():
    return """
    <style>
        /* Force dark theme aggressively */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117 !important;
            color: #ffffff !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
            background-color: #1a1a1a !important;
            border-right: 1px solid #30363d !important;
        }
        
        /* Headers and Titles */
        .stTitle, h1, h2, h3 {
            color: #00bcd4 !important;
            font-weight: 700 !important;
        }
        
        .stTitle {
            background: linear-gradient(90deg, #00bcd4, #00acc1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            padding-bottom: 20px;
        }

        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #00bcd4 !important;
        }
        
        [data-testid="stMetric"] {
            background-color: #1a1a1a !important;
            padding: 20px !important;
            border-radius: 12px !important;
            border: 1px solid #30363d !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }

        /* Buttons */
        .stButton > button {
            width: 100% !important;
            border-radius: 8px !important;
            height: 3.5em !important;
            background: linear-gradient(90deg, #00bcd4, #00acc1) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(0, 188, 212, 0.4) !important;
            color: white !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px !important;
            background-color: transparent !important;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px !important;
            background-color: transparent !important;
            padding: 10px 20px !important;
            color: #8892b0 !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 188, 212, 0.1) !important;
            border-bottom: 2px solid #00bcd4 !important;
            color: #00bcd4 !important;
        }

        /* Input boxes, Select boxes */
        [data-testid="stSidebar"] [data-baseweb="select"] {
            background-color: #0e1117 !important;
        }
        
        div.stNumberInput input, div.stTextInput input, div.stDateInput input {
            background-color: #0e1117 !important;
            color: white !important;
            border: 1px solid #30363d !important;
        }

        /* Success/Info boxes */
        .stAlert {
            border-radius: 10px !important;
            background-color: #1a1a1a !important;
            border: 1px solid #30363d !important;
            color: #ffffff !important;
        }
    </style>
    """
