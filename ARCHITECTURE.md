# ZMTech Quant Portal: High-Level Architecture

## Overview
The ZMTech Quant Portal is an institutional-grade quantitative trading and research platform. It integrates technical analysis, statistical arbitrage, risk modeling, and macro-economic research into a unified interactive dashboard powered by Streamlit and Gemini AI.

---

## 🏗 System Design

### 1. Presentation Layer (`streamlit_app.py`)
The central hub of the application. It orchestrates the user interface across four specialized research labs:
- **🚀 Trading Strategy Lab**: Interactive TradingView-style backtesting.
- **⚖️ Statistical Arbitrage Lab**: Rolling cointegration and pair trading analysis.
- **🎲 Quantitative Risk Lab**: Monte Carlo price forecasting and VIX sentiment analysis.
- **🌎 Macro Research Lab**: Global production mapping and commodity-currency correlation.

### 2. Strategy Engine (`strategies.py`)
A modular framework defining trading logic. All strategies inherit from `StrategyBase` and implement `generate_signals` and `run_backtest`.
- **Implemented Strategies**: Awesome Oscillator, MACD, Heikin-Ashi, Bollinger Bands, RSI Pattern, Shooting Star, Parabolic SAR, London Breakout, Dual Thrust, and Pair Trading.

### 3. AI Intelligence Layer (`ai_reporting.py`)
Leverages Google Gemini (2.x/1.5) to provide deep analytical insights.
- **Features**: Automatic sentiment tagging, strategy efficiency evaluation, and institutional-grade PDF report generation.
- **Resiliency**: Implements multi-model fallback (Flash -> Pro) and quota-aware error handling.

### 4. Quantitative Utils (`utils.py`)
Core mathematics for performance measurement.
- **Metrics**: CAGR, Sharpe Ratio, Maximum Drawdown, Volatility, and Total Return.
- **Robustness**: Handles complex numbers, NaNs, and infinite values in financial time-series.

---

## 📊 Data & Research Modules

### Macro Research Assets
- **Ore Money Project**: Global Iron Ore production tracking (`iron ore production bubble map.csv`).
- **Oil Money Project**: Energy markets and currency impact analysis (`wcs crude cadaud.csv`).

### Risk Framework
- **Monte Carlo Engine**: Geometric Brownian Motion (GBM) for price projection.
- **Fear Gauge**: Real-time SPY/VIX correlation tracking using dual-axis Plotly visualizations.

---

## 🛠 Technology Stack
- **Dashboard**: Streamlit (Premium Dark Theme)
- **Visuals**: Plotly (TradingView Aesthetic), Matplotlib
- **Data**: yfinance, Pandas, NumPy
- **Statistics**: Statsmodels (OLS, ADF Testing), SciPy
- **AI**: Google Generative AI (Gemini)
- **Reporting**: FPDF2, Python-Docx

---

## 🚦 Workflow
1. **Data Ingestion**: Fetch market data via `yfinance` with MultiIndex flattening.
2. **Signal Generation**: Execute strategy-specific logic in `strategies.py`.
3. **Performance Backtest**: Calculate equity curves and drawdown in `StrategyBase`.
4. **AI Analysis**: Send stats and signals to `AIReportingLayer` for Gemini-powered synthesis.
5. **Visualization**: Render interactive Plotly charts in the Streamlit dashboard.
6. **Report Export**: Generate and preview professional PDF reports within the browser.
