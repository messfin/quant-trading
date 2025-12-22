import pandas as pd
import numpy as np
import yfinance as yf
import copy

class StrategyBase:
    def __init__(self, name):
        self.name = name
    
    def generate_signals(self, df, **kwargs):
        raise NotImplementedError
    
    def run_backtest(self, df, capital=10000, position_size=100, **kwargs):
        # Flatten MultiIndex columns if they exist (common with yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            elif 'Close' in df.columns.get_level_values(1):
                df.columns = df.columns.get_level_values(1)
            
        signals = self.generate_signals(df, **kwargs)
        
        # Consistent portfolio logic
        port = pd.DataFrame(index=signals.index)
        port['Close'] = signals['Close']
        
        # cumsum check
        if 'cumsum' not in signals.columns:
            signals['cumsum'] = signals['signals'].cumsum()
            
        port['holdings'] = signals['cumsum'] * port['Close'] * position_size
        port['cash'] = capital - (signals['signals'] * port['Close'] * position_size).cumsum()
        port['total asset'] = port['holdings'] + port['cash']
        port['return'] = port['total asset'].pct_change()
        port['signals'] = signals['signals']
        
        return signals, port

class AwesomeOscillator(StrategyBase):
    def __init__(self):
        super().__init__("Awesome Oscillator vs MACD")
        
    def generate_signals(self, df, ma1=5, ma2=34):
        signals = df.copy()
        
        # MACD part
        signals['macd ma1'] = signals['Close'].ewm(span=ma1).mean()    
        signals['macd ma2'] = signals['Close'].ewm(span=ma2).mean()
        signals['macd positions'] = 0
        signals.iloc[ma1:, signals.columns.get_loc('macd positions')] = np.where(
            signals['macd ma1'][ma1:] >= signals['macd ma2'][ma1:], 1, 0
        )
        signals['macd signals'] = signals['macd positions'].diff()
        signals['macd oscillator'] = signals['macd ma1'] - signals['macd ma2']
        
        # Awesome part
        signals['awesome ma1'] = ((signals['High'] + signals['Low']) / 2).rolling(window=5).mean()
        signals['awesome ma2'] = ((signals['High'] + signals['Low']) / 2).rolling(window=34).mean()
        signals['awesome oscillator'] = signals['awesome ma1'] - signals['awesome ma2']
        
        # Signal loop optimized
        awesome_signals = np.zeros(len(signals))
        current_cumsum = 0
        opens = signals['Open'].values
        closes = signals['Close'].values
        ao = signals['awesome oscillator'].values
        ama1 = signals['awesome ma1'].values
        ama2 = signals['awesome ma2'].values

        for i in range(2, len(signals)):
            sig = 0
            # Saucer signals
            if (opens[i] > closes[i] and opens[i-1] < closes[i-1] and opens[i-2] < closes[i-2] and
                ao[i-1] > ao[i-2] and ao[i-1] < 0 and ao[i] < 0):
                sig = 1
            elif (opens[i] < closes[i] and opens[i-1] > closes[i-1] and opens[i-2] > closes[i-2] and
                  ao[i-1] < ao[i-2] and ao[i-1] > 0 and ao[i] > 0):
                sig = -1
            
            # MA crossover
            if ama1[i] > ama2[i]:
                sig = 1
                if current_cumsum + sig > 1: sig = 0
            elif ama1[i] < ama2[i]:
                sig = -1
                if current_cumsum + sig < 0: sig = 0
            
            awesome_signals[i] = sig
            current_cumsum += sig
            
        signals['signals'] = awesome_signals # For AO
        signals['awesome signals'] = awesome_signals
        signals['cumsum'] = signals['signals'].cumsum()
        return signals

class MACDOscillator(StrategyBase):
    def __init__(self):
        super().__init__("MACD Oscillator")
        
    def generate_signals(self, df, ma1=12, ma2=26):
        signals = df.copy()
        signals['ma1'] = signals['Close'].rolling(window=ma1).mean()
        signals['ma2'] = signals['Close'].rolling(window=ma2).mean()
        signals['positions'] = 0
        signals.iloc[ma1:, signals.columns.get_loc('positions')] = np.where(
            signals['ma1'][ma1:] >= signals['ma2'][ma1:], 1, 0
        )
        signals['signals'] = signals['positions'].diff()
        signals['oscillator'] = signals['ma1'] - signals['ma2']
        return signals

class HeikinAshiStrategy(StrategyBase):
    def __init__(self):
        super().__init__("Heikin-Ashi")
        
    def generate_signals(self, df, stls=3):
        data = df.copy()
        # Heikin Ashi transformation
        data['HA close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        ha_open = np.zeros(len(data))
        ha_open[0] = data['Open'].iloc[0]
        for i in range(1, len(data)):
            ha_open[i] = (ha_open[i-1] + data['HA close'].iloc[i-1]) / 2
        data['HA open'] = ha_open
        data['HA high'] = data[['HA open', 'HA close', 'High']].max(axis=1)
        data['HA low'] = data[['HA open', 'HA close', 'Low']].min(axis=1)
        
        signals = np.zeros(len(data))
        current_cumsum = 0
        
        ha_o = data['HA open'].values
        ha_c = data['HA close'].values
        ha_h = data['HA high'].values
        ha_l = data['HA low'].values
        
        for n in range(1, len(data)):
            # Long
            if (ha_o[n] > ha_c[n] and ha_o[n] == ha_h[n] and
                np.abs(ha_o[n] - ha_c[n]) > np.abs(ha_o[n-1] - ha_c[n-1]) and
                ha_o[n-1] > ha_c[n-1]):
                
                sig = 1
                if current_cumsum + sig > stls:
                    sig = 0
                signals[n] = sig
                current_cumsum += sig
            
            # Exit
            elif (ha_o[n] < ha_c[n] and ha_o[n] == ha_l[n] and ha_o[n-1] < ha_c[n-1]):
                if current_cumsum > 0:
                    sig = -current_cumsum
                    signals[n] = sig
                    current_cumsum += sig
                    
        data['signals'] = signals
        data['cumsum'] = data['signals'].cumsum()
        return data

class BollingerBandsPattern(StrategyBase):
    def __init__(self):
        super().__init__("Bollinger Bands Pattern")
        
    def generate_signals(self, df, window=20, alpha=0.0001, beta=0.0001):
        data = df.copy()
        data['std'] = data['Close'].rolling(window=window).std()
        data['mid band'] = data['Close'].rolling(window=window).mean()
        data['upper band'] = data['mid band'] + 2 * data['std']
        data['lower band'] = data['mid band'] - 2 * data['std']
        
        signals = np.zeros(len(data))
        period = 75
        current_cumsum = 0
        
        # Simplified pattern recognition for Streamlit
        closes = data['Close'].values
        mids = data['mid band'].values
        uppers = data['upper band'].values
        lowers = data['lower band'].values
        stds = data['std'].values
        
        for i in range(period, len(data)):
            moveon = False
            threshold = 0.0
            
            if current_cumsum == 0 and closes[i] > uppers[i]:
                # W pattern check (simplified)
                sig = 0
                # Look for mid band touch
                for j in range(i, i-period, -1):
                    if np.abs(mids[j] - closes[j]) < alpha:
                        moveon = True
                        break
                if moveon:
                    # Look for lower band touch
                    for k in range(j, i-period, -1):
                        if np.abs(lowers[k] - closes[k]) < alpha:
                            threshold = closes[k]
                            sig = 1
                            break
                if sig == 1:
                    signals[i] = 1
                    current_cumsum += 1
            
            elif current_cumsum != 0 and stds[i] < beta:
                signals[i] = -current_cumsum
                current_cumsum = 0
                
        data['signals'] = signals
        data['cumsum'] = data['signals'].cumsum()
        return data

class RSIPattern(StrategyBase):
    def __init__(self):
        super().__init__("RSI Overbought/Oversold")
        
    def generate_signals(self, df, n=14):
        data = df.copy()
        delta = data['Close'].diff().dropna()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        # SMMA
        def get_smma(series, window):
            res = [series.iloc[0]]
            for i in range(1, len(series)):
                res.append((res[-1] * (window - 1) + series.iloc[i]) / window)
            return np.array(res)

        up_smma = get_smma(up, n)
        down_smma = get_smma(down, n)
        
        rs = up_smma / down_smma
        rsi = 100 - (100 / (1 + rs))
        
        full_rsi = np.zeros(len(data))
        full_rsi[n:] = rsi[n-1:]
        data['rsi'] = full_rsi
        
        data['positions'] = np.select([data['rsi'] < 30, data['rsi'] > 70], [1, -1], default=0)
        data['signals'] = pd.Series(data['positions']).diff().fillna(0)
        return data

class ShootingStar(StrategyBase):
    def __init__(self):
        super().__init__("Shooting Star")
        
    def generate_signals(self, df):
        data = df.copy()
        # open>close, red color
        cond1 = (data['Open'] >= data['Close'])
        # a candle with little or no lower wick
        cond2 = (data['Close'] - data['Low']) < 0.2 * abs(data['Close'] - data['Open'])
        # a long upper wick at least 2x the body
        cond4 = (data['High'] - data['Open']) >= 2 * abs(data['Open'] - data['Close'])
        # price uptrend
        cond56 = (data['Close'] >= data['Close'].shift(1)) & (data['Close'].shift(1) >= data['Close'].shift(2))
        
        data['signals'] = 0
        data.loc[cond1 & cond2 & cond4 & cond56, 'signals'] = -1 # Short signal
        
        # Simple exit after 5 days for this pattern
        idx_list = data[data['signals'] == -1].index
        for ind in idx_list:
            exit_idx = data.index.get_loc(ind) + 5
            if exit_idx < len(data):
                data.iloc[exit_idx, data.columns.get_loc('signals')] = 1
        
        data['positions'] = data['signals'].cumsum().fillna(0)
        return data


class PairTrading(StrategyBase):
    def __init__(self):
        super().__init__("Pair Trading")
        
    def run_backtest(self, df1, df2, ticker1="Asset 1", ticker2="Asset 2", capital=10000, **kwargs):
        # Flattening
        if isinstance(df1.columns, pd.MultiIndex): df1.columns = df1.columns.get_level_values(0)
        if isinstance(df2.columns, pd.MultiIndex): df2.columns = df2.columns.get_level_values(0)
            
        import statsmodels.api as sm
        bandwidth = kwargs.get('bandwidth', 250)
        
        signals = pd.DataFrame(index=df1.index)
        signals['asset1'] = df1['Close']
        signals['asset2'] = df2['Close']
        signals['signals1'] = 0
        
        # Vectorized-ish rolling cointegration is heavy, simplified for dashboard
        for i in range(bandwidth, len(signals), 5):
            y_window = signals['asset2'].iloc[i-bandwidth:i]
            x_window = signals['asset1'].iloc[i-bandwidth:i]
            
            # Clean windows
            valid = np.isfinite(y_window) & np.isfinite(x_window)
            y_clean = y_window[valid]
            x_clean = x_window[valid]
            
            if len(y_clean) > bandwidth // 2:
                model = sm.OLS(y_clean, sm.add_constant(x_clean)).fit()
                
                if sm.tsa.stattools.adfuller(model.resid)[1] < 0.05:
                    current_resid = signals['asset2'].iloc[i] - (model.params[0] + model.params[1] * signals['asset1'].iloc[i])
                    z = (current_resid - model.resid.mean()) / model.resid.std()
                    
                    if z > 1: signals.iloc[i, signals.columns.get_loc('signals1')] = 1
                    elif z < -1: signals.iloc[i, signals.columns.get_loc('signals1')] = -1

        signals['positions1'] = signals['signals1'].cumsum().ffill().fillna(0)
        signals['signals'] = signals['positions1'].diff().fillna(0)
        signals['Close'] = signals['asset1'] # For base compatibility
        
        # Portfolio logic for pairs
        pos1 = (capital // 2) // signals['asset1'].max()
        pos2 = (capital // 2) // signals['asset2'].max()
        
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['total asset'] = capital + (signals['positions1'] * (signals['asset1'].diff().fillna(0)) * pos1).cumsum() \
                                     - (signals['positions1'] * (signals['asset2'].diff().fillna(0)) * pos2).cumsum()
        portfolio['return'] = portfolio['total asset'].pct_change().fillna(0)
        
        return signals, portfolio

class ParabolicSAR(StrategyBase):
    def __init__(self):
        super().__init__("Parabolic SAR")
        
    def generate_signals(self, df, initial_af=0.02, step_af=0.02, end_af=0.2):
        new = df.copy().reset_index()
        new['trend'] = 0
        new['sar'] = 0.0
        new['real sar'] = 0.0
        new['ep'] = 0.0
        new['af'] = 0.0

        # Initial conditions
        if len(new) < 2: return new
        # Use .iloc and ensure we are comparing scalars
        close_1 = new['Close'].iloc[1]
        close_0 = new['Close'].iloc[0]
        high_0 = new['High'].iloc[0]
        low_0 = new['Low'].iloc[0]
        high_1 = new['High'].iloc[1]
        low_1 = new['Low'].iloc[1]

        new.at[1, 'trend'] = 1 if close_1 > close_0 else -1
        new.at[1, 'sar'] = high_0 if new.at[1, 'trend'] > 0 else low_0
        new.at[1, 'real sar'] = new.at[1, 'sar']
        new.at[1, 'ep'] = high_1 if new.at[1, 'trend'] > 0 else low_1
        new.at[1, 'af'] = initial_af

        # Recursion
        for i in range(2, len(new)):
            temp = new.at[i-1, 'sar'] + new.at[i-1, 'af'] * (new.at[i-1, 'ep'] - new.at[i-1, 'sar'])
            if new.at[i-1, 'trend'] < 0:
                new.at[i, 'sar'] = max(temp, new.at[i-1, 'High'], new.at[i-2, 'High'])
                trend = 1 if new.at[i, 'sar'] < new.at[i, 'High'] else new.at[i-1, 'trend'] - 1
            else:
                new.at[i, 'sar'] = min(temp, new.at[i-1, 'Low'], new.at[i-2, 'Low'])
                trend = -1 if new.at[i, 'sar'] > new.at[i, 'Low'] else new.at[i-1, 'trend'] + 1
            new.at[i, 'trend'] = trend
            
            if new.at[i, 'trend'] < 0:
                ep = min(new.at[i, 'Low'], new.at[i-1, 'ep']) if new.at[i, 'trend'] != -1 else new.at[i, 'Low']
            else:
                ep = max(new.at[i, 'High'], new.at[i-1, 'ep']) if new.at[i, 'trend'] != 1 else new.at[i, 'High']
            new.at[i, 'ep'] = ep
            
            if np.abs(new.at[i, 'trend']) == 1:
                real_sar = new.at[i-1, 'ep']
                new.at[i, 'af'] = initial_af
            else:
                real_sar = new.at[i, 'sar']
                if new.at[i, 'ep'] == new.at[i-1, 'ep']:
                    new.at[i, 'af'] = new.at[i-1, 'af']
                else:
                    new.at[i, 'af'] = min(end_af, new.at[i-1, 'af'] + step_af)
            new.at[i, 'real sar'] = real_sar

        new.set_index('Date', inplace=True)
        new['positions'] = np.where(new['real sar'] < new['Close'], 1, 0)
        new['signals'] = new['positions'].diff().fillna(0)
        return new

class LondonBreakout(StrategyBase):
    def __init__(self):
        super().__init__("London Breakout")
        
    def generate_signals(self, df):
        # Since this is daily data, we implement a Daily Open Range Breakout (ORB)
        data = df.copy()
        data['range'] = (data['High'] - data['Low']).shift(1)
        data['upper'] = data['Open'] + 0.5 * data['range']
        data['lower'] = data['Open'] - 0.5 * data['range']
        data['positions'] = np.where(data['Close'] > data['upper'], 1, 
                                     np.where(data['Close'] < data['lower'], -1, 0))
        data['signals'] = pd.Series(data['positions']).diff().fillna(0)
        return data

class DualThrust(StrategyBase):
    def __init__(self):
        super().__init__("Dual Thrust")
        
    def generate_signals(self, df):
        data = df.copy()
        rg = 5 # default range
        k = 0.5 # default param
        
        # Calculate range components
        data['hh'] = data['High'].rolling(rg).max()
        data['lc'] = data['Close'].rolling(rg).min()
        data['hc'] = data['Close'].rolling(rg).max()
        data['ll'] = data['Low'].rolling(rg).min()
        
        data['range'] = np.maximum(data['hh'] - data['lc'], data['hc'] - data['ll'])
        data['upper'] = data['Open'] + k * data['range'].shift(1)
        data['lower'] = data['Open'] - k * data['range'].shift(1)
        
        data['positions'] = np.where(data['Close'] > data['upper'], 1, 
                                     np.where(data['Close'] < data['lower'], -1, 0))
        data['signals'] = pd.Series(data['positions']).diff().fillna(0)
        return data

STRATEGIES = {
    "Awesome Oscillator": AwesomeOscillator(),
    "MACD Oscillator": MACDOscillator(),
    "Heikin-Ashi": HeikinAshiStrategy(),
    "Bollinger Bands Pattern": BollingerBandsPattern(),
    "RSI Strategy": RSIPattern(),
    "Shooting Star": ShootingStar(),
    "Parabolic SAR": ParabolicSAR(),
    "London Breakout": LondonBreakout(),
    "Dual Thrust": DualThrust()
}
