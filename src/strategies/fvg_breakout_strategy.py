"""
FVG Breakout Strategy
หา Fair Value Gaps และคำนวณ entry/exit levels พร้อม indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class FVGData:
    """ข้อมูล Fair Value Gap"""
    fvg_type: str  # 'bullish' หรือ 'bearish'
    start_index: int
    end_index: int
    gap_high: float
    gap_low: float
    gap_size: float
    entry_level: float
    stop_loss: float
    take_profit: float
    status: str  # 'FORMED', 'RETEST', 'BREAKOUT', 'FILLED'
    timestamp: datetime
    confidence: float

@dataclass 
class TechnicalIndicators:
    """Technical Indicators สำหรับ confirmation"""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    support_level: float
    resistance_level: float
    atr: float

class FVGBreakoutStrategy:
    def __init__(self, risk_percent: float = 2.0):
        self.risk_percent = risk_percent
        self.fvg_list: List[FVGData] = []
        
    def identify_fvg_gaps(self, df: pd.DataFrame) -> List[FVGData]:
        """
        หา Fair Value Gaps
        Bullish FVG: prev['high'] < next_candle['low']
        Bearish FVG: prev['low'] > next_candle['high']
        """
        fvg_gaps = []
        
        if len(df) < 3:
            return fvg_gaps
            
        for i in range(1, len(df) - 1):
            prev_candle = df.iloc[i-1]
            current_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG
            if prev_candle['high'] < next_candle['low']:
                gap_high = next_candle['low']
                gap_low = prev_candle['high']
                gap_size = gap_high - gap_low
                
                if gap_size > 0:  # ตรวจสอบว่า gap มีขนาดมากกว่า 0
                    fvg = FVGData(
                        fvg_type='bullish',
                        start_index=i-1,
                        end_index=i+1,
                        gap_high=gap_high,
                        gap_low=gap_low,
                        gap_size=gap_size,
                        entry_level=gap_low + (gap_size * 0.5),  # Entry ที่กึ่งกลาง gap
                        stop_loss=gap_low - (gap_size * 0.5),
                        take_profit=gap_high + (gap_size * 2.0),
                        status='FORMED',
                        timestamp=df.iloc[i+1]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                        confidence=self._calculate_fvg_confidence(df, i, 'bullish')
                    )
                    fvg_gaps.append(fvg)
            
            # Bearish FVG  
            elif prev_candle['low'] > next_candle['high']:
                gap_high = prev_candle['low']
                gap_low = next_candle['high']
                gap_size = gap_high - gap_low
                
                if gap_size > 0:
                    fvg = FVGData(
                        fvg_type='bearish',
                        start_index=i-1,
                        end_index=i+1,
                        gap_high=gap_high,
                        gap_low=gap_low,
                        gap_size=gap_size,
                        entry_level=gap_high - (gap_size * 0.5),  # Entry ที่กึ่งกลาง gap
                        stop_loss=gap_high + (gap_size * 0.5),
                        take_profit=gap_low - (gap_size * 2.0),
                        status='FORMED',
                        timestamp=df.iloc[i+1]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                        confidence=self._calculate_fvg_confidence(df, i, 'bearish')
                    )
                    fvg_gaps.append(fvg)
        
        logger.info(f"Found {len(fvg_gaps)} FVG gaps")
        return fvg_gaps
    
    def _calculate_fvg_confidence(self, df: pd.DataFrame, index: int, 
                                 fvg_type: str) -> float:
        """คำนวณความมั่นใจของ FVG"""
        confidence = 50.0  # base confidence
        
        try:
            # ตรวจสอบ volume ในช่วงที่เกิด FVG
            if index > 0 and index < len(df) - 1:
                if 'volume' in df.columns:
                    avg_volume = df['volume'].rolling(10).mean().iloc[index]
                    current_volume = df.iloc[index]['volume']
                    
                    if current_volume > avg_volume * 1.5:
                        confidence += 20.0
                
                # ตรวจสอบ momentum
                if len(df) >= 5:
                    recent_high = df['high'].rolling(5).max().iloc[index]
                    recent_low = df['low'].rolling(5).min().iloc[index]
                    current_close = df.iloc[index]['close']
                    
                    if fvg_type == 'bullish' and current_close > recent_high * 0.9:
                        confidence += 15.0
                    elif fvg_type == 'bearish' and current_close < recent_low * 1.1:
                        confidence += 15.0
        
        except Exception as e:
            logger.warning(f"Error calculating FVG confidence: {e}")        
        return min(confidence, 100.0)
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """คำนวณ RSI"""
        try:
            prices_array = np.array(prices)
            if len(prices_array) < period + 1:
                return 50.0
            
            deltas = np.diff(prices_array)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
            avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
            
            if len(avg_gains) == 0 or avg_losses[-1] == 0:
                return 50.0
            
            rs = avg_gains[-1] / avg_losses[-1]
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
        except Exception:
            return 50.0
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """คำนวณ MACD"""
        try:
            prices_array = np.array(prices)
            if len(prices_array) < slow + signal:
                return 0.0, 0.0, 0.0
            
            # คำนวณ EMA
            def ema(data, span):
                return pd.Series(data).ewm(span=span).mean().values
            
            ema_fast = ema(prices_array, fast)
            ema_slow = ema(prices_array, slow)
            
            # Convert to numpy arrays for arithmetic operations
            macd_line = np.array(ema_fast) - np.array(ema_slow)
            signal_line = ema(macd_line, signal)
            histogram = np.array(macd_line) - np.array(signal_line)
            
            return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _calculate_atr(self, high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
        """คำนวณ ATR"""
        try:
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            
            if len(high_array) < period + 1:
                return 0.001
            
            # คำนวณ True Range
            prev_close = np.roll(close_array, 1)
            prev_close[0] = close_array[0]  # ใช้ close แรกแทน
            
            tr1 = high_array - low_array
            tr2 = np.abs(high_array - prev_close)
            tr3 = np.abs(low_array - prev_close)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # คำนวณ ATR (Simple Moving Average ของ True Range)
            atr = np.mean(true_range[-period:])
            
            return float(atr)
        except Exception:
            return 0.001

    def calculate_technical_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """คำนวณ Technical Indicators"""
        try:
            close_prices = df['close'].values.tolist()
            high_prices = df['high'].values.tolist()
            low_prices = df['low'].values.tolist()
            
            # RSI calculation
            rsi = self._calculate_rsi(close_prices, 14)
            
            # MACD calculation
            macd, macd_signal, macd_histogram = self._calculate_macd(close_prices)
            
            # ATR calculation
            atr = self._calculate_atr(high_prices, low_prices, close_prices, 14)
            
            # Support & Resistance (ใช้ pivot points)
            support, resistance = self._calculate_support_resistance(df)
            
            return TechnicalIndicators(
                rsi=rsi if not np.isnan(rsi) else 50.0,
                macd=macd if not np.isnan(macd) else 0.0,
                macd_signal=macd_signal if not np.isnan(macd_signal) else 0.0,
                macd_histogram=macd_histogram if not np.isnan(macd_histogram) else 0.0,
                support_level=support,
                resistance_level=resistance,
                atr=atr if not np.isnan(atr) else 0.001
            )
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return TechnicalIndicators(50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001)
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """คำนวณ Support และ Resistance levels"""
        try:
            # ใช้ rolling min/max สำหรับ support/resistance
            period = min(20, len(df))
            support = df['low'].rolling(period).min().iloc[-1]
            resistance = df['high'].rolling(period).max().iloc[-1]
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            current_price = df['close'].iloc[-1]
            return current_price * 0.99, current_price * 1.01
    
    def update_fvg_status(self, fvg: FVGData, current_price: float) -> FVGData:
        """อัพเดท status ของ FVG"""
        if fvg.status == 'FORMED':
            # ตรวจสอบว่ามี retest หรือไม่
            if fvg.fvg_type == 'bullish':
                if fvg.gap_low <= current_price <= fvg.gap_high:
                    fvg.status = 'RETEST'
                elif current_price > fvg.gap_high:
                    fvg.status = 'BREAKOUT'
            else:  # bearish
                if fvg.gap_low <= current_price <= fvg.gap_high:
                    fvg.status = 'RETEST'
                elif current_price < fvg.gap_low:
                    fvg.status = 'BREAKOUT'
        
        elif fvg.status == 'RETEST':
            # ตรวจสอบ breakout
            if fvg.fvg_type == 'bullish' and current_price > fvg.gap_high:
                fvg.status = 'BREAKOUT'
            elif fvg.fvg_type == 'bearish' and current_price < fvg.gap_low:
                fvg.status = 'BREAKOUT'
        
        return fvg
    
    def generate_trading_signal(self, df: pd.DataFrame) -> Dict:
        """สร้าง trading signal"""
        try:
            # หา FVG gaps
            new_fvgs = self.identify_fvg_gaps(df)
            current_price = df['close'].iloc[-1]
            
            # อัพเดท status ของ FVG ที่มีอยู่
            for i, fvg in enumerate(self.fvg_list):
                self.fvg_list[i] = self.update_fvg_status(fvg, current_price)
            
            # เพิ่ม FVG ใหม่
            self.fvg_list.extend(new_fvgs)
            
            # คำนวณ technical indicators
            indicators = self.calculate_technical_indicators(df)
            
            # หา FVG ที่เกิด breakout
            breakout_fvgs = [fvg for fvg in self.fvg_list if fvg.status == 'BREAKOUT']
            
            signal = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'fvg_count': len(self.fvg_list),
                'new_fvgs': len(new_fvgs),
                'breakout_fvgs': len(breakout_fvgs),
                'indicators': indicators,
                'fvg_list': self.fvg_list,
                'recommendation': 'HOLD'
            }
            
            # สร้างคำแนะนำ
            if breakout_fvgs:
                latest_breakout = max(breakout_fvgs, key=lambda x: x.timestamp)
                
                if latest_breakout.fvg_type == 'bullish':
                    if (indicators.rsi < 70 and 
                        indicators.macd > indicators.macd_signal and
                        current_price > indicators.support_level):
                        signal['recommendation'] = 'BUY'
                        signal['entry_price'] = latest_breakout.entry_level
                        signal['stop_loss'] = latest_breakout.stop_loss
                        signal['take_profit'] = latest_breakout.take_profit
                        
                elif latest_breakout.fvg_type == 'bearish':
                    if (indicators.rsi > 30 and 
                        indicators.macd < indicators.macd_signal and
                        current_price < indicators.resistance_level):
                        signal['recommendation'] = 'SELL'
                        signal['entry_price'] = latest_breakout.entry_level
                        signal['stop_loss'] = latest_breakout.stop_loss
                        signal['take_profit'] = latest_breakout.take_profit
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'recommendation': 'HOLD'
            }
    
    def calculate_position_size(self, account_balance: float, 
                               entry_price: float, stop_loss: float) -> float:
        """คำนวณขนาด position"""
        try:
            risk_amount = account_balance * (self.risk_percent / 100)
            price_diff = abs(entry_price - stop_loss)
            
            if price_diff > 0:
                position_size = risk_amount / price_diff
                return round(position_size, 2)
            
            return 0.01  # minimum position size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # สร้าง sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    # สร้างข้อมูลราคาแบบ random walk
    price_changes = np.random.randn(100) * 0.01
    prices = 1.1000 + np.cumsum(price_changes)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.rand(100) * 0.005,
        'low': prices - np.random.rand(100) * 0.005,
        'close': prices + np.random.randn(100) * 0.002,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # ทดสอบ strategy
    strategy = FVGBreakoutStrategy()
    signal = strategy.generate_trading_signal(df)
    
    print(f"Signal: {signal['recommendation']}")
    print(f"FVG Count: {signal['fvg_count']}")
    print(f"Current Price: {signal['current_price']:.5f}")