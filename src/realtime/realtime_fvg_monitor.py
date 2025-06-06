"""
Real-time FVG Monitor
ติดตาม FVG real-time ทุก 5 วินาที และส่ง callbacks เมื่อเกิด breakout
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from dataclasses import asdict
import pandas as pd

from ..strategies.fvg_breakout_strategy import FVGBreakoutStrategy, FVGData
from ..data.ohlc_database import OHLCDatabase

logger = logging.getLogger(__name__)

class FVGMonitor:
    def __init__(self, db: OHLCDatabase, symbols: List[str], 
                 timeframes: List[str]):
        self.db = db
        self.symbols = symbols
        self.timeframes = timeframes
        self.strategies = {}
        self.callbacks = []
        self.monitoring = False
        self.last_fvg_count = {}
        
        # สร้าง strategy สำหรับแต่ละ symbol และ timeframe
        for symbol in symbols:
            self.strategies[symbol] = {}
            self.last_fvg_count[symbol] = {}
            for tf in timeframes:
                self.strategies[symbol][tf] = FVGBreakoutStrategy()
                self.last_fvg_count[symbol][tf] = 0
    
    def add_callback(self, callback: Callable):
        """เพิ่ม callback function ที่จะถูกเรียกเมื่อเกิดเหตุการณ์สำคัญ"""
        self.callbacks.append(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Dict):
        """เรียก callbacks ทั้งหมด"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def start_monitoring(self):
        """เริ่มติดตาม FVG real-time"""
        self.monitoring = True
        logger.info("FVG Real-time monitoring started")
        
        while self.monitoring:
            try:
                await self._scan_all_symbols()
                await asyncio.sleep(5)  # รอ 5 วินาที
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)  # รอนานขึ้นถ้าเกิด error
    
    async def _scan_all_symbols(self):
        """สแกน FVG ทุก symbols และ timeframes"""
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                await self._scan_symbol_timeframe(symbol, timeframe)
    
    async def _scan_symbol_timeframe(self, symbol: str, timeframe: str):
        """สแกน FVG สำหรับ symbol และ timeframe ที่กำหนด"""
        try:
            # ดึงข้อมูล OHLC
            df = await self.db.get_ohlc_data(symbol, timeframe, 100)
            
            if df.empty:
                return
            
            # วิเคราะห์ FVG
            strategy = self.strategies[symbol][timeframe]
            signal = strategy.generate_trading_signal(df)
            
            # ตรวจสอบ FVG ใหม่
            current_fvg_count = signal['fvg_count']
            last_count = self.last_fvg_count[symbol][timeframe]
            
            if signal['new_fvgs'] > 0:
                await self._handle_new_fvg(symbol, timeframe, signal)
            
            # ตรวจสอบ breakout
            if signal['breakout_fvgs'] > 0:
                await self._handle_breakout(symbol, timeframe, signal)
            
            # อัพเดท count
            self.last_fvg_count[symbol][timeframe] = current_fvg_count
            
        except Exception as e:
            logger.error(f"Error scanning {symbol} {timeframe}: {e}")
    
    async def _handle_new_fvg(self, symbol: str, timeframe: str, signal: Dict):
        """จัดการเมื่อเกิด FVG ใหม่"""
        new_fvgs = [fvg for fvg in signal['fvg_list'] 
                   if fvg.status == 'FORMED'][-signal['new_fvgs']:]
        
        for fvg in new_fvgs:
            event_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'fvg': asdict(fvg),
                'current_price': signal['current_price'],
                'indicators': asdict(signal['indicators']),
                'timestamp': datetime.now()
            }
            
            await self._notify_callbacks('FVG_FORMED', event_data)
            logger.info(f"New {fvg.fvg_type} FVG formed: {symbol} {timeframe}")
    
    async def _handle_breakout(self, symbol: str, timeframe: str, signal: Dict):
        """จัดการเมื่อเกิด breakout"""
        breakout_fvgs = [fvg for fvg in signal['fvg_list'] 
                        if fvg.status == 'BREAKOUT']
        
        for fvg in breakout_fvgs:
            event_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'fvg': asdict(fvg),
                'current_price': signal['current_price'],
                'indicators': asdict(signal['indicators']),
                'recommendation': signal['recommendation'],
                'timestamp': datetime.now()
            }
            
            await self._notify_callbacks('FVG_BREAKOUT', event_data)
            logger.info(f"FVG Breakout detected: {symbol} {timeframe} {fvg.fvg_type}")
    
    async def _handle_retest(self, symbol: str, timeframe: str, signal: Dict):
        """จัดการเมื่อเกิด retest"""
        retest_fvgs = [fvg for fvg in signal['fvg_list'] 
                      if fvg.status == 'RETEST']
        
        for fvg in retest_fvgs:
            event_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'fvg': asdict(fvg),
                'current_price': signal['current_price'],
                'indicators': asdict(signal['indicators']),
                'timestamp': datetime.now()
            }
            
            await self._notify_callbacks('FVG_RETEST', event_data)
            logger.info(f"FVG Retest: {symbol} {timeframe} {fvg.fvg_type}")
    
    def get_current_fvgs(self, symbol: str, timeframe: str) -> List[FVGData]:
        """ดึง FVG ปัจจุบันสำหรับ symbol และ timeframe"""
        if symbol in self.strategies and timeframe in self.strategies[symbol]:
            return self.strategies[symbol][timeframe].fvg_list
        return []
    
    def get_all_active_fvgs(self) -> Dict:
        """ดึง FVG ที่ active ทั้งหมด"""
        active_fvgs = {}
        
        for symbol in self.symbols:
            active_fvgs[symbol] = {}
            for timeframe in self.timeframes:
                fvg_list = self.get_current_fvgs(symbol, timeframe)
                active_fvgs[symbol][timeframe] = [
                    asdict(fvg) for fvg in fvg_list 
                    if fvg.status in ['FORMED', 'RETEST', 'BREAKOUT']
                ]
        
        return active_fvgs
    
    def get_statistics(self) -> Dict:
        """ดึงสถิติการทำงาน"""
        stats = {
            'total_symbols': len(self.symbols),
            'total_timeframes': len(self.timeframes),
            'monitoring': self.monitoring,
            'fvg_counts': {}
        }
        
        for symbol in self.symbols:
            stats['fvg_counts'][symbol] = {}
            for timeframe in self.timeframes:
                fvg_list = self.get_current_fvgs(symbol, timeframe)
                stats['fvg_counts'][symbol][timeframe] = {
                    'total': len(fvg_list),
                    'formed': len([f for f in fvg_list if f.status == 'FORMED']),
                    'retest': len([f for f in fvg_list if f.status == 'RETEST']),
                    'breakout': len([f for f in fvg_list if f.status == 'BREAKOUT'])
                }
        
        return stats
    
    def stop_monitoring(self):
        """หยุดการติดตาม"""
        self.monitoring = False
        logger.info("FVG monitoring stopped")
    
    async def force_scan(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """สแกนแบบ manual"""
        if symbol and timeframe:
            await self._scan_symbol_timeframe(symbol, timeframe)
        else:
            await self._scan_all_symbols()

# Event Handler Classes
class FVGEventHandler:
    """Base class สำหรับ handle FVG events"""
    
    async def on_fvg_formed(self, data: Dict):
        """เมื่อเกิด FVG ใหม่"""
        pass
    
    async def on_fvg_retest(self, data: Dict):
        """เมื่อเกิด retest"""
        pass
    
    async def on_fvg_breakout(self, data: Dict):
        """เมื่อเกิด breakout"""
        pass

class LoggingEventHandler(FVGEventHandler):
    """Event handler ที่บันทึก log"""
    
    async def on_fvg_formed(self, data: Dict):
        logger.info(f"FVG FORMED: {data['symbol']} {data['timeframe']} "
                   f"{data['fvg']['fvg_type']} at {data['current_price']}")
    
    async def on_fvg_retest(self, data: Dict):
        logger.info(f"FVG RETEST: {data['symbol']} {data['timeframe']} "
                   f"{data['fvg']['fvg_type']} at {data['current_price']}")
    
    async def on_fvg_breakout(self, data: Dict):
        logger.warning(f"FVG BREAKOUT: {data['symbol']} {data['timeframe']} "
                      f"{data['fvg']['fvg_type']} at {data['current_price']}")

# ตัวอย่างการใช้งาน
async def main():
    # ตั้งค่า database
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'trading_user',
        'password': 'your_password',
        'database': 'trading_db'
    }
    
    db = OHLCDatabase(db_config)
    await db.initialize()
    
    # สร้าง monitor
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    timeframes = ['M1', 'M5', 'M15']
    
    monitor = FVGMonitor(db, symbols, timeframes)
    
    # เพิ่ม event handlers
    logging_handler = LoggingEventHandler()
    
    async def event_callback(event_type: str, data: Dict):
        if event_type == 'FVG_FORMED':
            await logging_handler.on_fvg_formed(data)
        elif event_type == 'FVG_RETEST':
            await logging_handler.on_fvg_retest(data)
        elif event_type == 'FVG_BREAKOUT':
            await logging_handler.on_fvg_breakout(data)
    
    monitor.add_callback(event_callback)
    
    # เริ่มติดตาม
    await monitor.start_monitoring()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
