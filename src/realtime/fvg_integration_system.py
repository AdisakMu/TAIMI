"""
FVG Integration System
รวมทุกระบบเข้าด้วยกัน: Database, FVG Monitor, Claude Analysis, MT5 Trading
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from dataclasses import asdict

from ..data.ohlc_database import OHLCDatabase
from ..realtime.realtime_fvg_monitor import FVGMonitor, FVGEventHandler
from ..analysis.claude_signal_analyzer import ClaudeSignalAnalyzer
from ..trading.mt5_order_manager import MT5OrderManager, TradeRequest, OrderType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FVGTradingEventHandler(FVGEventHandler):
    """Event handler ที่รวม Claude analysis และ MT5 trading"""
    
    def __init__(self, claude_analyzer: ClaudeSignalAnalyzer, 
                 order_manager: MT5OrderManager, 
                 min_confidence_score: int = 70):
        self.claude_analyzer = claude_analyzer
        self.order_manager = order_manager
        self.min_confidence_score = min_confidence_score
        self.active_trades = {}
        
    async def on_fvg_breakout(self, data: Dict):
        """จัดการเมื่อเกิด FVG breakout"""
        try:
            logger.info(f"Processing FVG breakout: {data['symbol']} {data['timeframe']}")
            
            # ส่งไป Claude วิเคราะห์
            analysis = await self.claude_analyzer.analyze_signal(data)
            
            logger.info(f"Claude analysis - Score: {analysis.get('confidence_score', 0)}, "
                       f"Decision: {analysis.get('trading_decision', 'HOLD')}")
            
            # ตรวจสอบว่าผ่านเกณฑ์หรือไม่
            confidence_score = analysis.get('confidence_score', 0)
            trading_decision = analysis.get('trading_decision', 'HOLD')
            
            if (confidence_score >= self.min_confidence_score and 
                trading_decision in ['BUY', 'SELL']):
                
                await self._execute_trade(data, analysis)
            else:
                logger.info(f"Trade skipped - Score: {confidence_score}, Decision: {trading_decision}")
                
        except Exception as e:
            logger.error(f"Error handling FVG breakout: {e}")
    
    async def _execute_trade(self, fvg_data: Dict, claude_analysis: Dict):
        """Execute trade based on FVG and Claude analysis"""
        try:
            symbol = fvg_data['symbol']
            fvg = fvg_data['fvg']
            
            # คำนวณ position size
            account_info = self.order_manager.get_account_info()
            if not account_info:
                logger.error("Cannot get account info")
                return
            
            entry_price = fvg['entry_level']
            stop_loss = fvg['stop_loss']
            take_profit = fvg['take_profit']
            
            position_size = self.order_manager.calculate_position_size(
                symbol, entry_price, stop_loss
            )
            
            # กำหนดประเภท order
            trading_decision = claude_analysis.get('trading_decision')
            if trading_decision == 'BUY':
                order_type = OrderType.BUY
            else:  # SELL
                order_type = OrderType.SELL
            
            # สร้าง trade request
            trade_request = TradeRequest(
                symbol=symbol,
                order_type=order_type,
                volume=position_size,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=f"FVG-{fvg['fvg_type']}-{fvg_data['timeframe']}-{claude_analysis.get('confidence_score', 0)}"
            )
            
            # Execute trade
            result = await self.order_manager.place_market_order(trade_request)
            
            if result.success:
                logger.info(f"Trade executed successfully: {symbol} {trading_decision} "
                           f"Size: {position_size} Price: {result.price_filled}")
                
                # เก็บข้อมูล trade
                self.active_trades[result.order_id] = {
                    'symbol': symbol,
                    'timeframe': fvg_data['timeframe'],
                    'fvg_data': fvg_data,
                    'claude_analysis': claude_analysis,
                    'trade_result': asdict(result),
                    'timestamp': datetime.now()
                }
            else:
                logger.error(f"Trade execution failed: {result.message}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")

class FVGIntegrationSystem:
    """ระบบรวมทั้งหมดสำหรับ FVG Breakout Trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db = None
        self.monitor = None
        self.claude_analyzer = None
        self.order_manager = None
        self.event_handler = None
        self.running = False
        
    async def initialize(self) -> bool:
        """เริ่มต้นระบบทั้งหมด"""
        try:
            logger.info("Initializing FVG Integration System...")
            
            # 1. เริ่มต้น Database
            self.db = OHLCDatabase(self.config['database'])
            if not await self.db.initialize():
                logger.error("Database initialization failed")
                return False
            
            # 2. เริ่มต้น Claude Analyzer
            self.claude_analyzer = ClaudeSignalAnalyzer(
                self.config['claude']['api_key'],
                self.config['claude'].get('model', 'claude-3-sonnet-20240229')
            )
            await self.claude_analyzer.initialize()
            
            # 3. เริ่มต้น MT5 Order Manager
            mt5_config = self.config['mt5']
            self.order_manager = MT5OrderManager(
                account=mt5_config['account'],
                password=mt5_config['password'],
                server=mt5_config['server'],
                max_risk_percent=mt5_config.get('max_risk_percent', 2.0)
            )
            
            if not await self.order_manager.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # 4. สร้าง Event Handler
            self.event_handler = FVGTradingEventHandler(
                self.claude_analyzer,
                self.order_manager,
                self.config.get('min_confidence_score', 70)
            )
            
            # 5. เริ่มต้น FVG Monitor
            self.monitor = FVGMonitor(
                self.db,
                self.config['symbols'],
                self.config['timeframes']
            )
            
            # เพิ่ม event callbacks
            self.monitor.add_callback(self._handle_fvg_event)
            
            logger.info("All systems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            return False
    
    async def _handle_fvg_event(self, event_type: str, data: Dict):
        """จัดการ FVG events"""
        try:
            if event_type == 'FVG_FORMED':
                await self.event_handler.on_fvg_formed(data)
            elif event_type == 'FVG_RETEST':
                await self.event_handler.on_fvg_retest(data)
            elif event_type == 'FVG_BREAKOUT':
                await self.event_handler.on_fvg_breakout(data)
                
        except Exception as e:
            logger.error(f"Error handling FVG event {event_type}: {e}")
    
    async def start(self):
        """เริ่มระบบทั้งหมด"""
        try:
            if not await self.initialize():
                logger.error("System initialization failed")
                return
            
            self.running = True
            logger.info("FVG Integration System started")
            
            # เริ่ม tasks แบบ concurrent
            tasks = [
                asyncio.create_task(self._start_data_collection()),
                asyncio.create_task(self._start_fvg_monitoring()),
                asyncio.create_task(self._start_periodic_tasks())
            ]
            
            # รอให้ tasks ทำงาน
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.stop()
    
    async def _start_data_collection(self):
        """เริ่ม data collection"""
        try:
            symbols = self.config['symbols']
            timeframes = {
                'M1': 1,    # MT5 timeframe constants
                'M5': 5,
                'M15': 15,
                'H1': 16385
            }
            
            await self.db.start_data_collection(symbols, timeframes)
            
        except Exception as e:
            logger.error(f"Data collection error: {e}")
    
    async def _start_fvg_monitoring(self):
        """เริ่ม FVG monitoring"""
        try:
            await self.monitor.start_monitoring()
        except Exception as e:
            logger.error(f"FVG monitoring error: {e}")
    
    async def _start_periodic_tasks(self):
        """งานที่ทำเป็นระยะ"""
        while self.running:
            try:
                # ตรวจสอบสถานะบัญชี
                await self._check_account_status()
                
                # ตรวจสอบและปิด trades ที่ควรปิด
                await self._manage_active_trades()
                
                # รอ 1 นาที
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Periodic task error: {e}")
                await asyncio.sleep(30)
    
    async def _check_account_status(self):
        """ตรวจสอบสถานะบัญชี"""
        try:
            account_info = self.order_manager.get_account_info()
            if account_info:
                margin_level = account_info.get('margin_level', 0)
                if margin_level < 200 and margin_level > 0:  # Margin call level
                    logger.warning(f"Low margin level: {margin_level}%")
                    
        except Exception as e:
            logger.error(f"Error checking account status: {e}")
    
    async def _manage_active_trades(self):
        """จัดการ trades ที่เปิดอยู่"""
        try:
            positions = self.order_manager.get_positions()
            
            for position in positions:
                # ตรวจสอบเงื่อนไขการปิด trade
                # เช่น trailing stop, time-based exit, etc.
                pass
                
        except Exception as e:
            logger.error(f"Error managing active trades: {e}")
    
    def get_system_status(self) -> Dict:
        """ดึงสถานะของระบบ"""
        try:
            account_info = self.order_manager.get_account_info() if self.order_manager else {}
            fvg_stats = self.monitor.get_statistics() if self.monitor else {}
            
            return {
                'running': self.running,
                'timestamp': datetime.now().isoformat(),
                'account_info': account_info,
                'fvg_statistics': fvg_stats,
                'active_trades': len(self.event_handler.active_trades) if self.event_handler else 0,
                'systems': {
                    'database': self.db is not None,
                    'claude_analyzer': self.claude_analyzer is not None,
                    'order_manager': self.order_manager is not None and self.order_manager.connected,
                    'fvg_monitor': self.monitor is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    async def stop(self):
        """หยุดระบบ"""
        try:
            logger.info("Stopping FVG Integration System...")
            self.running = False
            
            if self.monitor:
                self.monitor.stop_monitoring()
            
            if self.claude_analyzer:
                await self.claude_analyzer.close()
            
            if self.order_manager:
                self.order_manager.disconnect()
            
            if self.db:
                await self.db.close()
            
            logger.info("System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")

# Configuration และ Main Function
def load_config() -> Dict:
    """โหลด configuration"""
    return {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'user': 'trading_user',
            'password': 'your_db_password',
            'database': 'trading_db'
        },
        'claude': {
            'api_key': 'your_claude_api_key',
            'model': 'claude-3-sonnet-20240229'
        },
        'mt5': {
            'account': 123456789,
            'password': 'your_mt5_password',
            'server': 'your_broker_server',
            'max_risk_percent': 2.0
        },
        'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        'timeframes': ['M5', 'M15', 'H1'],
        'min_confidence_score': 70
    }

async def main():
    """Main function"""
    try:
        config = load_config()
        system = FVGIntegrationSystem(config)
        
        # เริ่มระบบ
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Main error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
