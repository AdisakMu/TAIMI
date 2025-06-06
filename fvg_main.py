"""
FVG System Main Launcher
เริ่มระบบ FVG Breakout Trading System ทั้งหมด
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime
import json

from src.realtime.fvg_integration_system import FVGIntegrationSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'fvg_system_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FVGSystemLauncher:
    def __init__(self):
        self.system = None
        self.running = False
    
    def load_config(self):
        """โหลด configuration จากไฟล์หรือ environment variables"""
        # ในการใช้งานจริง ควรโหลดจากไฟล์ config หรือ environment variables
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
            'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
            'timeframes': ['M5', 'M15', 'H1'],
            'min_confidence_score': 70,
            'settings': {
                'data_collection_interval': 300,  # 5 minutes
                'fvg_scan_interval': 5,          # 5 seconds
                'max_concurrent_trades': 5,
                'trading_hours': {
                    'start': '00:00',
                    'end': '23:59'
                }
            }
        }
    
    def setup_signal_handlers(self):
        """ตั้งค่า signal handlers สำหรับ graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.running = False
            if self.system:
                asyncio.create_task(self.system.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def display_system_info(self):
        """แสดงข้อมูลระบบ"""
        print("=" * 60)
        print("🚀 FVG BREAKOUT TRADING SYSTEM")
        print("=" * 60)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("🔧 System Components:")
        print("   ├── PostgreSQL Database (OHLC Data Storage)")
        print("   ├── Real-time FVG Monitor (5-second scanning)")
        print("   ├── Claude AI Signal Analysis")
        print("   ├── MetaTrader 5 Order Management")
        print("   └── Risk Management & Position Sizing")
        print()
        print("📊 Trading Logic:")
        print("   • Bullish FVG: prev_high < next_low")
        print("   • Bearish FVG: prev_low > next_high")
        print("   • Confirmation: RSI, MACD, Support/Resistance")
        print("   • Claude AI Score: 0-100 (Minimum: 70)")
        print()
        print("⚠️  Press Ctrl+C to stop the system")
        print("=" * 60)
    
    async def monitor_system_status(self):
        """ติดตามสถานะระบบเป็นระยะ"""
        last_status_time = datetime.now()
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # แสดงสถานะทุก 5 นาที
                if (current_time - last_status_time).seconds >= 300:
                    if self.system:
                        status = self.system.get_system_status()
                        logger.info("=== SYSTEM STATUS ===")
                        logger.info(f"Running: {status.get('running', False)}")
                        logger.info(f"Active Trades: {status.get('active_trades', 0)}")
                        
                        account_info = status.get('account_info', {})
                        if account_info:
                            logger.info(f"Account Balance: {account_info.get('balance', 0)}")
                            logger.info(f"Account Equity: {account_info.get('equity', 0)}")
                        
                        fvg_stats = status.get('fvg_statistics', {})
                        if fvg_stats:
                            logger.info(f"FVG Monitoring: {fvg_stats.get('total_symbols', 0)} symbols, "
                                      f"{fvg_stats.get('total_timeframes', 0)} timeframes")
                    
                    last_status_time = current_time
                
                await asyncio.sleep(30)  # ตรวจสอบทุก 30 วินาที
                
            except Exception as e:
                logger.error(f"Status monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def run(self):
        """เริ่มรันระบบ"""
        try:
            self.setup_signal_handlers()
            await self.display_system_info()
            
            # โหลด configuration
            config = self.load_config()
            logger.info("Configuration loaded")
            
            # สร้างและเริ่มระบบ
            self.system = FVGIntegrationSystem(config)
            self.running = True
            
            # เริ่มการติดตามสถานะ
            status_task = asyncio.create_task(self.monitor_system_status())
            
            # เริ่มระบบหลัก
            system_task = asyncio.create_task(self.system.start())
            
            # รอให้ tasks เสร็จ
            done, pending = await asyncio.wait(
                [status_task, system_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # ยกเลิก tasks ที่ยังทำงานอยู่
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"System launcher error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """ทำความสะอาดก่อนปิดระบบ"""
        try:
            logger.info("Performing cleanup...")
            self.running = False
            
            if self.system:
                await self.system.stop()
            
            logger.info("Cleanup completed")
            print("\n👋 FVG Trading System stopped. Goodbye!")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

async def main():
    """Main function"""
    launcher = FVGSystemLauncher()
    await launcher.run()

if __name__ == "__main__":
    asyncio.run(main())
