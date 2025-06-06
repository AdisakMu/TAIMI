
import asyncio
from datetime import datetime

class TradingBot:
    def __init__(self):
        self.running = False
        
    async def start(self):
        self.running = True
        print("Trading bot started...")
        print("Press Ctrl+C to stop")
        
        while self.running:
            try:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Current time: {current_time}")
                
                await asyncio.sleep(60)  # หน่วงเวลา 60 วินาที (1 นาที)
                
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(5)
    
    def stop(self):
        self.running = False
        print("Trading bot stopping...")
