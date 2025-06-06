# import asyncio
# import signal
# import sys
# from src.core.trading_bot import TradingBot

# def signal_handler(signum, frame):
    
#     print("\nReceived shutdown signal")
#     bot.stop()
    
#     sys.exit(0)

# async def main():
#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)
    
#     global bot
#     bot = TradingBot()
#     await bot.start()

# if __name__ == "__main__":
#     asyncio.run(main())