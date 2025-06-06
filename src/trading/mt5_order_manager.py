"""
MT5 Order Manager
จัดการการเทรดใน MetaTrader 5 พร้อม risk management และ position sizing
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

# Handle MT5 import with type ignore for missing stubs
try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    mt5 = None
    print("MetaTrader5 not installed. Please install: pip install MetaTrader5")

logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"

@dataclass
class TradeRequest:
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    stop_loss: float
    take_profit: float
    comment: str = ""
    magic_number: int = 12345

@dataclass
class TradeResult:
    success: bool
    order_id: Optional[int]
    message: str
    price_filled: Optional[float]
    volume_filled: Optional[float]
    timestamp: datetime

class MT5OrderManager:
    def __init__(self, account: int, password: str, server: str, 
                 max_risk_percent: float = 2.0):
        self.account = account
        self.password = password
        self.server = server
        self.max_risk_percent = max_risk_percent
        self.connected = False
        self.positions = {}
        self.orders = {}
        
        if mt5 is None:
            logger.error("MetaTrader5 not available")
            
    async def initialize(self) -> bool:
        """เริ่มต้นการเชื่อมต่อ MT5"""
        if mt5 is None:
            logger.error("MetaTrader5 not installed")
            return False
            
        try:
            if not mt5.initialize():  # type: ignore
                logger.error("MT5 initialization failed")
                return False
            
            # เชื่อมต่อบัญชี
            authorized = mt5.login(self.account, self.password, self.server)  # type: ignore
            if not authorized:
                logger.error(f"Login failed: {mt5.last_error()}")  # type: ignore
                return False
            
            self.connected = True
            account_info = mt5.account_info()  # type: ignore
            logger.info(f"Connected to MT5 - Account: {account_info.login}, "
                       f"Balance: {account_info.balance}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    def disconnect(self):
        """ปิดการเชื่อมต่อ MT5"""
        if mt5 is not None:
            mt5.shutdown()  # type: ignore
        self.connected = False
        logger.info("Disconnected from MT5")
    
    def get_account_info(self) -> Dict:
        """ดึงข้อมูลบัญชี"""
        if not self.connected or mt5 is None:
            return {}
        
        try:
            account = mt5.account_info()  # type: ignore
            return {
                'login': account.login,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'margin_free': account.margin_free,
                'margin_level': account.margin_level,
                'profit': account.profit
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, risk_amount: Optional[float] = None) -> float:
        """คำนวณขนาด position ตาม risk management"""
        if mt5 is None:
            return 0.01
            
        try:
            if not risk_amount:
                account = mt5.account_info()  # type: ignore
                risk_amount = account.balance * (self.max_risk_percent / 100)
            
            # ดึงข้อมูล symbol
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found")
                return 0.01
            
            # คำนวณ risk per unit
            price_diff = abs(entry_price - stop_loss)
            pip_value = symbol_info.trade_tick_value
            
            if price_diff > 0 and pip_value > 0:
                # คำนวณ position size
                position_size = risk_amount / (price_diff / symbol_info.point * pip_value)
                
                # ปรับให้อยู่ในขอบเขตที่อนุญาต
                min_lot = symbol_info.volume_min
                max_lot = symbol_info.volume_max
                lot_step = symbol_info.volume_step
                
                # ปรับ position size ให้เป็นทวีคูณของ lot_step
                position_size = round(position_size / lot_step) * lot_step
                position_size = max(min_lot, min(position_size, max_lot))
                
                return position_size
            
            return symbol_info.volume_min
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01
    
    async def place_market_order(self, trade_request: TradeRequest) -> TradeResult:
        """วาง market order"""
        if mt5 is None:
            return TradeResult(
                success=False,
                order_id=None,
                message="MetaTrader5 not available",
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )
            
        try:
            symbol_info = mt5.symbol_info(trade_request.symbol)  # type: ignore
            if not symbol_info:
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=f"Symbol {trade_request.symbol} not found",
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
            
            # เตรียม request
            if trade_request.order_type == OrderType.BUY:
                order_type = mt5.ORDER_TYPE_BUY  # type: ignore
                price = symbol_info.ask
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL  # type: ignore
                price = symbol_info.bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,  # type: ignore
                "symbol": trade_request.symbol,
                "volume": trade_request.volume,
                "type": order_type,
                "price": price,
                "sl": trade_request.stop_loss,
                "tp": trade_request.take_profit,
                "magic": trade_request.magic_number,
                "comment": trade_request.comment,
                "type_time": mt5.ORDER_TIME_GTC,  # type: ignore
                "type_filling": mt5.ORDER_FILLING_IOC,  # type: ignore
            }
            
            # ส่ง order
            result = mt5.order_send(request)  # type: ignore
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:  # type: ignore
                logger.info(f"Market order placed successfully: {result.order}")
                return TradeResult(
                    success=True,
                    order_id=result.order,
                    message="Order placed successfully",
                    price_filled=result.price,
                    volume_filled=result.volume,
                    timestamp=datetime.now()
                )
            else:
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                logger.error(error_msg)
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=error_msg,
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            error_msg = f"Error placing market order: {e}"
            logger.error(error_msg)
            return TradeResult(
                success=False,
                order_id=None,
                message=error_msg,
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )
    
    async def place_pending_order(self, trade_request: TradeRequest) -> TradeResult:
        """วาง pending order"""
        if mt5 is None:
            return TradeResult(
                success=False,
                order_id=None,
                message="MetaTrader5 not available",
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )
            
        try:
            symbol_info = mt5.symbol_info(trade_request.symbol)  # type: ignore
            if not symbol_info:
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=f"Symbol {trade_request.symbol} not found",
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
            
            # กำหนดประเภท order
            order_type_map = {
                OrderType.BUY_LIMIT: mt5.ORDER_TYPE_BUY_LIMIT,  # type: ignore
                OrderType.SELL_LIMIT: mt5.ORDER_TYPE_SELL_LIMIT,  # type: ignore
                OrderType.BUY_STOP: mt5.ORDER_TYPE_BUY_STOP,  # type: ignore
                OrderType.SELL_STOP: mt5.ORDER_TYPE_SELL_STOP  # type: ignore
            }
            
            if trade_request.order_type not in order_type_map:
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=f"Invalid pending order type: {trade_request.order_type}",
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
            
            request = {
                "action": mt5.TRADE_ACTION_PENDING,  # type: ignore
                "symbol": trade_request.symbol,
                "volume": trade_request.volume,
                "type": order_type_map[trade_request.order_type],
                "price": trade_request.price,
                "sl": trade_request.stop_loss,
                "tp": trade_request.take_profit,
                "magic": trade_request.magic_number,
                "comment": trade_request.comment,
                "type_time": mt5.ORDER_TIME_GTC,  # type: ignore
            }
            
            # ส่ง order
            result = mt5.order_send(request)  # type: ignore
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:  # type: ignore
                logger.info(f"Pending order placed successfully: {result.order}")
                return TradeResult(
                    success=True,
                    order_id=result.order,
                    message="Pending order placed successfully",
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
            else:
                error_msg = f"Pending order failed: {result.retcode} - {result.comment}"
                logger.error(error_msg)
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=error_msg,
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            error_msg = f"Error placing pending order: {e}"
            logger.error(error_msg)
            return TradeResult(
                success=False,
                order_id=None,
                message=error_msg,
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )
    
    def get_positions(self) -> List[Dict]:
        """ดึงรายการ positions ที่เปิดอยู่"""
        if mt5 is None:
            return []
            
        try:
            positions = mt5.positions_get()  # type: ignore
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == 0 else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'comment': pos.comment,
                    'magic': pos.magic,
                    'time': datetime.fromtimestamp(pos.time)
                })
            
            return position_list
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_orders(self) -> List[Dict]:
        """ดึงรายการ pending orders"""
        if mt5 is None:
            return []
            
        try:
            orders = mt5.orders_get()  # type: ignore
            if orders is None:
                return []
            
            order_list = []
            for order in orders:
                order_type_map = {
                    getattr(mt5, 'ORDER_TYPE_BUY_LIMIT', None): 'BUY_LIMIT',
                    getattr(mt5, 'ORDER_TYPE_SELL_LIMIT', None): 'SELL_LIMIT',
                    getattr(mt5, 'ORDER_TYPE_BUY_STOP', None): 'BUY_STOP',
                    getattr(mt5, 'ORDER_TYPE_SELL_STOP', None): 'SELL_STOP'
                }
                
                order_list.append({
                    'ticket': order.ticket,
                    'symbol': order.symbol,
                    'type': order_type_map.get(order.type, 'UNKNOWN'),
                    'volume': order.volume_initial,
                    'price_open': order.price_open,
                    'sl': order.sl,
                    'tp': order.tp,
                    'comment': order.comment,
                    'magic': order.magic,
                    'time_setup': datetime.fromtimestamp(order.time_setup)
                })
            
            return order_list
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    async def close_position(self, ticket: int) -> TradeResult:
        """ปิด position"""
        if mt5 is None:
            return TradeResult(
                success=False,
                order_id=None,
                message="MetaTrader5 not available",
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )
            
        try:
            positions = mt5.positions_get(ticket=ticket)  # type: ignore
            if not positions:
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=f"Position {ticket} not found",
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
            
            position = positions[0]
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
            
            if position.type == 0:  # BUY position
                order_type = mt5.ORDER_TYPE_SELL  # type: ignore
                price = symbol_info.bid
            else:  # SELL position
                order_type = mt5.ORDER_TYPE_BUY  # type: ignore
                price = symbol_info.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,  # type: ignore
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "comment": "Close by FVG system",
                "type_time": mt5.ORDER_TIME_GTC,  # type: ignore
                "type_filling": mt5.ORDER_FILLING_IOC,  # type: ignore
            }
            
            result = mt5.order_send(request)  # type: ignore
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:  # type: ignore
                logger.info(f"Position {ticket} closed successfully")
                return TradeResult(
                    success=True,
                    order_id=result.order,
                    message="Position closed successfully",
                    price_filled=result.price,
                    volume_filled=result.volume,
                    timestamp=datetime.now()
                )
            else:
                error_msg = f"Failed to close position: {result.retcode} - {result.comment}"
                logger.error(error_msg)
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=error_msg,
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            error_msg = f"Error closing position: {e}"
            logger.error(error_msg)
            return TradeResult(
                success=False,
                order_id=None,
                message=error_msg,
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )
    
    async def cancel_order(self, ticket: int) -> TradeResult:
        """ยกเลิก pending order"""
        if mt5 is None:
            return TradeResult(
                success=False,
                order_id=None,
                message="MetaTrader5 not available",
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )
            
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,  # type: ignore
                "order": ticket,
            }
            
            result = mt5.order_send(request)  # type: ignore
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:  # type: ignore
                logger.info(f"Order {ticket} cancelled successfully")
                return TradeResult(
                    success=True,
                    order_id=ticket,
                    message="Order cancelled successfully",
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
            else:
                error_msg = f"Failed to cancel order: {result.retcode} - {result.comment}"
                logger.error(error_msg)
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=error_msg,
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            error_msg = f"Error cancelling order: {e}"
            logger.error(error_msg)
            return TradeResult(
                success=False,
                order_id=None,
                message=error_msg,
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )
    
    def modify_position(self, ticket: int, new_sl: Optional[float] = None, 
                       new_tp: Optional[float] = None) -> TradeResult:
        """แก้ไข SL/TP ของ position"""
        if mt5 is None:
            return TradeResult(
                success=False,
                order_id=None,
                message="MetaTrader5 not available",
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )
            
        try:
            positions = mt5.positions_get(ticket=ticket)  # type: ignore
            if not positions:
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=f"Position {ticket} not found",
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
            
            position = positions[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,  # type: ignore
                "symbol": position.symbol,
                "position": ticket,
                "sl": new_sl if new_sl else position.sl,
                "tp": new_tp if new_tp else position.tp,
            }
            
            result = mt5.order_send(request)  # type: ignore
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:  # type: ignore
                logger.info(f"Position {ticket} modified successfully")
                return TradeResult(
                    success=True,
                    order_id=ticket,
                    message="Position modified successfully",
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
            else:
                error_msg = f"Failed to modify position: {result.retcode} - {result.comment}"
                logger.error(error_msg)
                return TradeResult(
                    success=False,
                    order_id=None,
                    message=error_msg,
                    price_filled=None,
                    volume_filled=None,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            error_msg = f"Error modifying position: {e}"
            logger.error(error_msg)
            return TradeResult(
                success=False,
                order_id=None,
                message=error_msg,
                price_filled=None,
                volume_filled=None,
                timestamp=datetime.now()
            )

# ตัวอย่างการใช้งาน
async def main():
    # ตั้งค่าการเชื่อมต่อ MT5
    account = 123456789
    password = "your_password"
    server = "your_broker_server"
    
    order_manager = MT5OrderManager(account, password, server, max_risk_percent=2.0)
    
    try:
        # เชื่อมต่อ MT5
        if await order_manager.initialize():
            print("Connected to MT5 successfully")
            
            # ดูข้อมูลบัญชี
            account_info = order_manager.get_account_info()
            print(f"Account Balance: {account_info.get('balance', 0)}")
            
            # คำนวณ position size
            position_size = order_manager.calculate_position_size(
                "EURUSD", 1.10450, 1.10400
            )
            print(f"Calculated position size: {position_size}")
            
            # สร้าง trade request
            trade_request = TradeRequest(
                symbol="EURUSD",
                order_type=OrderType.BUY,
                volume=position_size,
                price=1.10450,
                stop_loss=1.10400,
                take_profit=1.10550,
                comment="FVG Breakout Trade"
            )
            
            # วาง market order (ในการใช้งานจริง)
            # result = await order_manager.place_market_order(trade_request)
            # print(f"Trade result: {result}")
            
            # ดู positions และ orders
            positions = order_manager.get_positions()
            orders = order_manager.get_orders()
            
            print(f"Open positions: {len(positions)}")
            print(f"Pending orders: {len(orders)}")
            
        else:
            print("Failed to connect to MT5")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        order_manager.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
