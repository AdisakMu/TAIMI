"""
FVG Trading System - Main Package

Comprehensive FVG (Fair Value Gap) Breakout Trading System with:
- Real-time FVG detection and monitoring
- Claude AI signal analysis 
- MT5 order management
- PostgreSQL data storage
- Risk management
"""

# Import main components
from .analysis import ClaudeSignalAnalyzer
from .data import OHLCDatabase
from .trading import MT5OrderManager, OrderType, TradeRequest, TradeResult
from .strategies import FVGBreakoutStrategy, FVGData, TechnicalIndicators
from .realtime import FVGMonitor, FVGIntegrationSystem

__version__ = "1.0.0"

__all__ = [
    'ClaudeSignalAnalyzer',
    'OHLCDatabase', 
    'MT5OrderManager',
    'OrderType',
    'TradeRequest', 
    'TradeResult',
    'FVGBreakoutStrategy',
    'FVGData',
    'TechnicalIndicators',
    'FVGMonitor',
    'FVGIntegrationSystem'
]
