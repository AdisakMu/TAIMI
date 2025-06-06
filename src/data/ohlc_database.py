"""
OHLC Database Manager
Handles PostgreSQL database operations for OHLC data storage and retrieval
with automatic schema creation and data validation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

# Optional database import with fallback
try:
    import asyncpg  # type: ignore
    HAS_ASYNCPG = True
except ImportError:
    asyncpg = None
    HAS_ASYNCPG = False

# Optional MT5 import with fallback
try:
    import MetaTrader5 as mt5  # type: ignore
    HAS_MT5 = True
except ImportError:
    mt5 = None
    HAS_MT5 = False


@dataclass
class OHLCData:
    """OHLC data structure"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_volume: int = None
    spread: int = None


class OHLCDatabaseManager:
    """
    Manages PostgreSQL database operations for OHLC data
    Handles data insertion, retrieval, and schema management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database manager
        
        Args:
            config: Database configuration dictionary
        """
        self.config = config
        self.db_config = config.get('database', {})
        self.logger = logging.getLogger(__name__)
        self.pool = None  # Optional database connection pool
        
        # Database connection parameters
        self.host = self.db_config.get('host', 'localhost')
        self.port = self.db_config.get('port', 5432)
        self.user = self.db_config.get('user', 'trading_user')
        self.password = self.db_config.get('password', '')
        self.database = self.db_config.get('database', 'trading_db')
        
        # Validate database configuration
        if not HAS_ASYNCPG:
            self.logger.error("asyncpg not installed. Database operations will be disabled.")
            
        # Initialize MT5 if available
        self.mt5_initialized = False
        if HAS_MT5:
            self._initialize_mt5()
    
    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection if available"""
        try:
            if not mt5.initialize():
                self.logger.error("Failed to initialize MT5")
                return False
            
            # Login if credentials provided
            mt5_config = self.config.get('mt5', {})
            if all(key in mt5_config for key in ['account', 'password', 'server']):
                login_result = mt5.login(
                    login=mt5_config['account'],
                    password=mt5_config['password'],
                    server=mt5_config['server']
                )
                if not login_result:
                    self.logger.error("Failed to login to MT5")
                    return False
            
            self.mt5_initialized = True
            self.logger.info("MT5 initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5: {e}")
            return False
    
    async def connect(self) -> bool:
        """
        Establish database connection pool
        
        Returns:
            bool: True if connection successful
        """
        if not HAS_ASYNCPG:
            self.logger.error("Cannot connect: asyncpg not available")
            return False
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Test connection and create schema
            await self._create_schema()
            
            self.logger.info("Database connection pool created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database connection pool closed")
        
        # Shutdown MT5 if initialized
        if self.mt5_initialized and HAS_MT5:
            mt5.shutdown()
            self.logger.info("MT5 connection closed")
    
    async def _create_schema(self):
        """Create database schema if not exists"""
        if not self.pool:
            return
        
        schema_sql = """
        -- Create OHLC data table
        CREATE TABLE IF NOT EXISTS ohlc_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            open_price DECIMAL(10,5) NOT NULL,
            high_price DECIMAL(10,5) NOT NULL,
            low_price DECIMAL(10,5) NOT NULL,
            close_price DECIMAL(10,5) NOT NULL,
            volume BIGINT DEFAULT 0,
            tick_volume BIGINT DEFAULT 0,
            spread INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(symbol, timeframe, timestamp)
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_timeframe 
        ON ohlc_data(symbol, timeframe);
        
        CREATE INDEX IF NOT EXISTS idx_ohlc_timestamp 
        ON ohlc_data(timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_timestamp 
        ON ohlc_data(symbol, timestamp DESC);
        
        -- Create FVG analysis table
        CREATE TABLE IF NOT EXISTS fvg_analysis (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            fvg_type VARCHAR(10) NOT NULL, -- 'bullish' or 'bearish'
            gap_start_time TIMESTAMP WITH TIME ZONE NOT NULL,
            gap_end_time TIMESTAMP WITH TIME ZONE NOT NULL,
            gap_high DECIMAL(10,5) NOT NULL,
            gap_low DECIMAL(10,5) NOT NULL,
            gap_size_pips DECIMAL(6,2) NOT NULL,
            is_filled BOOLEAN DEFAULT FALSE,
            filled_at TIMESTAMP WITH TIME ZONE,
            confidence_score INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create index for FVG analysis
        CREATE INDEX IF NOT EXISTS idx_fvg_symbol_timeframe 
        ON fvg_analysis(symbol, timeframe);
        
        CREATE INDEX IF NOT EXISTS idx_fvg_timestamp 
        ON fvg_analysis(gap_start_time DESC);
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("Database schema created/verified successfully")
        except Exception as e:
            self.logger.error(f"Error creating database schema: {e}")
            raise
    
    async def insert_ohlc_data(self, ohlc_data: List[OHLCData]) -> bool:
        """
        Insert OHLC data into database
        
        Args:
            ohlc_data: List of OHLC data objects
            
        Returns:
            bool: True if insertion successful
        """
        if not self.pool or not ohlc_data:
            return False
        
        try:
            async with self.pool.acquire() as conn:
                # Prepare batch insert
                insert_sql = """
                INSERT INTO ohlc_data 
                (symbol, timeframe, timestamp, open_price, high_price, 
                 low_price, close_price, volume, tick_volume, spread)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (symbol, timeframe, timestamp) 
                DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    tick_volume = EXCLUDED.tick_volume,
                    spread = EXCLUDED.spread
                """
                
                # Prepare data for batch insert
                insert_data = [
                    (
                        data.symbol, data.timeframe, data.timestamp,
                        data.open, data.high, data.low, data.close,
                        data.volume or 0, data.tick_volume or 0, data.spread or 0
                    )
                    for data in ohlc_data
                ]
                
                # Execute batch insert
                await conn.executemany(insert_sql, insert_data)
                
            self.logger.info(f"Inserted {len(ohlc_data)} OHLC records")
            return True
            
        except Exception as e:
            self.logger.error(f"Error inserting OHLC data: {e}")
            return False
    
    async def get_ohlc_data(self, 
                           symbol: str, 
                           timeframe: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: Optional[int] = None) -> List[OHLCData]:
        """
        Retrieve OHLC data from database
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M5, M15, H1, etc.)
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            limit: Maximum number of records to retrieve
            
        Returns:
            List[OHLCData]: List of OHLC data objects
        """
        if not self.pool:
            return []
        
        try:
            # Build query
            query = """
            SELECT symbol, timeframe, timestamp, open_price, high_price,
                   low_price, close_price, volume, tick_volume, spread
            FROM ohlc_data
            WHERE symbol = $1 AND timeframe = $2
            """
            params = [symbol, timeframe]
            param_count = 2
            
            if start_time:
                param_count += 1
                query += f" AND timestamp >= ${param_count}"
                params.append(start_time)
            
            if end_time:
                param_count += 1
                query += f" AND timestamp <= ${param_count}"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                param_count += 1
                query += f" LIMIT ${param_count}"
                params.append(limit)
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
            
            # Convert rows to OHLCData objects
            ohlc_data = [
                OHLCData(
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    timestamp=row['timestamp'],
                    open=float(row['open_price']),
                    high=float(row['high_price']),
                    low=float(row['low_price']),
                    close=float(row['close_price']),
                    volume=row['volume'],
                    tick_volume=row['tick_volume'],
                    spread=row['spread']
                )
                for row in rows
            ]
            
            return ohlc_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving OHLC data: {e}")
            return []
    
    async def collect_ohlc_from_mt5(self, symbols: List[str], timeframes: List[str]) -> bool:
        """
        Collect OHLC data from MT5 and store in database
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
            
        Returns:
            bool: True if collection successful
        """
        if not self.mt5_initialized or not HAS_MT5:
            self.logger.error("MT5 not initialized")
            return False
        
        try:
            all_ohlc_data = []
            
            # MT5 timeframe mapping
            mt5_timeframes = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            for symbol in symbols:
                for timeframe in timeframes:
                    if timeframe not in mt5_timeframes:
                        continue
                    
                    # Get recent data (last 1000 candles)
                    rates = mt5.copy_rates_from_pos(
                        symbol,
                        mt5_timeframes[timeframe],
                        0,  # Start from current
                        1000  # Number of candles
                    )
                    
                    if rates is None or len(rates) == 0:
                        self.logger.warning(f"No data received for {symbol} {timeframe}")
                        continue
                    
                    # Convert MT5 data to OHLCData objects
                    for rate in rates:
                        ohlc = OHLCData(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=datetime.fromtimestamp(rate['time']),
                            open=rate['open'],
                            high=rate['high'],
                            low=rate['low'],
                            close=rate['close'],
                            volume=rate['tick_volume'],
                            tick_volume=rate['tick_volume'],
                            spread=rate['spread'] if 'spread' in rate else 0
                        )
                        all_ohlc_data.append(ohlc)
            
            # Insert collected data
            if all_ohlc_data:
                success = await self.insert_ohlc_data(all_ohlc_data)
                if success:
                    self.logger.info(f"Collected and stored {len(all_ohlc_data)} OHLC records")
                return success
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting OHLC data from MT5: {e}")
            return False
    
    async def get_latest_ohlc(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """
        Get latest OHLC data as pandas DataFrame
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            count: Number of latest records
            
        Returns:
            pd.DataFrame: OHLC data as DataFrame
        """
        ohlc_data = await self.get_ohlc_data(symbol, timeframe, limit=count)
        
        if not ohlc_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': data.timestamp,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume
            }
            for data in reversed(ohlc_data)  # Reverse to get chronological order
        ])
        
        df.set_index('timestamp', inplace=True)
        return df
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """
        Clean up old OHLC data beyond specified days
        
        Args:
            days_to_keep: Number of days to keep
            
        Returns:
            bool: True if cleanup successful
        """
        if not self.pool:
            return False
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM ohlc_data WHERE timestamp < $1",
                    cutoff_date
                )
                
                # Extract number of deleted rows
                deleted_count = int(result.split()[-1]) if result else 0
                
            self.logger.info(f"Cleaned up {deleted_count} old OHLC records")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return False
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dict: Database statistics
        """
        if not self.pool:
            return {}
        
        try:
            async with self.pool.acquire() as conn:
                # Get record counts
                total_records = await conn.fetchval("SELECT COUNT(*) FROM ohlc_data")
                
                # Get date range
                date_range = await conn.fetchrow("""
                    SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest
                    FROM ohlc_data
                """)
                
                # Get symbol counts
                symbol_stats = await conn.fetch("""
                    SELECT symbol, timeframe, COUNT(*) as count
                    FROM ohlc_data
                    GROUP BY symbol, timeframe
                    ORDER BY symbol, timeframe
                """)
                
            return {
                'total_records': total_records,
                'date_range': {
                    'earliest': date_range['earliest'] if date_range else None,
                    'latest': date_range['latest'] if date_range else None
                },
                'symbol_stats': [
                    {
                        'symbol': row['symbol'],
                        'timeframe': row['timeframe'],
                        'count': row['count']
                    }
                    for row in symbol_stats
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}


# Async context manager for database operations
class AsyncOHLCDatabase:
    """Async context manager for OHLC database operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.db_manager = OHLCDatabaseManager(config)
    
    async def __aenter__(self):
        await self.db_manager.connect()
        return self.db_manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.db_manager.disconnect()


# Standalone function for easy database operations
async def run_ohlc_collection(config: Dict[str, Any], 
                             symbols: List[str], 
                             timeframes: List[str]) -> bool:
    """
    Standalone function to run OHLC data collection
    
    Args:
        config: Configuration dictionary
        symbols: List of symbols to collect
        timeframes: List of timeframes to collect
        
    Returns:
        bool: True if collection successful
    """
    async with AsyncOHLCDatabase(config) as db:
        return await db.collect_ohlc_from_mt5(symbols, timeframes)


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load configuration
    try:
        with open('../config/config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config file not found. Please create config.json from template.")
        exit(1)
    
    # Example: Collect OHLC data
    symbols = config.get('symbols', ['EURUSD', 'GBPUSD'])
    timeframes = config.get('timeframes', ['M5', 'M15', 'H1'])
    
    async def main():
        success = await run_ohlc_collection(config, symbols, timeframes)
        if success:
            print("OHLC data collection completed successfully")
        else:
            print("OHLC data collection failed")
    
    # Run collection
    asyncio.run(main())