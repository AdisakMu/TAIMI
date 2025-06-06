"""
Claude Signal Analyzer
Integrates with Claude API for advanced FVG signal analysis and scoring
Provides intelligent signal evaluation with 0-100 confidence scoring.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

# Optional HTTP client import with fallback
try:
    import aiohttp  # type: ignore
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False


@dataclass
class FVGSignal:
    """FVG signal data structure"""
    symbol: str
    timeframe: str
    timestamp: datetime
    fvg_type: str  # 'bullish' or 'bearish'
    gap_high: float
    gap_low: float
    gap_size_pips: float
    current_price: float
    
    # Technical indicators
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    support_resistance_level: Optional[float] = None
    
    # Market context
    trend_direction: Optional[str] = None
    volatility_level: Optional[str] = None
    volume_analysis: Optional[str] = None


@dataclass
class SignalAnalysis:
    """Signal analysis result structure"""
    confidence_score: int  # 0-100
    recommendation: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    risk_level: str  # 'low', 'medium', 'high'
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    key_factors: List[str] = None
    
    def __post_init__(self):
        if self.key_factors is None:
            self.key_factors = []


class ClaudeSignalAnalyzer:
    """
    Advanced signal analysis using Claude API
    Provides intelligent FVG signal evaluation and scoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Claude Signal Analyzer
        
        Args:
            config: Configuration dictionary containing Claude API settings
        """
        self.config = config
        self.claude_config = config.get('claude', {})
        self.logger = logging.getLogger(__name__)
        
        # Claude API configuration
        self.api_key = self.claude_config.get('api_key', '')
        self.model = self.claude_config.get('model', 'claude-3-sonnet-20240229')
        self.api_url = "https://api.anthropic.com/v1/messages"
        
        # Analysis parameters
        self.min_confidence_score = config.get('trading_settings', {}).get('min_confidence_score', 70)
        
        # Validate configuration
        if not HAS_AIOHTTP:
            self.logger.error("aiohttp not installed. Claude API integration will be disabled.")
        
        if not self.api_key:
            self.logger.warning("Claude API key not provided. Using fallback analysis.")
    
    async def analyze_fvg_signal(self, signal: FVGSignal, market_data: pd.DataFrame) -> SignalAnalysis:
        """
        Analyze FVG signal using Claude API
        
        Args:
            signal: FVG signal to analyze
            market_data: Historical market data for context
            
        Returns:
            SignalAnalysis: Detailed analysis with confidence score
        """
        try:
            # Try Claude API analysis first
            if HAS_AIOHTTP and self.api_key:
                analysis = await self._analyze_with_claude(signal, market_data)
                if analysis:
                    return analysis
            
            # Fallback to rule-based analysis
            self.logger.info("Using fallback analysis for FVG signal")
            return await self._fallback_analysis(signal, market_data)
            
        except Exception as e:
            self.logger.error(f"Error analyzing FVG signal: {e}")
            return self._create_neutral_analysis(signal)
    
    async def _analyze_with_claude(self, signal: FVGSignal, market_data: pd.DataFrame) -> Optional[SignalAnalysis]:
        """
        Analyze signal using Claude API
        
        Args:
            signal: FVG signal data
            market_data: Market context data
            
        Returns:
            Optional[SignalAnalysis]: Analysis result or None if failed
        """
        try:
            # Prepare analysis prompt
            prompt = self._create_analysis_prompt(signal, market_data)
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': self.api_key,
                    'anthropic-version': '2023-06-01'
                }
                
                payload = {
                    'model': self.model,
                    'max_tokens': 1500,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'temperature': 0.3
                }
                
                async with session.post(self.api_url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_claude_response(result, signal)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Claude API error {response.status}: {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            self.logger.error("Claude API request timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error calling Claude API: {e}")
            return None
    
    def _create_analysis_prompt(self, signal: FVGSignal, market_data: pd.DataFrame) -> str:
        """
        Create detailed analysis prompt for Claude
        
        Args:
            signal: FVG signal data
            market_data: Market context data
            
        Returns:
            str: Formatted analysis prompt
        """
        # Calculate recent price action
        recent_high = market_data['high'].tail(20).max() if len(market_data) > 0 else 0
        recent_low = market_data['low'].tail(20).min() if len(market_data) > 0 else 0
        price_range = recent_high - recent_low
        
        # Get latest candle info
        latest_candle = market_data.iloc[-1] if len(market_data) > 0 else None
        
        prompt = f"""
You are an expert forex trader analyzing a Fair Value Gap (FVG) trading signal. 
Please provide a detailed analysis and assign a confidence score from 0-100.

SIGNAL DETAILS:
- Symbol: {signal.symbol}
- Timeframe: {signal.timeframe}
- FVG Type: {signal.fvg_type.upper()}
- Gap Range: {signal.gap_low:.5f} - {signal.gap_high:.5f}
- Gap Size: {signal.gap_size_pips:.1f} pips
- Current Price: {signal.current_price:.5f}
- Timestamp: {signal.timestamp}

TECHNICAL INDICATORS:
- RSI: {signal.rsi if signal.rsi else 'N/A'}
- MACD Signal: {signal.macd_signal if signal.macd_signal else 'N/A'}
- Support/Resistance: {signal.support_resistance_level if signal.support_resistance_level else 'N/A'}

MARKET CONTEXT:
- Recent High: {recent_high:.5f}
- Recent Low: {recent_low:.5f}
- Price Range (20 periods): {price_range:.5f}
- Trend Direction: {signal.trend_direction if signal.trend_direction else 'N/A'}
- Volatility Level: {signal.volatility_level if signal.volatility_level else 'N/A'}

LATEST CANDLE INFO:
{f"- Open: {latest_candle['open']:.5f}" if latest_candle is not None else "- No recent data available"}
{f"- High: {latest_candle['high']:.5f}" if latest_candle is not None else ""}
{f"- Low: {latest_candle['low']:.5f}" if latest_candle is not None else ""}
{f"- Close: {latest_candle['close']:.5f}" if latest_candle is not None else ""}

Please analyze this FVG signal and provide your response in the following JSON format:

{{
    "confidence_score": <integer 0-100>,
    "recommendation": "<strong_buy|buy|hold|sell|strong_sell>",
    "risk_level": "<low|medium|high>",
    "entry_price": <float or null>,
    "stop_loss": <float or null>,
    "take_profit": <float or null>,
    "reasoning": "<detailed explanation>",
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"]
}}

Consider these factors in your analysis:
1. FVG gap quality and size significance
2. Current price position relative to the gap
3. Technical indicator confluence
4. Market structure and trend alignment
5. Risk-reward ratio potential
6. Market volatility and timing
7. Support/resistance level interactions
"""
        
        return prompt
    
    def _parse_claude_response(self, response: Dict[str, Any], signal: FVGSignal) -> Optional[SignalAnalysis]:
        """
        Parse Claude API response into SignalAnalysis object
        
        Args:
            response: Claude API response
            signal: Original signal data
            
        Returns:
            Optional[SignalAnalysis]: Parsed analysis or None if failed
        """
        try:
            # Extract content from Claude response
            content = response.get('content', [])
            if not content:
                return None
            
            # Get the text response
            text_response = content[0].get('text', '') if content else ''
            
            # Try to extract JSON from the response
            json_start = text_response.find('{')
            json_end = text_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = text_response[json_start:json_end]
                analysis_data = json.loads(json_str)
                
                # Create SignalAnalysis object
                return SignalAnalysis(
                    confidence_score=max(0, min(100, analysis_data.get('confidence_score', 50))),
                    recommendation=analysis_data.get('recommendation', 'hold'),
                    risk_level=analysis_data.get('risk_level', 'medium'),
                    entry_price=analysis_data.get('entry_price'),
                    stop_loss=analysis_data.get('stop_loss'),
                    take_profit=analysis_data.get('take_profit'),
                    reasoning=analysis_data.get('reasoning', ''),
                    key_factors=analysis_data.get('key_factors', [])
                )
            
            return None
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Claude response JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing Claude response: {e}")
            return None
    
    async def _fallback_analysis(self, signal: FVGSignal, market_data: pd.DataFrame) -> SignalAnalysis:
        """
        Rule-based fallback analysis when Claude API is unavailable
        
        Args:
            signal: FVG signal data
            market_data: Market context data
            
        Returns:
            SignalAnalysis: Rule-based analysis result
        """
        try:
            # Initialize scoring components
            confidence_score = 50  # Base score
            key_factors = []
            reasoning_parts = []
            
            # Analyze gap size significance
            if signal.gap_size_pips >= 5:
                confidence_score += 15
                key_factors.append("Significant gap size")
                reasoning_parts.append(f"Gap size of {signal.gap_size_pips:.1f} pips is significant")
            elif signal.gap_size_pips >= 2:
                confidence_score += 8
                key_factors.append("Moderate gap size")
                reasoning_parts.append(f"Gap size of {signal.gap_size_pips:.1f} pips is moderate")
            
            # Analyze price position relative to gap
            gap_center = (signal.gap_high + signal.gap_low) / 2
            price_vs_gap = self._analyze_price_gap_relationship(signal.current_price, signal.gap_low, signal.gap_high, signal.fvg_type)
            
            if price_vs_gap['favorable']:
                confidence_score += 10
                key_factors.append("Favorable price position")
                reasoning_parts.append(price_vs_gap['reason'])
            
            # Analyze RSI if available
            if signal.rsi is not None:
                rsi_analysis = self._analyze_rsi(signal.rsi, signal.fvg_type)
                confidence_score += rsi_analysis['score_adjustment']
                if rsi_analysis['factor']:
                    key_factors.append(rsi_analysis['factor'])
                    reasoning_parts.append(rsi_analysis['reason'])
            
            # Analyze MACD if available
            if signal.macd_signal:
                macd_analysis = self._analyze_macd(signal.macd_signal, signal.fvg_type)
                confidence_score += macd_analysis['score_adjustment']
                if macd_analysis['factor']:
                    key_factors.append(macd_analysis['factor'])
                    reasoning_parts.append(macd_analysis['reason'])
            
            # Analyze trend alignment
            if signal.trend_direction:
                trend_analysis = self._analyze_trend_alignment(signal.trend_direction, signal.fvg_type)
                confidence_score += trend_analysis['score_adjustment']
                if trend_analysis['factor']:
                    key_factors.append(trend_analysis['factor'])
                    reasoning_parts.append(trend_analysis['reason'])
            
            # Analyze market volatility
            if signal.volatility_level:
                volatility_analysis = self._analyze_volatility(signal.volatility_level)
                confidence_score += volatility_analysis['score_adjustment']
                if volatility_analysis['factor']:
                    key_factors.append(volatility_analysis['factor'])
                    reasoning_parts.append(volatility_analysis['reason'])
            
            # Ensure confidence score is within bounds
            confidence_score = max(0, min(100, confidence_score))
            
            # Determine recommendation based on confidence score
            recommendation = self._determine_recommendation(confidence_score, signal.fvg_type)
            
            # Determine risk level
            risk_level = self._determine_risk_level(confidence_score, signal.volatility_level)
            
            # Calculate entry, stop loss, and take profit levels
            entry_levels = self._calculate_entry_levels(signal)
            
            return SignalAnalysis(
                confidence_score=confidence_score,
                recommendation=recommendation,
                risk_level=risk_level,
                entry_price=entry_levels['entry'],
                stop_loss=entry_levels['stop_loss'],
                take_profit=entry_levels['take_profit'],
                reasoning=" | ".join(reasoning_parts) if reasoning_parts else "Standard rule-based analysis applied",
                key_factors=key_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error in fallback analysis: {e}")
            return self._create_neutral_analysis(signal)
    
    def _analyze_price_gap_relationship(self, current_price: float, gap_low: float, gap_high: float, fvg_type: str) -> Dict[str, Any]:
        """Analyze price position relative to FVG"""
        if fvg_type == 'bullish':
            if current_price <= gap_low:
                return {
                    'favorable': True,
                    'reason': f"Price ({current_price:.5f}) is below bullish FVG, good for entry"
                }
            elif gap_low < current_price < gap_high:
                return {
                    'favorable': False,
                    'reason': f"Price ({current_price:.5f}) is within the gap, wait for better entry"
                }
        else:  # bearish
            if current_price >= gap_high:
                return {
                    'favorable': True,
                    'reason': f"Price ({current_price:.5f}) is above bearish FVG, good for entry"
                }
            elif gap_low < current_price < gap_high:
                return {
                    'favorable': False,
                    'reason': f"Price ({current_price:.5f}) is within the gap, wait for better entry"
                }
        
        return {'favorable': False, 'reason': "Price position not optimal for FVG trade"}
    
    def _analyze_rsi(self, rsi: float, fvg_type: str) -> Dict[str, Any]:
        """Analyze RSI indicator"""
        if fvg_type == 'bullish':
            if rsi < 30:
                return {
                    'score_adjustment': 15,
                    'factor': "RSI oversold support",
                    'reason': f"RSI ({rsi:.1f}) shows oversold conditions supporting bullish FVG"
                }
            elif rsi < 50:
                return {
                    'score_adjustment': 8,
                    'factor': "RSI below midline",
                    'reason': f"RSI ({rsi:.1f}) below 50 suggests potential upside"
                }
            elif rsi > 70:
                return {
                    'score_adjustment': -10,
                    'factor': "RSI overbought warning",
                    'reason': f"RSI ({rsi:.1f}) overbought, reduces bullish confidence"
                }
        else:  # bearish
            if rsi > 70:
                return {
                    'score_adjustment': 15,
                    'factor': "RSI overbought support",
                    'reason': f"RSI ({rsi:.1f}) shows overbought conditions supporting bearish FVG"
                }
            elif rsi > 50:
                return {
                    'score_adjustment': 8,
                    'factor': "RSI above midline",
                    'reason': f"RSI ({rsi:.1f}) above 50 suggests potential downside"
                }
            elif rsi < 30:
                return {
                    'score_adjustment': -10,
                    'factor': "RSI oversold warning",
                    'reason': f"RSI ({rsi:.1f}) oversold, reduces bearish confidence"
                }
        
        return {'score_adjustment': 0, 'factor': None, 'reason': "RSI neutral"}
    
    def _analyze_macd(self, macd_signal: str, fvg_type: str) -> Dict[str, Any]:
        """Analyze MACD signal"""
        if (fvg_type == 'bullish' and macd_signal == 'bullish') or \
           (fvg_type == 'bearish' and macd_signal == 'bearish'):
            return {
                'score_adjustment': 12,
                'factor': "MACD confluence",
                'reason': f"MACD {macd_signal} signal aligns with FVG direction"
            }
        elif macd_signal == 'neutral':
            return {
                'score_adjustment': 0,
                'factor': None,
                'reason': "MACD neutral"
            }
        else:
            return {
                'score_adjustment': -8,
                'factor': "MACD divergence",
                'reason': f"MACD {macd_signal} signal conflicts with FVG direction"
            }
    
    def _analyze_trend_alignment(self, trend_direction: str, fvg_type: str) -> Dict[str, Any]:
        """Analyze trend alignment"""
        if (fvg_type == 'bullish' and trend_direction == 'uptrend') or \
           (fvg_type == 'bearish' and trend_direction == 'downtrend'):
            return {
                'score_adjustment': 15,
                'factor': "Trend alignment",
                'reason': f"FVG aligns with {trend_direction}"
            }
        elif trend_direction == 'sideways':
            return {
                'score_adjustment': -5,
                'factor': "Sideways market",
                'reason': "Sideways trend reduces FVG effectiveness"
            }
        else:
            return {
                'score_adjustment': -12,
                'factor': "Counter-trend trade",
                'reason': f"FVG against {trend_direction}, higher risk"
            }
    
    def _analyze_volatility(self, volatility_level: str) -> Dict[str, Any]:
        """Analyze market volatility impact"""
        if volatility_level == 'high':
            return {
                'score_adjustment': -8,
                'factor': "High volatility risk",
                'reason': "High volatility increases trade risk"
            }
        elif volatility_level == 'medium':
            return {
                'score_adjustment': 5,
                'factor': "Moderate volatility",
                'reason': "Moderate volatility favorable for FVG trades"
            }
        else:  # low
            return {
                'score_adjustment': 2,
                'factor': "Low volatility",
                'reason': "Low volatility reduces noise"
            }
    
    def _determine_recommendation(self, confidence_score: int, fvg_type: str) -> str:
        """Determine trading recommendation based on confidence score"""
        if confidence_score >= 80:
            return f"strong_{'buy' if fvg_type == 'bullish' else 'sell'}"
        elif confidence_score >= 65:
            return f"{'buy' if fvg_type == 'bullish' else 'sell'}"
        elif confidence_score >= 40:
            return "hold"
        elif confidence_score >= 25:
            return f"{'sell' if fvg_type == 'bullish' else 'buy'}"
        else:
            return f"strong_{'sell' if fvg_type == 'bullish' else 'buy'}"
    
    def _determine_risk_level(self, confidence_score: int, volatility_level: Optional[str]) -> str:
        """Determine risk level based on confidence and volatility"""
        if confidence_score >= 75 and volatility_level != 'high':
            return "low"
        elif confidence_score >= 50 and volatility_level != 'high':
            return "medium"
        else:
            return "high"
    
    def _calculate_entry_levels(self, signal: FVGSignal) -> Dict[str, Optional[float]]:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            gap_size = signal.gap_high - signal.gap_low
            
            if signal.fvg_type == 'bullish':
                # For bullish FVG, enter near gap low, target above gap high
                entry = signal.gap_low - (gap_size * 0.1)  # Slight buffer below gap
                stop_loss = signal.gap_low - (gap_size * 0.5)  # Stop below gap with buffer
                take_profit = signal.gap_high + (gap_size * 1.5)  # Target above gap
            else:  # bearish
                # For bearish FVG, enter near gap high, target below gap low
                entry = signal.gap_high + (gap_size * 0.1)  # Slight buffer above gap
                stop_loss = signal.gap_high + (gap_size * 0.5)  # Stop above gap with buffer
                take_profit = signal.gap_low - (gap_size * 1.5)  # Target below gap
            
            return {
                'entry': round(entry, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating entry levels: {e}")
            return {'entry': None, 'stop_loss': None, 'take_profit': None}
    
    def _create_neutral_analysis(self, signal: FVGSignal) -> SignalAnalysis:
        """Create neutral analysis when other methods fail"""
        return SignalAnalysis(
            confidence_score=50,
            recommendation='hold',
            risk_level='medium',
            reasoning="Unable to perform detailed analysis, using neutral assessment",
            key_factors=["Neutral analysis applied"]
        )
    
    async def batch_analyze_signals(self, signals: List[FVGSignal], market_data_dict: Dict[str, pd.DataFrame]) -> List[SignalAnalysis]:
        """
        Analyze multiple FVG signals in batch
        
        Args:
            signals: List of FVG signals to analyze
            market_data_dict: Dictionary of market data keyed by symbol
            
        Returns:
            List[SignalAnalysis]: List of analysis results
        """
        analyses = []
        
        for signal in signals:
            try:
                # Get market data for this signal's symbol
                market_data = market_data_dict.get(signal.symbol, pd.DataFrame())
                
                # Analyze the signal
                analysis = await self.analyze_fvg_signal(signal, market_data)
                analyses.append(analysis)
                
                # Add small delay to avoid overwhelming API
                if HAS_AIOHTTP and self.api_key:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing signal for {signal.symbol}: {e}")
                analyses.append(self._create_neutral_analysis(signal))
        
        return analyses
    
    def filter_high_confidence_signals(self, analyses: List[SignalAnalysis]) -> List[SignalAnalysis]:
        """
        Filter signals based on minimum confidence threshold
        
        Args:
            analyses: List of signal analyses
            
        Returns:
            List[SignalAnalysis]: Filtered high-confidence signals
        """
        return [
            analysis for analysis in analyses 
            if analysis.confidence_score >= self.min_confidence_score
        ]
    
    async def get_api_status(self) -> Dict[str, Any]:
        """
        Check Claude API status and availability
        
        Returns:
            Dict: API status information
        """
        status = {
            'aiohttp_available': HAS_AIOHTTP,
            'api_key_configured': bool(self.api_key),
            'api_accessible': False,
            'model': self.model,
            'fallback_mode': not (HAS_AIOHTTP and self.api_key)
        }
        
        # Test API accessibility if configured
        if HAS_AIOHTTP and self.api_key:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'x-api-key': self.api_key,
                        'anthropic-version': '2023-06-01'
                    }
                    
                    # Simple API test (this might need adjustment based on actual Claude API)
                    async with session.get('https://api.anthropic.com/v1/messages', 
                                         headers=headers, timeout=10) as response:
                        status['api_accessible'] = response.status in [200, 400, 401]  # 400/401 means API is reachable
                        
            except Exception as e:
                self.logger.debug(f"API test failed: {e}")
                status['api_accessible'] = False
        
        return status


# Convenience function for standalone analysis
async def analyze_single_fvg(config: Dict[str, Any], 
                           signal: FVGSignal, 
                           market_data: pd.DataFrame) -> SignalAnalysis:
    """
    Analyze a single FVG signal using Claude
    
    Args:
        config: Configuration dictionary
        signal: FVG signal to analyze
        market_data: Market context data
        
    Returns:
        SignalAnalysis: Analysis result
    """
    analyzer = ClaudeSignalAnalyzer(config)
    return await analyzer.analyze_fvg_signal(signal, market_data)


if __name__ == "__main__":
    # Example usage
    import json
    from datetime import datetime
    
    # Load configuration
    try:
        with open('../config/config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config file not found. Please create config.json from template.")
        exit(1)
    
    # Example FVG signal
    test_signal = FVGSignal(
        symbol='EURUSD',
        timeframe='M15',
        timestamp=datetime.now(),
        fvg_type='bullish',
        gap_high=1.0850,
        gap_low=1.0830,
        gap_size_pips=2.0,
        current_price=1.0825,
        rsi=35.5,
        macd_signal='bullish',
        trend_direction='uptrend',
        volatility_level='medium'
    )
    
    # Create sample market data
    sample_data = pd.DataFrame({
        'open': [1.0820, 1.0825, 1.0830],
        'high': [1.0835, 1.0840, 1.0845],
        'low': [1.0815, 1.0820, 1.0825],
        'close': [1.0825, 1.0830, 1.0835],
        'volume': [1000, 1100, 1200]
    })
    
    async def main():
        analysis = await analyze_single_fvg(config, test_signal, sample_data)
        print(f"Analysis Result:")
        print(f"Confidence Score: {analysis.confidence_score}")
        print(f"Recommendation: {analysis.recommendation}")
        print(f"Risk Level: {analysis.risk_level}")
        print(f"Reasoning: {analysis.reasoning}")
        print(f"Key Factors: {', '.join(analysis.key_factors)}")
    
    # Run analysis
    asyncio.run(main())