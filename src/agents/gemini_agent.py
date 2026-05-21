"""
Gemini AI Agent for Trading Signal Explanations

Provides natural language explanations for trading signals
using Google's Gemini API.
"""

import os
from typing import Dict, List, Optional


class GeminiTradingAgent:
    """
    AI agent that provides explanations for trading signals.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini agent.
        
        Args:
            api_key: Gemini API key. If not provided, uses GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
            except ImportError:
                print("google-generativeai not installed. Run: pip install google-generativeai")
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")
    
    def explain_signal(self, signal: Dict) -> str:
        """
        Generate a natural language explanation for a trading signal.
        
        Args:
            signal: Dictionary containing signal data
            
        Returns:
            Human-readable explanation string
        """
        if not self.model:
            return self._generate_template_explanation(signal)
        
        try:
            prompt = self._build_explanation_prompt(signal)
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return self._generate_template_explanation(signal)
    
    def _build_explanation_prompt(self, signal: Dict) -> str:
        """Build a prompt for signal explanation."""
        return f"""
You are a quantitative trading analyst. Explain this trading signal concisely in 2-3 sentences:

Ticker: {signal.get('ticker', 'N/A')}
Current Price: ${signal.get('price', 0):.2f}
Signal: {signal.get('soft_decision', signal.get('decision', 'N/A'))}
Signal Strength: {signal.get('signal_strength', 0):.0f}/100
Expected 30-day Return: {signal.get('exp_30d_return', 0)*100:+.1f}%
Confidence: {signal.get('confidence', 0)*100:.1f}%
ADX (Trend Strength): {signal.get('adx14', 0):.1f}
60-day Momentum: {signal.get('mom_60', 0)*100:+.1f}%
RSI: {signal.get('rsi14', 50):.1f}

Explain why this signal was generated and key factors to consider.
Keep the explanation professional and actionable.
"""
    
    def _generate_template_explanation(self, signal: Dict) -> str:
        """Generate a template-based explanation when AI is unavailable."""
        ticker = signal.get('ticker', 'N/A')
        decision = signal.get('soft_decision', signal.get('decision', 'HOLD'))
        strength = signal.get('signal_strength', 0)
        exp_ret = signal.get('exp_30d_return', 0) * 100
        conf = signal.get('confidence', 0) * 100
        adx = signal.get('adx14', 0)
        mom = signal.get('mom_60', 0) * 100
        
        # Build explanation
        if decision == "LONG":
            direction = "bullish"
            action = "buy"
        elif decision == "SHORT":
            direction = "bearish"
            action = "sell"
        else:
            direction = "neutral"
            action = "hold"
        
        # Strength description
        if strength > 70:
            strength_desc = "strong"
        elif strength > 40:
            strength_desc = "moderate"
        else:
            strength_desc = "weak"
        
        # Trend description
        if adx > 25:
            trend_desc = "strong trending"
        elif adx > 15:
            trend_desc = "moderate trending"
        else:
            trend_desc = "range-bound"
        
        return (
            f"{ticker} shows a {strength_desc} {direction} signal with {conf:.0f}% confidence. "
            f"Expected 30-day return of {exp_ret:+.1f}% in a {trend_desc} market "
            f"with {mom:+.1f}% 60-day momentum. "
            f"Consider {action}ing with appropriate risk management."
        )
    
    def get_market_context(self, signals: List[Dict]) -> str:
        """
        Generate overall market context from all signals.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Market context string
        """
        if not signals:
            return "No signals available for market context."
        
        # Calculate aggregate metrics
        bullish = sum(1 for s in signals if s.get('soft_decision') == 'LONG')
        bearish = sum(1 for s in signals if s.get('soft_decision') == 'SHORT')
        neutral = len(signals) - bullish - bearish
        
        avg_strength = sum(s.get('signal_strength', 0) for s in signals) / len(signals)
        avg_conf = sum(s.get('confidence', 0) for s in signals) / len(signals)
        avg_exp_ret = sum(s.get('exp_30d_return', 0) for s in signals) / len(signals)
        
        # Determine market sentiment
        if bullish > bearish * 1.5:
            sentiment = "bullish"
        elif bearish > bullish * 1.5:
            sentiment = "bearish"
        else:
            sentiment = "mixed"
        
        if not self.model:
            return (
                f"Market shows {sentiment} sentiment: {bullish} bullish, {bearish} bearish, "
                f"{neutral} neutral signals. Average signal strength: {avg_strength:.0f}/100. "
                f"Average expected return: {avg_exp_ret*100:+.1f}%."
            )
        
        try:
            prompt = f"""
Summarize the market outlook in 2 sentences based on these metrics:
- Bullish signals: {bullish}
- Bearish signals: {bearish}
- Neutral signals: {neutral}
- Average signal strength: {avg_strength:.0f}/100
- Average confidence: {avg_conf*100:.1f}%
- Average expected return: {avg_exp_ret*100:+.1f}%

Provide a brief, professional market context.
"""
            response = self.model.generate_content(prompt)
            return response.text
        except:
            return (
                f"Market shows {sentiment} sentiment with {bullish} bullish and {bearish} bearish signals. "
                f"Average signal strength: {avg_strength:.0f}/100."
            )


def create_agent(api_key: Optional[str] = None) -> GeminiTradingAgent:
    """
    Factory function to create a Gemini trading agent.
    
    Args:
        api_key: Optional API key
        
    Returns:
        GeminiTradingAgent instance
    """
    return GeminiTradingAgent(api_key)
