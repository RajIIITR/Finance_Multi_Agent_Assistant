import os
from dotenv import load_dotenv
from typing import Union, Dict, Set, List, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import yfinance as yf
import datetime as dt
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price
import traceback
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from .prompt import FUNDAMENTAL_ANALYST_PROMPT

# Load environment variables
load_dotenv()

# Environment setup
def setup_environment():
    """Setup environment variables for API keys and tracing."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGSMITH_PROJECT = "Financial_stock_Eval"

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

class State(TypedDict):
    """State definition for the graph."""
    messages: Annotated[list, add_messages]
    stock: str

@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches historical stock price data and technical indicator for a given ticker."""
    try:
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=24*3),
            end=dt.datetime.now(),
            interval='1d'
        )
        df = data.copy()
        if len(df.columns[0]) > 1:
            df.columns = [i[0] for i in df.columns]
        data.reset_index(inplace=True)
        data.Date = data.Date.astype(str)

        indicators = {}

        # Momentum Indicators
        rsi_series = RSIIndicator(df['Close'], window=14).rsi().iloc[-12:]
        indicators["RSI"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in rsi_series.dropna().to_dict().items()}
        
        sto_series = StochasticOscillator(
            df['High'], df['Low'], df['Close'], window=14).stoch().iloc[-12:]
        indicators["Stochastic_Oscillator"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in sto_series.dropna().to_dict().items()}

        macd = MACD(df['Close'])
        macd_series = macd.macd().iloc[-12:]
        indicators["MACD"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in macd_series.to_dict().items()}
        
        macd_signal_series = macd.macd_signal().iloc[-12:]
        indicators["MACD_Signal"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in macd_signal_series.to_dict().items()}

        vwap_series = volume_weighted_average_price(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
        ).iloc[-12:]
        indicators["vwap"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in vwap_series.to_dict().items()}

        return {'stock_price': data.to_dict(orient='records'), 'indicators': indicators}
    except Exception as e:
        return f"Error fetching price data: {str(e)}"

@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Fetches key financial ratios for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'pe_ratio': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'profit_margins': info.get('profitMargins')
        }
    except Exception as e:
        return f"Error fetching ratios: {str(e)}"

def create_llm():
    """Create and return configured LLM instance."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def fundamental_analyst(state: State):
    """Fundamental analyst node function."""
    llm = create_llm()
    tools = [get_stock_prices, get_financial_metrics]
    llm_with_tool = llm.bind_tools(tools)
    
    messages = [
        SystemMessage(content=FUNDAMENTAL_ANALYST_PROMPT.format(company=state['stock'])),
    ] + state['messages']
    
    return {
        'messages': llm_with_tool.invoke(messages)
    }

def create_analysis_graph():
    """Create and return the analysis graph."""
    graph_builder = StateGraph(State)
    tools = [get_stock_prices, get_financial_metrics]
    
    graph_builder.add_node('fundamental_analyst', fundamental_analyst)
    graph_builder.add_edge(START, 'fundamental_analyst')
    graph_builder.add_node(ToolNode(tools))
    graph_builder.add_conditional_edges('fundamental_analyst', tools_condition)
    graph_builder.add_edge('tools', 'fundamental_analyst')
    
    return graph_builder.compile()

def analyze_stock(ticker: str, question: str = "Should I buy this stock?") -> str:
    """Analyze a stock and return the analysis result."""
    try:
        setup_environment()
        graph = create_analysis_graph()
        
        events = graph.stream({
            'messages': [('user', question)],
            'stock': ticker
        }, stream_mode='values')
        
        result = ""
        for event in events:
            if 'messages' in event:
                last_message = event['messages'][-1]
                if hasattr(last_message, 'content'):
                    result = last_message.content
                    
        return result
    except Exception as e:
        return f"Error analyzing stock: {str(e)}"