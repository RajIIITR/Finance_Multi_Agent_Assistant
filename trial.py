import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT="Financial_stock_Eval"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "Financial_stock_Eval"

from typing import Union, Dict, Set, List, TypedDict, Annotated
import pandas as pd
from langchain_core.tools import tool
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm.invoke("Hlo")

"""To extract company name and ticker name"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Company extraction system
class CompanyExtraction(BaseModel):
    """Extract company name and stock symbol from user question."""

    company_name: str = Field(description="The company name mentioned in the question")
    stock_symbol: str = Field(description="The stock ticker symbol (e.g., TSLA, AAPL)")

structured_llm_extractor = llm.with_structured_output(CompanyExtraction)

company_extract_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting company names and stock symbols from user questions.
    Extract the company name and its corresponding stock ticker symbol from the user's question.
    If the user mentions a company name, provide the appropriate stock symbol.
    If the user mentions a stock symbol, provide the company name.
    Common examples: Tesla -> TSLA, Apple -> AAPL, Microsoft -> MSFT, Amazon -> AMZN, Google -> GOOGL"""),
    ("human", "User question: {question}")
])

company_extractor = company_extract_prompt | structured_llm_extractor

"""To check whether given query by user is regarding financial stock or not"""

class GradeQuestion(BaseModel):
    """Binary score for relevance check on Question given by user whether is it valid financial stock related Question or Not."""

    binary_score: str = Field(
        description="Question are relavant to stock, 'yes' or 'no'"
    )

# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeQuestion)
# Prompt
system = """You are a grader assessing relevance of a user question. \n
    If the user question contains keyword(s) or semantic meaning related to the financial stock aspects, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the question is relavant to stock aspects."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

question = "When will wednesday season 2 will release?"
print(retrieval_grader.invoke({"question": question}))

question = "What is current price of Tesla?"
print(retrieval_grader.invoke({"question": question}))

def grade_Question(state):
    """
    Determine whether the user question are relavant to medical domain.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECKING DOCUMENT RELEVANT IS TO QUESTION OR NOT---")

    question = state["question"]

    Question_valid = "Yes"

    score = retrieval_grader.invoke({"question": question})
    grade = score.binary_score
    if grade == "yes":
        print("---GRADE: Question IS RELEVANT---")
        try:
            company_info = company_extractor.invoke({"question": question})
            stock_symbol = company_info.stock_symbol
            company_name = company_info.company_name
            print(f"---EXTRACTED COMPANY: {company_name} ({stock_symbol})---")
        except Exception as e:
            print(f"---ERROR EXTRACTING COMPANY: {e}---")
            stock_symbol = "UNKNOWN"
            company_name = "UNKNOWN"
    else:
        print("---GRADE: Question IS NOT RELEVANT---")
        Question_valid = "No"
        stock_symbol = ""
        company_name = ""


    return {
        "question": question,
        "Question_valid": Question_valid,
        "messages": state.get("messages", []),
        "news": state.get("news", []),
        "stock": stock_symbol,
        "company_name": company_name
    }

def decide_to_generate(state):
    """Route based on question validity"""
    if state.get("Question_valid") == "Yes":
        return "fundamental_analyst"
    else:
        return "reject"

def reject_question(state):
    """Handle rejection of non-financial questions"""
    print("---REJECTING: Not a finance stock question---")
    return {
        "question": state["question"],
        "messages": ["Given Question Doesn't belong to finance stock Domain"],
        "Question_valid": "No",
        "news": state.get("news", []),
        "stock": state.get("stock", ""),
        "company_name": state.get("company_name", ""),
        "generation": "This question is not related to financial stock analysis."
    }

def decide_after_tools(state):
    """Route after tool execution back to fundamental_analyst"""
    return "fundamental_analyst"

def decide_after_fundamental_analyst(state):
    """Route from fundamental_analyst to transform_question"""
    return "transform_question"

"""Let's Create Question Re-Writer so while using tavily search we can find optimal news which can significantly effect the stock pricing. (Will focus on news till last month)"""

### Question Re-writer
# Prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

system = """You a question re-writer that converts an input question to a better version that is optimized \n
     for web search. Give only 1 optimized question. Look at the input and try to reason about the underlying semantic intent / meaning.
     Your main role is define a better question for extracting past one month news about the company due to which it may be a factor of rise or fall of the stock."""


re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

question = "How does stock price of tesla was affected after US election?"
question_rewriter.invoke({"question": question})

def transform_question(state):
    """
    Transform the query to produce a better question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrase question
    """
    print("---TRANSFORMER QUERY---")

    question = state["question"]
    messages = state["messages"]
    company_name = state.get("company_name", "")

    # Include company name in the question for better search and better transform user query for optimize web search
    enhanced_question = f"{question} {company_name}" if company_name else question
    better_question = question_rewriter.invoke({"question": enhanced_question})

    if isinstance(better_question, dict):
        better_question = better_question.get("question", "")

    print("---NEW QUESTION---")
    print(better_question)

    return {
        "messages": messages,
        "question": better_question,
        "Question_valid": state.get("Question_valid", ""),
        "news": state.get("news", []),
        "stock": state.get("stock", ""),
        "company_name": state.get("company_name", "")
    }

"""Let's Define our Tavily Search to get recent news about particular company stocks etc...."""

### Search

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=5)

def web_search(state):
    """Web search based on the re-phrased question."""
    print("---WEB SEARCH---")

    question = state["question"]
    news = state.get("news", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    news.append(web_results)

    return {
        "news": news,
        "question": question,
        "messages": state.get("messages", []),
        "Question_valid": state.get("Question_valid", ""),
        "stock": state.get("stock", ""),
        "company_name": state.get("company_name", "")
    }

"""<b>Fetching Stock Prices:</b> This tool fetches the stock’s historical data and computes several technical indicators."""

@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches historical stock price data and technical indicator for a given ticker."""
    try:
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=24*3),
            end=dt.datetime.now(),
            interval='1wk'
        )
        df = data.copy()
        data.reset_index(inplace=True)
        data.Date = data.Date.astype(str)

        indicators = {}

        rsi_series = RSIIndicator(df['Close'], window=14).rsi().iloc[-12:]
        indicators["RSI"] = {date.strftime('%Y-%m-%d'): int(value)
                    for date, value in rsi_series.dropna().to_dict().items()}

        sto_series = StochasticOscillator(
            df['High'], df['Low'], df['Close'], window=14).stoch().iloc[-12:]
        indicators["Stochastic_Oscillator"] = {
                    date.strftime('%Y-%m-%d'): int(value)
                    for date, value in sto_series.dropna().to_dict().items()}

        macd = MACD(df['Close'])
        macd_series = macd.macd().iloc[-12:]
        indicators["MACD"] = {date.strftime('%Y-%m-%d'): int(value)
                    for date, value in macd_series.to_dict().items()}

        macd_signal_series = macd.macd_signal().iloc[-12:]
        indicators["MACD_Signal"] = {date.strftime('%Y-%m-%d'): int(value)
                    for date, value in macd_signal_series.to_dict().items()}

        vwap_series = volume_weighted_average_price(
            high=df['High'], low=df['Low'], close=df['Close'],
            volume=df['Volume'],
        ).iloc[-12:]
        indicators["vwap"] = {date.strftime('%Y-%m-%d'): int(value)
                    for date, value in vwap_series.to_dict().items()}

        return {'stock_price': data.to_dict(orient='records'),
                'indicators': indicators}

    except Exception as e:
        return f"Error fetching price data: {str(e)}"

"""<b>Financial Ratios:</b> This tool retrieves key financial health ratios."""

@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Fetches key financial ratios for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'PE_Ratio': info.get('forwardPE'),
            'Price_to_Book': info.get('priceToBook'),
            'Debt_to_Equity': info.get('debtToEquity'),
            'Profit_Margins': info.get('profitMargins'),
            # 'Operating_Margin': info.get('operatingMargins'),
            # 'Gross_Margin': info.get('grossMargins'),
            # 'Return_on_Equity (ROE)': info.get('returnOnEquity'),
            # 'Return_on_Assets (ROA)': info.get('returnOnAssets'),
            # 'Current_Ratio': info.get('currentRatio'),
            # 'Quick_Ratio': info.get('quickRatio'),
            # 'Operating_Cash_Flow': info.get('operatingCashflow'),
            # 'Free_Cash_Flow': info.get('freeCashflow'),
            # 'Revenue_Growth': info.get('revenueGrowth'),
            # 'Earnings_Growth': info.get('earningsGrowth')
        }
    except Exception as e:
        return f"Error fetching ratios: {str(e)}"

"""Let's Define our State and StateGraph"""

tools = [get_stock_prices, get_financial_metrics]
llm_with_tool = llm.bind_tools(tools)

def generate_final_analysis(state):
    """Generate final analysis combining all data"""
    print("---GENERATE FINAL ANALYSIS---")

    messages = state.get("messages", [])
    news = state.get("news", [])
    stock_symbol = state.get("stock", "")
    company_name = state.get("company_name", "")

    # Create a comprehensive prompt for final analysis
    final_prompt = f"""Based on all the gathered information, provide a comprehensive analysis for {company_name} ({stock_symbol}):

    Tools Data Available: {len([m for m in messages if hasattr(m, 'tool_calls') and m.tool_calls])} tool responses
    News Data Available: {len(news)} news sources

    Please provide a structured analysis with:
    1. Stock Price Analysis
    2. Technical Indicators Analysis
    3. Financial Metrics Analysis
    4. News Impact Analysis
    5. Final Summary and Recommendation
    6. Answer to the original question: {state['question']}

    Use all the data gathered from tools and news to provide actionable insights."""

    # Get the latest messages and create analysis
    analysis_messages = [HumanMessage(content=final_prompt)] + messages

    # Generate final analysis
    final_response = llm.invoke(analysis_messages)

    return {
        "question": state["question"],
        "news": news,
        "messages": messages + [final_response],
        "generation": final_response.content,
        "Question_valid": state.get("Question_valid", ""),
        "stock": stock_symbol,
        "company_name": company_name
    }

class State(TypedDict):
    messages: List
    question: str
    Question_valid: str
    news: List
    stock: str
    company_name: str
    generation: str

FUNDAMENTAL_ANALYST_PROMPT = """You are a fundamental analyst specializing in evaluating company performance based on stock prices, technical indicators, financial metrics, and recent news. Your task is to provide a comprehensive summary of the fundamental analysis for {company} (Symbol: {stock_symbol}).

You have access to the following tools:
1. **get_stock_prices**: Retrieves the latest stock price, historical price data and technical Indicators like RSI, MACD, and VWAP.
2. **get_financial_metrics**: Retrieves key financial metrics, such as revenue, earnings per share (EPS), price-to-earnings ratio (P/E), and debt-to-equity ratio.

You also have access to recent news and market information that has been gathered through web search.

### Your Task:
1. **Use Tools**: Call the appropriate tools to gather stock price data and financial metrics for {stock_symbol}
2. **Analyze Data**: Evaluate the results and identify trends, strengths, or concerns
3. **Incorporate News**: Consider the recent news in your analysis
4. **Provide Summary**: Write a comprehensive analysis

Give Final Answer on the basis of given user question.

Please start by calling the tools to gather the necessary data for {stock_symbol}."""


def fundamental_analyst(state: State):
    """Fundamental analyst node that calls tools and provides analysis"""
    print("---FUNDAMENTAL ANALYST---")

    stock_symbol = state.get('stock', 'UNKNOWN')
    company_name = state.get('company_name', 'UNKNOWN')

    if stock_symbol == 'UNKNOWN':
        error_msg = "No stock symbol identified. Cannot perform analysis."
        return {
            'messages': state['messages'] + [AIMessage(content=error_msg)],
            'question': state['question'],
            'Question_valid': state['Question_valid'],
            'news': state['news'],
            'stock': state['stock'],
            'company_name': state['company_name']
        }

    # Create system message with the prompt
    system_message = SystemMessage(
        content=FUNDAMENTAL_ANALYST_PROMPT.format(
            company=company_name,
            stock_symbol=stock_symbol
        )
    )

    # Add user question as human message
    human_message = HumanMessage(content=state['question'])

    messages = [system_message, human_message] + state['messages']

    # Invoke LLM with tools - this should trigger tool calls
    response = llm_with_tool.invoke(messages)

    return {
        'messages': state['messages'] + [response],
        'question': state['question'],
        'Question_valid': state['Question_valid'],
        'news': state['news'],
        'stock': state['stock'],
        'company_name': state['company_name']
    }

def tools_condition(state):
    """
    Always routes to tools for stock data gathering
    """
    return "tools"

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

graph_builder = StateGraph(State)

def decide_to_generate(state):
    """Route based on question validity"""
    if state.get("Question_valid") == "Yes":
        return "fundamental_analyst"
    else:
        return "reject"

graph_builder.add_node('grade_question', grade_Question)
graph_builder.add_node('fundamental_analyst', fundamental_analyst)
graph_builder.add_node('reject', reject_question)
graph_builder.add_node('tools', ToolNode(tools))
graph_builder.add_node('transform_question', transform_question)
graph_builder.add_node('web_search', web_search)
graph_builder.add_node('generate', generate_final_analysis)

graph_builder.add_edge(START, 'grade_question')
graph_builder.add_conditional_edges('grade_question', decide_to_generate,
                                    {"fundamental_analyst": "fundamental_analyst", "reject": "reject"})

# Main workflow: grade_question -> fundamental_analyst -> tools -> transform_question -> web_search -> generate
graph_builder.add_edge('fundamental_analyst', 'tools')
graph_builder.add_edge('tools', 'transform_question')
graph_builder.add_edge('transform_question', 'web_search')
graph_builder.add_edge('web_search', 'generate')

# Rejection path
graph_builder.add_edge('reject', END)
graph_builder.add_edge('generate', END)

# Compile the graph
graph = graph_builder.compile()

# Display the graph
from IPython.display import Image, display # type: ignore
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

from pprint import pprint

# Test inputs
inputs = {
    "question": "What is current price of Tesla?",
    "messages": [],
    "news": [],
    "Question_valid": "",
    "stock": "",
    "company_name": "",
    "generation": ""
}

print("Starting Financial Analysis...")

final_result = None
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"Node '{key}' completed")
        final_result = value
    print("---")

if final_result and "generation" in final_result:
    print("\n=== FINAL ANALYSIS ===")
    print(final_result["generation"])
else:
    print("\n=== ERROR ===")
    print("No final generation produced")

from pprint import pprint

# Test inputs
inputs = {
    "question": "Who is the CEO of Tesla?",
    "messages": [],
    "news": [],
    "Question_valid": "",
    "stock": "",
    "company_name": "",
    "generation": ""
}

print("Starting Financial Analysis...")

final_result = None
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"Node '{key}' completed")
        final_result = value
    print("---")

if final_result and "generation" in final_result:
    print("\n=== FINAL ANALYSIS ===")
    print(final_result["generation"])
else:
    print("\n=== ERROR ===")
    print("No final generation produced")

from pprint import pprint

# Test inputs
inputs = {
    "question": "what is stock price of Amazon?",
    "messages": [],
    "news": [],
    "Question_valid": "",
    "stock": "",
    "company_name": "",
    "generation": ""
}

print("Starting Financial Analysis...")

final_result = None
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"Node '{key}' completed")
        final_result = value
    print("---")

if final_result and "generation" in final_result:
    print("\n=== FINAL ANALYSIS ===")
    print(final_result["generation"])
else:
    print("\n=== ERROR ===")
    print("No final generation produced")

from pprint import pprint

# Test inputs
inputs = {
    "question": "Shall I buy dollar stocks or wait?",
    "messages": [],
    "news": [],
    "Question_valid": "",
    "stock": "",
    "company_name": "",
    "generation": ""
}

print("Starting Financial Analysis...")

final_result = None
for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"Node '{key}' completed")
        final_result = value
    print("---")

if final_result and "generation" in final_result:
    print("\n=== FINAL ANALYSIS ===")
    print(final_result["generation"])
else:
    print("\n=== ERROR ===")
    print("No final generation produced")