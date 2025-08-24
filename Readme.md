# Stock Analysis Tool

A comprehensive AI-powered stock analysis tool that provides fundamental analysis, technical indicators, and investment recommendations using LangGraph and Google's Gemini AI.

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![AI](https://img.shields.io/badge/AI-Powered-brightgreen?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-3C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Flow%20Control-orange?style=for-the-badge)


## Features

- **Comprehensive Stock Analysis**: Get detailed fundamental analysis for any stock
- **Technical Indicators**: RSI, MACD, Stochastic Oscillator, VWAP analysis
- **Financial Metrics**: P/E ratio, Price-to-Book, Debt-to-Equity, Profit Margins
- **AI-Powered Insights**: Leveraging Google's Gemini AI for intelligent analysis
- **Web Interface**: Clean, responsive web interface
- **Real-time Data**: Live stock data from Yahoo Finance
- **Fast API**: Built with FastAPI for high performance

##  Tech Stack

- **Backend**: FastAPI, Python
- **AI/ML**: LangChain, LangGraph, Google Gemini AI
- **Data**: Yahoo Finance API, Technical Analysis Library
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Uvicorn ASGI server

### Run the Application on Local
using uvicorn:
```bash
uvicorn app:app --reload 
```

### Web Interface
1. Navigate to `http://localhost:8000`
2. Enter a stock symbol (e.g., `AAPL`, `TSLA`, `TATASTEEL.NS`)
3. Ask your question about the stock
4. Click "Analyze Stock" to get comprehensive analysis



## Supported Stock Symbols

- **US Stocks**: AAPL, GOOGL, MSFT, TSLA, AMZN, etc.
- **Indian Stocks**: TATASTEEL.NS, RELIANCE.NS, INFY.NS, etc.

## Project Structure

```
stock-analysis-tool/
├── research
│   ├── trial1.ipynb       # Prototyping part 1
│   ├── trial.ipynb        # Prototyping part 2 (Advancing)
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── README.md             # This file
├── src/
│   ├── __init__.py       # Package initialization
│   ├── helper.py         # Helper functions and tools
│   └── prompt.py         # AI prompts and templates
├── templates/
│   ├── index.html        # Main page template
│   ├── result.html       # Analysis results page
│   └── error.html        # Error page template
└── static/
    ├── style.css         # Minimal CSS styles
    └── script.js         # JavaScript functionality
```

## Key Components

### Stock Analysis Features
- **Price Analysis**: Historical price trends and movements
- **Technical Indicators**: 
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Stochastic Oscillator
  - VWAP (Volume Weighted Average Price)
- **Financial Metrics**:
  - P/E Ratio
  - Price-to-Book Ratio
  - Debt-to-Equity Ratio
  - Profit Margins


## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for stock data
- [Google Gemini AI](https://ai.google.dev/) for AI-powered analysis
- [LangChain](https://python.langchain.com/) for AI framework
- [LangGraph](https://www.langchain.com/langgraph) for AI framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Technical Analysis Library](https://github.com/bukosabino/ta) for indicators
- [Got idea of various key metrics from this blog and implementation](https://abhinavk910.medium.com/building-an-agentic-financial-analyst-with-langgraph-and-openai-5138192c9783)


## Future Enhancements (WIll add soon)

- [ ] Stock comparison tool
- [ ] Adding advanced evaluation metrics (https://www.netsuite.com/portal/resource/articles/accounting/financial-kpis-metrics.shtml)
- [ ] Social sentiment analysis (Incorporating current affairs)
- [ ] Machine learning predictions

---

⭐ **Star this repository if you find it helpful!** ⭐
