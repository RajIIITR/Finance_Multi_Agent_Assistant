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

## Installation (FOr your and mine reference in future)

### Prerequisites
- Python 3.10 
- pip package manager
- Google API key for Gemini AI

### 1. Clone the Repository
```bash
git clone https://github.com/RajIIITR/Finance_Multi_Agent_Assistant.git
cd Finance_Multi_Agent_Assistant
```

### 2. Create Virtual Environment
```bash
conda create -p finance_assist python=3.10 -y
conda activate finance_assist/ 
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional
```

### 5. Run the Application
```bash
python app.py
```

Or using uvicorn:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Usage

### Web Interface
1. Navigate to `http://localhost:8000`
2. Enter a stock symbol (e.g., `AAPL`, `TSLA`, `TATASTEEL.NS`)
3. Ask your question about the stock
4. Click "Analyze Stock" to get comprehensive analysis

### API Endpoints
- **GET** `/` - Main web interface
- **POST** `/analyze` - Web form analysis
- **POST** `/api/analyze` - JSON API endpoint
- **GET** `/health` - Health check
- **GET** `/docs` - Interactive API documentation

### API Usage Example
```python
import requests

response = requests.post("http://localhost:8000/api/analyze", json={
    "ticker": "AAPL",
    "question": "Should I buy this stock?"
})

print(response.json())
```

## Supported Stock Symbols

- **US Stocks**: AAPL, GOOGL, MSFT, TSLA, AMZN, etc.
- **Indian Stocks**: TATASTEEL.NS, RELIANCE.NS, INFY.NS, etc.

## 🏗️ Project Structure

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

### AI-Powered Analysis
- Uses Google's Gemini AI for intelligent insights
- LangGraph for structured analysis workflow
- Comprehensive fundamental analysis reports

## 🔧 Configuration

### Environment Variables
```env
# Required
GOOGLE_API_KEY=your_google_api_key

# Optional (for enhanced features)
TAVILY_API_KEY=your_tavily_api_key  (Will add in my future use case)
LANGCHAIN_API_KEY=your_langchain_api_key
```

## Deployment

### Local Development
```bash
uvicorn app:app --reload
```


### Docker 
```dockerfile
FROM python:3.10-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
   

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for stock data
- [Google Gemini AI](https://ai.google.dev/) for AI-powered analysis
- [LangChain](https://python.langchain.com/) for AI framework
- [LangGraph](https://www.langchain.com/langgraph) for AI framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Technical Analysis Library](https://github.com/bukosabino/ta) for indicators
- [Got idea of various key metrics from this blog and immplementation](https://abhinavk910.medium.com/building-an-agentic-financial-analyst-with-langgraph-and-openai-5138192c9783) referred this blog

## Known Issues

- API rate limits may apply for high-frequency requests
- Can't be used for future stock prediction currently.

## Future Enhancements

- [ ] Stock comparison tool
- [ ] Adding advanced evaluation metrics (https://www.netsuite.com/portal/resource/articles/accounting/financial-kpis-metrics.shtml)
- [ ] Social sentiment analysis (Incorporating current affairs)
- [ ] Machine learning predictions

---

⭐ **Star this repository if you find it helpful!** ⭐
