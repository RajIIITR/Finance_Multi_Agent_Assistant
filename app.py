from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from src.helper import analyze_stock
import logging

app = FastAPI(title="Stock Analysis Tool", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAnalysisRequest(BaseModel):
    ticker: str
    question: str = "Should I buy this stock?"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, ticker: str = Form(...), question: str = Form(...)):
    """Analyze a stock and return results."""
    try:
        logger.info(f"Analyzing stock: {ticker}")
        result = analyze_stock(ticker, question)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "ticker": ticker,
            "question": question,
            "result": result
        })
    except Exception as e:
        logger.error(f"Error analyzing stock {ticker}: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.post("/api/analyze")
async def api_analyze(request: StockAnalysisRequest):
    """API endpoint for stock analysis."""
    try:
        result = analyze_stock(request.ticker, request.question)
        return {
            "ticker": request.ticker,
            "question": request.question,
            "analysis": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "ticker": request.ticker,
            "question": request.question,
            "error": str(e),
            "status": "error"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)