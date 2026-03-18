# MMM Copilot — AI-Powered Marketing Mix Modeling

Upload marketing spend data, chat with an AI agent, and get ROAS analysis, budget optimization, and scenario forecasting — all with interactive charts.

## Architecture

- **Backend**: FastAPI + Pydantic AI (Gemini) + Ridge Regression MMM engine
- **Frontend**: React + TypeScript + Vite + Recharts
- **Agent**: Pydantic AI with tool-calling (fit model, get ROAS, optimize budget, forecast)

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
GOOGLE_API_KEY=your-key uvicorn main:app --reload --port 8000
```

> Without `GOOGLE_API_KEY`, the agent uses a keyword-matching fallback.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Usage

1. Upload `sample_data.csv` (or your own CSV with date, channel, spend, revenue columns)
2. Chat with the AI: "Analyze my marketing spend"
3. Get ROAS charts, contribution breakdowns, efficiency scatter plots
4. Optimize: "Optimize my budget for $100K/month"
5. Forecast: "What if I double TikTok spend?"

## CSV Format

| Column    | Description                        |
|-----------|------------------------------------|
| date      | Date (YYYY-MM-DD)                  |
| channel   | Marketing channel name             |
| spend     | Daily spend amount                 |
| revenue   | Daily revenue attributed           |

## Tech Stack

| Layer     | Technology                         |
|-----------|------------------------------------|
| Backend   | FastAPI, Pydantic AI, scikit-learn |
| Frontend  | React 18, TypeScript, Vite 6       |
| Charts    | Recharts 2                         |
| Icons     | Lucide React                       |
| AI Model  | Google Gemini 2.0 Flash            |

## License

MIT
