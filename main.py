"""
main.py
FastAPI бэкенд для SocialRadar.
Алихан — все эндпоинты задокументированы ниже.
Запуск: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np

from dataset import generate_dataset, get_latest_snapshot, DISTRICTS, DISTRICT_CODES
from ml_model import get_model, FEATURE_COLS, FEATURE_LABELS, _risk_level

app = FastAPI(
    title="SocialRadar API",
    description="AI-система раннего выявления социальных очагов — Алматы",
    version="1.0.0",
)

# ── CORS — Алихан, React сюда подключается ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # в проде заменить на конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    district: str
    social_appeals: float
    credit_delays: float
    police_calls: float
    unemployment_pct: float
    egov_complaints: float
    domestic_violence: float


class ForecastRequest(BaseModel):
    district: str
    months_ahead: Optional[int] = 4


# ── Startup: обучаем модель ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("🚀 SocialRadar: обучаем ML модель...")
    get_model()
    print("✅ Модель готова")


# ══════════════════════════════════════════════════════════════════════════════
# ЭНДПОИНТЫ
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "project": "SocialRadar",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


# ── 1. Все районы — текущий снапшот ──────────────────────────────────────────
# Алихан: GET /api/districts → список районов с риск-скорами для карты и бар-чарта
@app.get("/api/districts")
def get_districts():
    """
    Возвращает все 8 районов с текущим риск-скором.
    Используй для: карта Leaflet, бар-чарт, список алертов.
    """
    model = get_model()
    df = get_latest_snapshot()

    results = []
    for _, row in df.iterrows():
        features = {f: row[f] for f in FEATURE_COLS}
        prediction = model.predict_risk(features)

        results.append({
            "district":      row["district"],
            "code":          row["district_code"],
            "risk_score":    prediction["risk_score"],
            "risk_level":    prediction["risk_level"],
            "is_anomaly":    prediction["is_anomaly"],
            "top_factor":    prediction["contributions"][0]["label"] if prediction["contributions"] else "",
            # Координаты для Leaflet (центры районов Алматы)
            "lat":           _district_coords(row["district"])[0],
            "lng":           _district_coords(row["district"])[1],
        })

    # Сортировка по риску
    results.sort(key=lambda x: x["risk_score"], reverse=True)

    # Алерты
    critical = [r for r in results if r["risk_level"] == "Критический"]
    high =     [r for r in results if r["risk_level"] == "Высокий"]

    return {
        "districts": results,
        "alerts": {
            "critical_count": len(critical),
            "high_count":     len(high),
            "critical":       [r["district"] for r in critical],
            "message":        f"🚨 {len(critical)} критических алертов: {', '.join([r['district'] for r in critical])} — немедленные меры" if critical else "✅ Критических алертов нет",
        },
        "summary": {
            "avg_risk":       round(np.mean([r["risk_score"] for r in results]), 1),
            "max_risk":       max(r["risk_score"] for r in results),
            "anomaly_count":  sum(1 for r in results if r["is_anomaly"]),
        }
    }


# ── 2. Детальный анализ одного района ────────────────────────────────────────
# Алихан: GET /api/district/{name} → детальная карточка района
@app.get("/api/district/{district_name}")
def get_district_detail(district_name: str):
    """
    Детальный анализ района: риск, explainability, история, прогноз, брифинг.
    """
    model = get_model()
    df = generate_dataset()

    district_df = df[df["district"] == district_name]
    if district_df.empty:
        raise HTTPException(status_code=404, detail=f"Район '{district_name}' не найден")

    # Последний месяц
    latest = district_df[district_df["month"] == "2024-12"].iloc[0]
    features = {f: latest[f] for f in FEATURE_COLS}
    prediction = model.predict_risk(features)

    # История (все месяцы)
    history = []
    for _, row in district_df.sort_values("month").iterrows():
        history.append({
            "month":      row["month"],
            "risk_score": int(row["risk_score"]),
        })

    # Прогноз
    forecast = model.forecast(district_name, months_ahead=4)

    # AI брифинг
    briefing = model.generate_briefing(
        district_name,
        prediction["risk_score"],
        prediction["contributions"],
        forecast,
    )

    return {
        "district":      district_name,
        "current": {
            "month":      "2024-12",
            "risk_score": prediction["risk_score"],
            "risk_level": prediction["risk_level"],
            "is_anomaly": prediction["is_anomaly"],
        },
        "features":      {FEATURE_LABELS[f]: float(features[f]) for f in FEATURE_COLS},
        "explainability": prediction["contributions"],
        "history":        history,
        "forecast":       forecast,
        "briefing":       briefing,
    }


# ── 3. ML предсказание (для Танжарыка — можно заменить модель) ───────────────
# Алихан: POST /api/predict → получить риск-скор по произвольным данным
@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Получить риск-скор по введённым данным.
    Используй для: ручной ввод данных чиновником.
    """
    model = get_model()
    features = {
        "social_appeals":    req.social_appeals,
        "credit_delays":     req.credit_delays,
        "police_calls":      req.police_calls,
        "unemployment_pct":  req.unemployment_pct,
        "egov_complaints":   req.egov_complaints,
        "domestic_violence": req.domestic_violence,
    }
    prediction = model.predict_risk(features)
    forecast = model.forecast(req.district, months_ahead=4)
    briefing = model.generate_briefing(
        req.district, prediction["risk_score"],
        prediction["contributions"], forecast
    )

    return {
        "district":       req.district,
        "risk_score":     prediction["risk_score"],
        "risk_level":     prediction["risk_level"],
        "is_anomaly":     prediction["is_anomaly"],
        "explainability": prediction["contributions"],
        "forecast":       forecast,
        "briefing":       briefing,
    }


# ── 4. Прогноз по району ──────────────────────────────────────────────────────
@app.post("/api/forecast")
def forecast_district(req: ForecastRequest):
    """Прогноз риска на 1–4 месяца вперёд."""
    model = get_model()
    forecast = model.forecast(req.district, req.months_ahead)
    if not forecast:
        raise HTTPException(status_code=404, detail="Район не найден")
    return {"district": req.district, "forecast": forecast}


# ── 5. Исторические тренды (для графика) ─────────────────────────────────────
@app.get("/api/trends")
def get_trends():
    """
    История всех районов по месяцам.
    Алихан: используй для линейного графика трендов.
    """
    df = generate_dataset()
    result = {}
    for district in DISTRICTS:
        d_df = df[df["district"] == district].sort_values("month")
        result[district] = [
            {"month": row["month"], "risk_score": int(row["risk_score"])}
            for _, row in d_df.iterrows()
        ]
    return {"trends": result, "districts": DISTRICTS}


# ── 6. Датасет (для Зангара — скачать CSV) ───────────────────────────────────
@app.get("/api/dataset")
def get_dataset():
    """Полный датасет в JSON формате."""
    df = generate_dataset()
    return {"data": df.to_dict(orient="records"), "total": len(df)}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _district_coords(district: str) -> tuple[float, float]:
    """Приблизительные координаты центров районов Алматы."""
    coords = {
        "Наурызбайский": (43.198, 76.774),
        "Турксибский":   (43.312, 77.051),
        "Арычный":       (43.275, 76.895),
        "Алатауский":    (43.251, 76.820),
        "Жетысуский":    (43.290, 77.010),
        "Алмалинский":   (43.257, 76.932),
        "Бостандыкский": (43.220, 76.870),
        "Медеуский":     (43.195, 76.960),
    }
    return coords.get(district, (43.238, 76.889))
