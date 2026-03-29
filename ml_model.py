"""
ml_model.py — ML Pipeline: Isolation Forest + Risk Scoring + Linear Forecast
Фичелер: Танжарықтың feature list-іне сай (socialradar_feature_list.csv)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# Танжарықтың фичелері (lag + rolling + delta + ratio)
FEATURE_COLS = [
    "unemployment_rate", "youth_unemployment_rate",
    "avg_monthly_income_kzt", "below_subsistence_pct",
    "overdue_credit_rate", "avg_debt_kzt", "microfinance_loans",
    "social_appeals_total", "social_poverty", "social_housing", "social_child",
    "police_calls_total", "police_domestic_violence", "police_theft",
    # lag фичелер
    "unemployment_rate_lag1", "overdue_credit_rate_lag1",
    "social_appeals_total_lag1", "police_calls_total_lag1",
    # rolling фичелер
    "unemployment_rate_roll3", "social_appeals_total_roll3", "police_calls_total_roll3",
    # delta фичелер
    "unemployment_rate_delta", "social_appeals_total_delta",
    # ratio фичелер
    "social_per_1000", "police_per_1000", "dv_ratio", "poverty_appeal_ratio",
    # seasonality
    "sin_month", "cos_month",
]

FEATURE_LABELS = {
    "unemployment_rate":        "Уровень безработицы (%)",
    "youth_unemployment_rate":  "Молодёжная безработица (%)",
    "avg_monthly_income_kzt":   "Средний доход (тг)",
    "below_subsistence_pct":    "Ниже прожиточного минимума (%)",
    "overdue_credit_rate":      "Просрочка кредитов (%)",
    "avg_debt_kzt":             "Средний долг (тг)",
    "microfinance_loans":       "Микрокредиты",
    "social_appeals_total":     "Обращения в соцслужбы",
    "social_poverty":           "Обращения — бедность",
    "social_housing":           "Обращения — жильё",
    "social_child":             "Обращения — защита детей",
    "police_calls_total":       "Вызовы полиции",
    "police_domestic_violence": "Домашнее насилие",
    "police_theft":             "Кражи",
    "unemployment_rate_lag1":   "Безработица (лаг 1 мес)",
    "overdue_credit_rate_lag1": "Просрочка (лаг 1 мес)",
    "social_appeals_total_lag1":"Обращения (лаг 1 мес)",
    "police_calls_total_lag1":  "Полиция (лаг 1 мес)",
    "unemployment_rate_roll3":  "Безработица (скол. среднее 3м)",
    "social_appeals_total_roll3":"Обращения (скол. среднее 3м)",
    "police_calls_total_roll3": "Полиция (скол. среднее 3м)",
    "unemployment_rate_delta":  "Изменение безработицы",
    "social_appeals_total_delta":"Изменение обращений",
    "social_per_1000":          "Обращений на 1000 жит.",
    "police_per_1000":          "Вызовов на 1000 жит.",
    "dv_ratio":                 "Доля домашнего насилия",
    "poverty_appeal_ratio":     "Доля обращений по бедности",
    "sin_month":                "Сезонность (sin)",
    "cos_month":                "Сезонность (cos)",
}


def _risk_level(score: float) -> str:
    if score >= 75: return "Критический"
    if score >= 55: return "Высокий"
    if score >= 35: return "Средний"
    return "Низкий"


class SocialRadarML:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        self.risk_model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
        self.forecast_models = {}
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        # Берём только те фичи которые реально есть в датасете
        available = [f for f in FEATURE_COLS if f in df.columns]
        self._feature_cols = available

        X = df[available].fillna(0).values
        y = df["risk_score"].values

        X_scaled = self.scaler.fit_transform(X)
        self.anomaly_model.fit(X_scaled)
        self.risk_model.fit(X_scaled, y)

        # Forecast per district
        for district in df["district"].unique():
            d_df = df[df["district"] == district].sort_values("month")
            if len(d_df) >= 3:
                X_t = np.arange(len(d_df)).reshape(-1, 1)
                y_t = d_df["risk_score"].values
                reg = LinearRegression().fit(X_t, y_t)
                self.forecast_models[district] = {
                    "model": reg, "last_idx": len(d_df) - 1,
                }

        self.is_trained = True
        return {"status": "trained", "samples": len(df), "features": len(available)}

    def predict_risk(self, features: dict) -> dict:
        available = self._feature_cols
        X = np.array([[features.get(f, 0) for f in available]])
        X_scaled = self.scaler.transform(X)

        risk = float(np.clip(self.risk_model.predict(X_scaled)[0], 0, 100))
        is_anomaly = self.anomaly_model.predict(X_scaled)[0] == -1
        anomaly_score = float(self.anomaly_model.decision_function(X_scaled)[0])

        importances = self.risk_model.feature_importances_
        contributions = []
        for i, feat in enumerate(available):
            val_norm = float(X_scaled[0][i])
            contribution = float(importances[i] * val_norm * 100)
            contributions.append({
                "feature":      feat,
                "label":        FEATURE_LABELS.get(feat, feat),
                "value":        features.get(feat, 0),
                "importance":   round(float(importances[i]), 3),
                "contribution": round(contribution, 1),
                "direction":    "negative" if contribution > 8 else "neutral",
            })
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "risk_score":    round(risk, 1),
            "is_anomaly":    bool(is_anomaly),
            "anomaly_score": round(anomaly_score, 3),
            "risk_level":    _risk_level(risk),
            "contributions": contributions,
        }

    def forecast(self, district: str, months_ahead: int = 4) -> list:
        if district not in self.forecast_models:
            return []
        fm = self.forecast_models[district]
        future = ["2024-01","2024-02","2024-03","2024-04"][:months_ahead]
        last_idx = fm["last_idx"]
        result = []
        for i, month in enumerate(future):
            predicted = float(np.clip(fm["model"].predict([[last_idx + i + 1]])[0], 0, 100))
            result.append({"month": month, "predicted_risk": round(predicted, 1), "risk_level": _risk_level(predicted)})
        return result

    def generate_briefing(self, district: str, risk_score: float, contributions: list, forecast: list) -> str:
        level = _risk_level(risk_score)
        top_factors = [c["label"] for c in contributions[:3]]
        trend = ""
        if forecast:
            last = forecast[-1]["predicted_risk"]
            if last > risk_score + 5:   trend = "📈 Тренд ухудшается — без вмешательства риск вырастет."
            elif last < risk_score - 5: trend = "📉 Тренд улучшается, продолжайте мониторинг."
            else:                       trend = "➡️ Ситуация стабильна."

        urgency = {
            "Критический": "⚠️ ТРЕБУЕТСЯ НЕМЕДЛЕННОЕ ВМЕШАТЕЛЬСТВО",
            "Высокий":     "🔶 Рекомендуется приоритетное внимание",
            "Средний":     "🔷 Плановый мониторинг",
            "Низкий":      "✅ Ситуация под контролем",
        }.get(level, "")

        rec = {
            "Критический": "Направить выездную бригаду соцзащиты, усилить патрулирование, созвать экстренное совещание акимата.",
            "Высокий":     "Провести встречи с населением, активировать программы занятости.",
            "Средний":     "Продолжить плановый мониторинг, проверить исполнение программ.",
            "Низкий":      "Поддерживать текущий уровень услуг.",
        }.get(level, "")

        return f"""{urgency}

Район: {district}
Индекс социального риска: {risk_score}/100 ({level})

Ключевые факторы:
• {top_factors[0] if len(top_factors) > 0 else '—'}
• {top_factors[1] if len(top_factors) > 1 else '—'}
• {top_factors[2] if len(top_factors) > 2 else '—'}

{trend}

Рекомендация: {rec}""".strip()


_model_instance = None

def get_model() -> SocialRadarML:
    global _model_instance
    if _model_instance is None:
        from dataset import generate_dataset
        _model_instance = SocialRadarML()
        df = generate_dataset()
        result = _model_instance.train(df)
        print(f"✅ Модель обучена: {result['samples']} строк, {result['features']} фичей")
    return _model_instance
