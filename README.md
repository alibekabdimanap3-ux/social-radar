# 🚨 SocialRadar — Backend API

## Быстрый запуск

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Открой: http://localhost:8000/docs — вся документация там.

---

## Алихан — эндпоинты для фронта

| Метод | URL | Что возвращает |
|---|---|---|
| GET | `/api/districts` | Все 8 районов + риск-скоры + алерты |
| GET | `/api/district/{name}` | Детали района: история, прогноз, explainability, брифинг |
| GET | `/api/trends` | История по месяцам для линейного графика |
| POST | `/api/predict` | Риск-скор по введённым данным |
| POST | `/api/forecast` | Прогноз на 4 месяца |

### Пример запроса:
```js
// Все районы для главного дашборда
const res = await fetch("http://localhost:8000/api/districts")
const data = await res.json()
// data.districts — массив районов
// data.alerts.message — алерт строка
// data.summary.avg_risk — средний риск
```

---

## Танжарық — как заменить модель

В `ml_model.py` класс `SocialRadarML`:
- Метод `train(df)` — замени логику обучения своей
- Метод `predict_risk(features)` — замени предсказание
- Остальное не трогай — API не сломается

---

## Зангар — датасет

`dataset.py` → `generate_dataset()` возвращает DataFrame.
Замени синтетику реальными данными — структура та же:

```
district, district_code, month, social_appeals, credit_delays,
police_calls, unemployment_pct, egov_complaints, domestic_violence, risk_score
```

CSV: `GET /api/dataset` → скачать через API.

---

## Tech Stack
Python · FastAPI · Scikit-learn · Pandas · Uvicorn
