# Flight Fare Prediction – Streamlit App
Developed by Sindhu S

## Screenshots
| Home | Quick Prediction |
|---|---|
| ![Home](screenshots/home.png) | ![Quick Prediction](screenshots/quick_prediction.png) |

| Smart Analytics | Plan Better |
|---|---|
| ![Smart Analytics](screenshots/smart_analytics.png) | ![Plan Better](screenshots/plan_better.png) |

![About](screenshots/about.png)

## Sample report
Download the PDF the app generates for a prediction: [sample_report/flight_fare_prediction_report.pdf](sample_report/flight_fare_prediction_report.pdf)

## Sample report
The app's **Download Report** button creates a PDF like this one:
[sample_report/flight_fare_prediction_report.pdf](sample_report/flight_fare_prediction_report.pdf)

## Folder structure
```
flight_app/
├── app.py                    # Streamlit UI (5 pages)
├── flight_utils.py           # cleaning, features, model + prediction (same steps as the notebook)
├── requirements.txt
├── .streamlit/config.toml    # purple theme
├── sample_report/            # example PDF report (downloadable)
├── sample_report/            # example PDF created by the Download Report button
├── screenshots/              # images used in this README
├── assets/                   # banner photos (home_sky.webp, predict_sky.webp) + plan_hero.jpg (+ older fallbacks)
├── Flight_Fare.xlsx          # <-- copy your dataset here (required)
├── flight_price_model.pkl    # <-- optional: saved by your notebook
└── feature_columns.pkl       # <-- optional: saved by your notebook
```

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Notes
* If the two `.pkl` files are present, the app loads them. If not (or if they were saved with a very
  different scikit-learn version), it trains the same Tuned Random Forest
  (`n_estimators=100, max_depth=20`) once at start-up (~5-10 s).
* If `Flight_Fare.xlsx` is missing, the app shows an upload box.
* The "Download Report" button creates a PDF of the latest prediction (needs `reportlab`).
* Put your GitHub link in `GITHUB_URL` at the top of `app.py` to show the GitHub icon.
* The model only knows journeys from March-June 2019 and the 6 routes in the dataset, so the
  app only offers those routes/airlines and warns when the chosen month is outside that range.
