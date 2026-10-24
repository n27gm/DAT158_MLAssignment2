# Bruktbilpris
### Gruppemedlemer
- Naghm Alqaragei
- Emma Wergedahl

### Prosjekt
Nettside der man fyller inn info om en bruktbil og får et prisestimat fra en LightGBM-modell. 
Modellen er trent på dataene fra Kaggle konkurransen [Playground Series S4E9](https://www.kaggle.com/competitions/playground-series-s4e9).

### Rapport
//må fylles



#### Mapper
- `ml/` : nedlasting av data, dataanalyse (`eda.py`) og trening (`train.py`). Figurene ligger i `ml/figures/`.
- `backend/` : FastAPI server, feature engineering (`car_features.py`) og den trente modellen i `model/`.
- `frontend/` : React-nettsiden.



### Trene modellen selv
Må bruke Python 3.10 eller nyere.

```bash
pip install -r ml/requirements.txt
python ml/download_data.py   # krever Kaggle API nøkkel og at du har trykket "Join Competition"
python ml/eda.py
python ml/train.py           # lagrer modellen i backend/model/
```

### Kjøre nettsiden lokalt

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --port 8001
```
Frontend (i et annet terminalvindu, krever Node og yarn):
```bash
cd frontend
cp .env.example .env
yarn install
yarn start
```

Siden åpnes på http://localhost:3000.



### Resultater
Fra valideringssettet (20 % av dataene):
| Modell | RMSE | MAE | R² |
|---|---:|---:|---:|
| Gjennomsnitt | 74 573 | 29 087 | 0,000 |
| Ridge (log-pris) | 70 491 | 18 419 | 0,106 |
| LightGBM (log-pris) | 68 557 | 17 055 | 0,155 |
| LightGBM (pris) | 68 092 | 19 590 | 0,166 |

Vi bruker den siste på nettsiden. Mer om dette i rapporten.
