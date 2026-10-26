# Rapport: Hva er bruktbilen verdt?

## 1. Problemstilling

En nettside som kan gi et anslag på hva en brukt bil er verdt. 
Når man skal kjøpe eller selge bil er det ofte vanskelig å vite om prisen er rimelig. Prisen avhenger av: merke, modell, hvor gammel bilen er, hvor langt den har kjørt, 
motor, om den har vært i en ulykke og så videre. 
Tanken er at man skal kunne fylle inn det man vet om bilen og få et raskt estimat.

Datasettet fra Kaggle konkurransen *Regression of Used Car Prices* (Playground Series S4E9) ble brukt her.
Det er et syntetisk datasett som er laget ut fra et ekte datasett med amerikanske bilannonser. 
Kjørelengden er i miles, og siden modellen gjelder det amerikanske markedet og ikke det norske så er prisene i dollar. 


**Å måle om modellen er god:** 
Konkurransen bruker RMSE, det var hovedmetrikken, også MAE fordi den er lettere å forstå ("i snitt bommer modellen med X dollar") og R². 
Målet var å slå en enkel baseline som bare gjetter gjennomsnittsprisen og at nettsiden skulle svare raskt.

Alt er kjørt på vanlige bærbare PC-er uten GPU.
Brukt: 
- Python (pandas, scikit-learn, LightGBM) til modellen
- FastAPI + React til nettsiden.

## 2. Data

Treningsfilen har 188 533 biler og disse kolonnene: `brand`, `model`, `model_year`, `milage`, `fuel_type`,
`engine`, `transmission`, `ext_col`, `int_col`, `accident`, `clean_title` og `price`. 
Testfilen har 125 690 biler uten pris.

Vi fikk ikke lastet ned dataene med Kaggle-API-et underveis, så vi brukte en kopi av de samme filene som ligger åpent på GitHub. 
Filene er de samme som på Kaggle (samme antall rader og kolonner).
`ml/download_data.py` henter dem direkte fra Kaggle hvis man har en API-nøkkel.

### Hva ble funnet ved å se på dataene

Analysen ligger i `ml/eda.py`. Utskriften er lagret i `ml/eda_output.txt` og figurene i `ml/figures/`.

- Noen kolonner mangler verdier: `fuel_type` (5 083), `accident` (2 452) og `clean_title` (21 419). `clean_title` har bare verdien "Yes", tomme felt ble tolket som "nei".
- Prisen er veldig skjevt fordelt. Medianen er rundt 30 800 dollar, men den dyreste bilen koster nesten 3 millioner. Med log skala ser fordelingen mye mer normal ut (`pris_fordeling.png`).
- Ingen enkeltkolonne forklarer prisen spesielt godt. Korrelasjonen med pris var 0,23 for årsmodell, 0,28 for hestekrefter og -0,28 for kjørelengde.
  Det ble tolket som at det er kombinasjonen av ting som betyr noe, og at det er mye støy i dataene.
- Det er veldig mange forskjellige modeller (1 897) og motorer (1 117) og mange av dem dukker bare opp noen få ganger.

### Feature engineering

Motorkolonnen er en tekst, f.eks. "252.0HP 3.9L 8 Cylinder Engine Gasoline Fuel". Den inneholder nyttig informasjon, det ble hentet ut hestekrefter, liter og antall sylindre med regex, 
og la til om motoren er turbo eller elektrisk. Fra girkassen ble det hentet ut om den er automat og hvor mange gir den har. I tillegg ble det regnet ut alder (2024 minus årsmodell) og kjørelengde per år, og gjort `accident` og `clean_title` om til 0 og 1.

Kategorier som finnes færre enn 20 ganger ble det slått sammen til "Annet". Det gjør at modellen ikke lærer seg sjeldne verdier utenat, 
og at nettsiden ikke krasjer hvis noen velger noe modellen ikke har sett før.

Denne koden ligger i `backend/car_features.py` og brukes både når det trener og når nettsiden lager et estimat. For å unngå at dataene behandles forskjellig i trening og i bruk.

Dataene ble delt tilfeldig i 80 % trening og 20 % validering (`random_state=42`).

## 3. Modellering

Begynte med to enkle modeller for å ha noe å sammenligne med:

1. **Gjennomsnitt:** gjetter alltid gjennomsnittsprisen fra treningsdataene.
2. **Ridge-regresjon:** en lineær modell på de numeriske kolonnene, trent på log(pris).

Så ble **LightGBM** prøvdt. Gradient boosting ble valgt fordi det pleier å fungere godt på tabelldata med både tall og kategorier, og fordi LightGBM håndterer manglende verdier og kategorier selv. 
Den er også rask nok til å kjøre på en vanlig PC. To varianter ble prøvdt: én trent på log(pris) og én trent direkte på pris. Antall trær ble valgt med early stopping på valideringssettet.

Resultater på valideringssettet (37 707 biler):

| Modell | RMSE | MAE | R² |
|---|---:|---:|---:|
| Gjennomsnitt | 74 573 | 29 087 | 0,000 |
| Ridge (log-pris) | 70 491 | 18 419 | 0,106 |
| LightGBM (log-pris) | 68 557 | 17 055 | 0,155 |
| LightGBM (pris) | 68 092 | 19 590 | 0,166 |

LightGBM var best men forbedringen i RMSE er ikke så stor som det var håpet. R² er også lav. Etter å ha sett på figuren `predikert_vs_faktisk.png` skyldes det at noen få ekstremt dyre biler drar RMSE veldig opp og at modellen ikke klarer å treffe på dem. 
Dessuten er dataene syntetiske og det er mye støy i dem. De beste på Kaggle-ledertavlen hadde rundt 63 000 i RMSE, så vi er ikke så langt unna med en ganske enkel løsning.

Det var interessant at log-pris ga lavest MAE, altså bedre estimater for "vanlige" biler, mens modellen trent på vanlig pris ga lavest RMSE. 
Vi endte med å bruke den siste fordi RMSE er metrikken i konkurransen. Man kunne argumentert for den andre siden de fleste som bruker nettsiden har en helt vanlig bil.

Early stopping stoppet allerede rundt 100 trær, noe som betyr at modellen fort begynner å tilpasse seg støyen.
Kjørelengde, motor, modell og årsmodell var de viktigste kolonnene (`feature_importance.png`).

Til slutt ble modellen trent på alle dataene og lagret den i `backend/model/`. Også en `submission.csv` for Kaggle-testsettet ble laget.

Hvis vi hadde hatt mer tid ville vi prøvd kryssvalidering i stedet for én splitt, tunet hyperparametrene mer og kombinert flere modeller (f.eks. CatBoost og XGBoost) som mange av toppløsningene på Kaggle gjorde.

## 4. Nettsiden

Backend er skrevet i FastAPI og laster den lagrede modellen når den starter. Den har to endepunkter:

- `GET /api/options` gir lister over merker, modeller, motorer osv. til skjemaet og tallene fra valideringen.
- `POST /api/predict` tar imot opplysningene om bilen og sender tilbake en estimert pris.

Frontend er en React-side på norsk. Når man velger merke vises bare modellene til det merket, og når man har valgt modell vises bare motorene som finnes for den modellen i datasettet. 
Det ble gjort fordi motor er fritekst i datasettet, og det ville vært nesten umulig for en bruker å skrive den på akkurat samme måte selv. 
I første versjon viste det bare de 80 vanligste modellene per merke, men da manglet en del modeller for de store merkene, så nå vises alle.

Hvis dette skulle vært et ekte produkt måtte man ha fulgt med på om prisnivået endrer seg over tid og trent modellen på nytt med nyere annonser. 
Estimatet bør ikke brukes som eneste grunnlag for å kjøpe eller selge en bil, særlig siden det er basert på amerikanske priser. 
Nettsiden lagrer ingen opplysninger om brukerne.

## 5. Kilder

- Walter Reade og Ashley Chow, *Regression of Used Car Prices*, Kaggle Playground Series S4E9, 2024.
   https://www.kaggle.com/competitions/playground-series-s4e9
- Taeef Najib, *Used Car Price Prediction Dataset*, Kaggle.
   https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset
- Ke m.fl., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017. https://lightgbm.readthedocs.io
- Dokumentasjonen til scikit-learn, pandas, FastAPI (https://fastapi.tiangolo.com), React (https://react.dev) og
   shadcn/ui (https://ui.shadcn.com).
- Kopi av Kaggle-dataene på GitHub: https://github.com/IsharaParanagamaGedara/Regression-of-Used-Car-Prices
- Ai som hjalp til å skrive deler av koden og nettsiden.
