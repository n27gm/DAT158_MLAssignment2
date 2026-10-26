# Rapport: Hva er en bruktbil verdt?

## 1. Problemstilling

Hva er en bruktbil egentlig verdt? Når man skal kjøpe eller selge en brukt bil er det ofte vanskelig å vite hva som er en rimelig pris. Prisen kan blant annet avhenge av: merke, modell, hvor gammel bilen er, hvor langt den har kjørt, motor, om den har vært i en ulykke og så videre. Målet med dette prosjektet er å lage en maskinlæringsmodell som, basert på alt man vet om bruktbilen, skal kunne gi gi et raskt anslag på hva bilen er verdt. Modellen skal være tilgjengelig gjennom en nettside. For å trene modellen vår, bruker vi datasettet fra Kaggle konkurransen *Regression of Used Car Prices* (Playground Series S4E9). 
Det er et syntetisk datasett som er laget ut fra et ekte datasett med amerikanske bilannonser. Kjørelengden er oppgitt i miles, og ettersom modellen gjelder det amerikanske markedet, ikke det norske, er prisene oppgitt i dollar. 

**Ytelse av modellen:** 
Konkurransen bruker RMSE (Root Mean Squared Error) som hovedmetrikk, i tillegg til MAE (Mean Absolute Error) og R² (Coefficient of Determination). Målet er å slå en enkel baseline som gjetter gjennomsnittsprisen på en bruktbil utifra den gitte informasjonen, og at nettsiden skal svare raskt.

Alt er kjørt på vanlige bærbare PC-er uten GPU.
Brukt: 
- Python (pandas, scikit-learn, LightGBM) til modellen
- FastAPI + React til nettsiden.

Liknende løsninger som finnes i dag er blant annet nettbil.no og pristips på finn.no. Det er også mulig å bestille en vurdering av bilen på naf.no eller autodeals.no. Det er også mulig å finne et estimat på en rimelig pris manuelt, men dette vil kreve mye arbeid og tid ettersom prisen på en bruktbil avhenger av flere faktorer.

## 2. Data

Treningsfilen inneholder data for 188 533 biler og kolonnene: `brand`, `model`, `model_year`, `milage`, `fuel_type`,
`engine`, `transmission`, `ext_col`, `int_col`, `accident`, `clean_title` og `price`. 
125 690 av bilene mangler pris.

Vi fikk ikke lastet ned dataene med Kaggle-API-et underveis, så vi bruker en kopi av de samme filene som ligger åpent på GitHub. 
Filene er de samme som på Kaggle (samme antall rader og kolonner).
`ml/download_data.py` henter dem direkte fra Kaggle hvis man har en API-nøkkel.

### Hva ble funnet ved å se på dataene

Analysen ligger i `ml/eda.py`. Utskriften er lagret i `ml/eda_output.txt` og figurene i `ml/figures/`.

- Noen kolonner mangler verdier: `fuel_type` (5 083 manglende verdier), `accident` (2 452) og `clean_title` (21 419). `clean_title` har bare verdien "Yes", så tomme felt ble tolket som "nei".
- Prisen er veldig skjevt fordelt. Medianen er rundt 30 800 dollar, men den dyreste bilen koster nesten 3 millioner. Med log skala ser fordelingen mye mer normal ut (se `pris_fordeling.png`).
- Ingen enkeltkolonne forklarer prisen spesielt godt. Korrelasjonen med pris var 0.23 for årsmodell, 0.28 for hestekrefter og   0.28 for kjørelengde. Det ble tolket som at det er kombinasjonen av ting som betyr noe, og at det er mye støy i dataene.
- Det er veldig mange forskjellige modeller (1 897) og motorer (1 117), og mange av dem dukker kun opp noen få ganger.

### Feature engineering

Motor-kolonnen består av tekst-verdier, f.eks. "252.0HP 3.9L 8 Cylinder Engine Gasoline Fuel". Én slik tekst inneholder mye nyttig informasjon; vi henter blant annet ut antall hestekrefter, liter og antall sylindre med regex, og om motoren er turbo eller elektrisk. Fra girkassen ble det hentet ut om den er automat eller ikke, og hvor mange gir den har. I tillegg ble det regnet ut alder (2024 minus årsmodell) og kjørelengde per år, og gjort `accident` og `clean_title` om til 0 og 1.

Kategorier som opptrer mindre enn 20 ganger blir slått sammen til én kategori kalt "Annet". Dette gjør at modellen ikke lærer seg sjeldne verdier utenatt, og at nettsiden ikke krasjer hvis noen velger noe modellen ikke har sett før.

Denne koden ligger i `backend/car_features.py` og brukes både når modellen trenes og når nettsiden lager et estimat. For å unngå overfitting og at dataene behandles forskjellig under trening og i bruk, ble dataene delt inn i treningsdata og test/validerings data. 80% av dataene brukes til trening, mens de resterende 20%-ene brukes til testing. Fordelingen av data skjer tilfeldig (`random_state=42`).

## 3. Modellering

Vi begynte med to enkle modeller for å ha noe å sammenligne med:

1. **Gjennomsnitt:** gjetter alltid gjennomsnittsprisen fra treningsdataene.
2. **Ridge-regresjon:** en lineær modell på de numeriske kolonnene, trent på log(pris).

Deretter prøvde vi **LightGBM**. Gradient boosting ble valgt fordi denne metoden ofte fungerer godt på tabelldata med både tall og kategorier, og fordi LightGBM håndterer manglende verdier og kategorier selv. Den er også rask nok til å kjøre på en vanlig PC. Vi prøvde to ulike varianter: én trent på log(pris), og én trent direkte på pris. Antall trær ble valgt med early stopping på valideringssettet.

Resultater fra valideringssettet (37 707 biler):

| Modell | RMSE | MAE | R² |
|---|---:|---:|---:|
| Gjennomsnitt | 74 573 | 29 087 | 0,000 |
| Ridge (log-pris) | 70 491 | 18 419 | 0,106 |
| LightGBM (log-pris) | 68 557 | 17 055 | 0,155 |
| LightGBM (pris) | 68 092 | 19 590 | 0,166 |

LightGBM ga best resultat, men forbedringen i RMSE var ikke så stor som forventet. R² var også relativt lav. Etter å ha sett på figuren `predikert_vs_faktisk.png`, tenker vi at det mest sannsynlig skyldes at noen få ekstremt dyre biler drar RMSE-verdien veldig opp og at modellen ikke klarer å treffe på dem. Dataene er dessuten syntetiske og inneholder mye støy. 
De beste på Kaggle-ledertavlen hadde rundt 63 000 i RMSE, så vi er ikke så langt unna med en ganske enkel løsning.

Det var interessant at log-pris ga lavest MAE, altså bedre estimater for "vanlige" biler, mens modellen trent på vanlig pris ga lavest RMSE. Vi endte med å bruke den sistnevnte ettersom RMSE var metrikken i konkurransen. Man kunne derimot også argumentert for å bruke den første i stedet, ettersom de fleste som bruker nettsiden har en helt vanlig bil.

Early stopping stoppet allerede rundt 100 trær, noe som betyr at modellen fort begynte å tilpasse seg støyen.
Kjørelengde, motor, modell og årsmodell er de viktigste kolonnene (`feature_importance.png`).

Til slutt ble modellen trent på alle dataene og lagret i `backend/model/`. Vi opprettet også `submission.csv` for Kaggle-testsettet.

Hvis vi hadde hatt mer tid ville vi prøvd kryssvalidering i stedet for én splitt, tunet hyperparametrene mer og kombinert flere modeller (f.eks. CatBoost og XGBoost) som mange av toppløsningene på Kaggle gjorde.

## 4. Nettsiden

Backend er skrevet i FastAPI og laster den lagrede modellen når den starter. Den har to endepunkter:

- `GET /api/options` gir lister over merker, modeller, motorer osv. til skjemaet og tallene fra valideringen.
- `POST /api/predict` tar imot opplysningene om bilen og sender tilbake en estimert pris.

Frontend er en React-side på norsk. Når man velger bilmerke, vises bare modellene til dette merket, og når man har valgt modell vises bare motorene som finnes for denne modellen i datasettet. Dette ble gjort fordi motor er fritekst i datasettet, og det ville vært nesten umulig for en bruker å skrive den på akkurat samme måte selv. 
I første versjon viste det bare de 80 vanligste modellene per merke, men da manglet en del modeller for de store merkene, men med den nåværende metoden, vises alle.

Hvis dette hadde vært et ekte produkt måtte man ha fulgt med på om prisnivået endrer seg over tid, og trent modellen på nytt med nyere annonser. 

Estimatet modellen vår anslår bør ikke brukes som eneste grunnlag for å kjøpe eller selge en bruktbil, særlig siden det er basert på amerikanske priser, men kan gi brukerne en omtrentlig idé om hva bilen kan være verdt. 
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
