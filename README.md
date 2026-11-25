# Hotel Reservation Cancellation Predictor
**CIS 412 Team Project – 99% Confidence**
Vedika Thareja · Rhys Roberts · Luizeth Denogean · Anthony Caballero · Alexander Salazar · Larissa Backen

A Streamlit app that predicts whether a hotel booking will be canceled, using the tuned Random Forest from our Colab notebook (test accuracy 0.899, AUC 0.958).

## Files in this repository
| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python libraries Streamlit Cloud installs |
| `Hotel_Reservations.csv` | The dataset (the app trains the model from it on first load) |
| `Hotel_Cancellation_Project.ipynb` | Our full analysis notebook |

## Deploy it (about 10 minutes, one person does this)
1. **Create the GitHub repo:** go to github.com → **New repository** → name it `hotel-cancellation-predictor` → set it to **Public** → **Create**.
2. **Upload files:** click **Add file → Upload files** and drag in `app.py`, `requirements.txt`, `Hotel_Reservations.csv`, `README.md`, and the notebook. Click **Commit changes**.
3. **Deploy:** go to **share.streamlit.io**, sign in with GitHub, click **Create app → Deploy a public app from GitHub**.
   - Repository: `your-username/hotel-cancellation-predictor`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy**. The first load takes about 1–2 minutes while it installs libraries and trains the model.
5. Copy the public link (ends in `.streamlit.app`) and submit it together with the GitHub repo link.

## Run it on your own computer (optional)
```
pip install -r requirements.txt
streamlit run app.py
```
