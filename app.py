import streamlit as st
import openai
import pandas as pd
import json
from dotenv import load_dotenv
import os
from openai import OpenAI
from pycaret.regression import load_model
from langfuse import Langfuse
import traceback




# === Wczytanie modelu PyCaret ===
model = load_model("model_lasso_v1_2023")




# === Wczytanie zmiennych środowiskowych z .env ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)




# === Inicjalizacja Langfuse ===
langfuse = Langfuse()




# === Funkcja do zamiany sekund na HH:MM:SS ===
def format_seconds(seconds):
  minutes = int(seconds // 60)
  remaining_seconds = int(seconds % 60)
  hours = minutes // 60
  minutes = minutes % 60
  return f"{hours}h {minutes}m {remaining_seconds}s"




# === Konwersja z mm:ss lub sekund do sekund ===
def parse_time_input(time_str):
  try:
      if ":" in time_str:
          minutes, seconds = map(int, time_str.strip().split(":"))
          return minutes * 60 + seconds
      else:
          return int(time_str)
  except:
      st.error("Niepoprawny format czasu. Użyj mm:ss lub liczbę sekund.")
      return None




# === Nagłówek aplikacji ===
st.title("🏃‍♀️ Oszacuj swój czasu **półmaratonu**")


# === Pole tekstowe na dane użytkownika ===
user_input = st.text_area("Podaj dane: PŁEĆ, WIEK, CZAS NA 5 KM (np. Kobieta, 27 lat, 23:40 na 5 km):")


# === Przycisk predykcji ===
if st.button("Oblicz czas półmaratonu"):
   if not user_input.strip():
       st.warning("Najpierw wpisz wiadomość.")
   else:
       prompt = f"""Wyodrębnij z poniższej wiadomości dane użytkownika w formacie JSON:
{{
"Wiek": int,
"Płeć": 0 (kobieta) lub 1 (mężczyzna),
"5 km Czas": czas w sekundach (int)
}}
Wiadomość: {user_input}"""


       try:
           # === Ręczne śledzenie Langfuse ===
           trace = langfuse.trace(name="polmaraton_prediction", user_id="anon")
           span = trace.span(name="extract_user_data", input=prompt)


           response = client.chat.completions.create(
               model="gpt-4",
               messages=[{"role": "user", "content": prompt}],
               temperature=0.3
           )


           content = response.choices[0].message.content.strip()
           span.output = content
           span.end()
           #trace.end()


           if not content:
               st.error("❌ Nie udało się uzyskać odpowiedzi od modelu. Spróbuj ponownie później.")
               st.stop()


           try:
               extracted = json.loads(content)
           except json.JSONDecodeError:
               st.error("❌ Nie udało się odczytać danych. Upewnij się, że podałeś wiek, płeć i czas na 5 km.")
               st.stop()


           # Sprawdzenie, czy są wszystkie wymagane dane
           required_keys = ["Wiek", "Płeć", "5 km Czas"]
           missing_keys = [key for key in required_keys if key not in extracted]
           if missing_keys:
               st.error(f"⚠️ Brakuje danych: {', '.join(missing_keys)}")
               st.stop()


           czas_5km = extracted.get("5 km Czas")
           if not isinstance(czas_5km, int) or czas_5km < 800 or czas_5km > 2400:
               st.error("⛔️ Czas na 5 km wydaje się nielogiczny. Użyj formatu mm:ss lub sprawdź poprawność.")
               st.stop()


           # Stwórz DataFrame
           X = pd.DataFrame([{
               "Wiek": extracted["Wiek"],
               "Płeć": extracted["Płeć"],
               "5 km Czas": czas_5km
           }])


           prediction = model.predict(X)[0]


           if prediction < czas_5km * 3.5:
               st.error("⛔️ Model przewidział nielogicznie szybki półmaraton. Sprawdź dane wejściowe.")
           else:
               formatted = format_seconds(prediction)
               st.success(f"⏱️ Szacowany czas półmaratonu: {formatted}")


       except Exception as e:
           st.error(f"❌ Błąd podczas analizy lub predykcji: {e}")
           st.text(traceback.format_exc())  # debug: pełen traceback






