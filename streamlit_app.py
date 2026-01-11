import streamlit as st
import joblib
import pandas as pd

# Page settings
st.set_page_config(page_title="Insurance Scoring", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('insurance_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_assets()

# --- Language Selection ---
with st.sidebar:
    st.title("Settings / Настройки")
    lang = st.radio("Select Language / Выберите язык", ["English", "Русский"])

# --- Translation Dictionary ---
texts = {
    "English": {
        "title": "Travel Insurance Propensity Scoring",
        "desc": "ML-driven solution for identifying high-propensity customers.",
        "header": "Customer Profile",
        "age": "Age",
        "income": "Annual Income (INR)",
        "family": "Family Members",
        "emp": "Employment Sector",
        "grad": "Higher Education",
        "chronic": "Chronic Conditions",
        "flyer": "Frequent Flyer Status",
        "abroad": "Previous International Travel",
        "button": "Calculate Probability",
        "result": "Conversion Probability",
        "high": "Status: High-Priority Lead",
        "med": "Status: Medium-Priority Lead",
        "low": "Status: Low-Priority Lead",
        "details": "Model Methodology",
        "private": "Private Sector/Self Employed",
        "gov": "Government Sector"
    },
    "Русский": {
        "title": "Скоринг туристического страхования",
        "desc": "ML-решение для выявления наиболее перспективных клиентов.",
        "header": "Профиль клиента",
        "age": "Возраст",
        "income": "Годовой доход (в рупиях)",
        "family": "Членов семьи",
        "emp": "Тип занятости",
        "grad": "Высшее образование",
        "chronic": "Хронические заболевания",
        "flyer": "Часто летает самолетами",
        "abroad": "Был за границей ранее",
        "button": "Рассчитать вероятность",
        "result": "Вероятность покупки",
        "high": "Статус: Высокий приоритет",
        "med": "Статус: Средний приоритет",
        "low": "Статус: Низкий приоритет",
        "details": "Методология модели",
        "private": "Частный сектор / ИП",
        "gov": "Госслужба"
    }
}

t = texts[lang]

# --- UI Layout ---
st.title(t["title"])
st.write(t["desc"])
st.markdown("---")

st.subheader(t["header"])
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(t["age"], 18, 100, 28)
    income = st.number_input(t["income"], 100000, 2500000, 800000)
    family = st.slider(t["family"], 1, 10, 4)

with col2:
    # We display translated text but keep original values for the model
    emp_display = [t["private"], t["gov"]]
    emp_choice = st.selectbox(t["emp"], emp_display)
    emp_raw = "Private Sector/Self Employed" if emp_choice == t["private"] else "Government Sector"
    
    grad_choice = st.radio(t["grad"], ["Yes", "No"], format_func=lambda x: "Да" if (x == "Yes" and lang == "Русский") else x)
    chronic = st.checkbox(t["chronic"])

with col3:
    flyer_choice = st.selectbox(t["flyer"], ["Yes", "No"], format_func=lambda x: "Да" if (x == "Yes" and lang == "Русский") else x)
    abroad_choice = st.selectbox(t["abroad"], ["Yes", "No"], format_func=lambda x: "Да" if (x == "Yes" and lang == "Русский") else x)

if st.button(t["button"]):
    input_df = pd.DataFrame([{
        "Age": age,
        "Employment Type": emp_raw,
        "GraduateOrNot": grad_choice,
        "AnnualIncome": income,
        "FamilyMembers": family,
        "ChronicDiseases": 1 if chronic else 0,
        "FrequentFlyer": flyer_choice,
        "EverTravelledAbroad": abroad_choice
    }])

    processed_data = preprocessor.transform(input_df)
    probability = model.predict_proba(processed_data)[0][1]
    
    st.markdown("---")
    res_col, bar_col = st.columns([1, 2])
    
    with res_col:
        st.metric(t["result"], f"{probability*100:.2f}%")
        if probability > 0.70: st.info(t["high"])
        elif 0.35 <= probability <= 0.70: st.info(t["med"])
        else: st.info(t["low"])

    with bar_col:
        st.progress(probability)

with st.expander(t["details"]):
    if lang == "English":
        st.write("- **Algorithm:** Gradient Boosting Classifier\n- **Accuracy:** 84.17%")
    else:
        st.write("- **Алгоритм:** Градиентный бустинг\n- **Точность:** 84.17%")
