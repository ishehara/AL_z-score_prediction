import streamlit as st
import pickle
import numpy as np

# Load the pickled model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
poly = pickle.load(open('poly_features.pkl', 'rb'))

# page configuration

st.set_page_config(page_title="Z-Score prediction", page_icon="🧑‍🎓", layout="centered")

st.title("Z-Score Prediction")

st.markdown(
    """
    This app predicts the A/L examination Z-Scores based on students performance in Sri Lanka 

    <hr>
    """,unsafe_allow_html=True
)

submap = {'ACCOUNTING': 0,
 'AGRICULTURAL SCIENCE': 1,
 'AGRO TECHNOLOGY': 2,
 'ART': 3,
 'BIO SYSTEMS TECHNOLOGY': 4,
 'BIO-RESOURCE TECHNOLOGY': 5,
 'BIOLOGY': 6,
 'BUDDHISM': 7,
 'BUDDHIST CIVILIZATION': 8,
 'BUSINESS STATISTICS': 9,
 'BUSINESS STUDIES': 10,
 'CARNATIC MUSIC': 11,
 'CHEMISTRY': 12,
 'CHRISTIAN CIVILIZATION': 13,
 'CHRISTIANITY': 14,
 'CIVIL TECHNOLOGY': 15,
 'COMBINED MATHEMATICS': 16,
 'COMMUNICATION & MEDIA STUDIES': 17,
 'DANCING(BHARATHA)': 18,
 'DANCING(INDIGENOUS)': 19,
 'DRAMA AND THEATRE (SINHALA)': 20,
 'ECONOMICS': 21,
 'ELECTRICAL,ELECTRONIC AND IT': 22,
 'ENGINEERING TECHNOLOGY': 23,
 'ENGLISH': 24,
 'FOOD TECHNOLOGY': 25,
 'GEOGRAPHY': 26,
 'GREEK & ROMAN CIVILIZATION': 27,
 'HIGHER MATHEMATICS': 28,
 'HINDU CIVILIZATION': 29,
 'HINDUISM': 30,
 'HISTORY OF EUROPE': 31,
 'HISTORY OF INDIA': 32,
 'HISTORY OF MODERN WORLD': 33,
 'HISTORY OF SRI LANKA & EUROPE': 34,
 'HISTORY OF SRI LANKA & INDIA': 35,
 'HISTORY OF SRI LANKA & MODERN WORLD': 36,
 'HOME ECONOMICS': 37,
 'INFORMATION & COMMUNICATION TECHNOLOGY': 38,
 'ISLAM': 39,
 'ISLAMIC CIVILIZATION': 40,
 'LOGIC & SCIENTIFIC METHOD': 41,
 'MATHEMATICS': 42,
 'MECHANICAL TECHNOLOGY': 43,
 'ORIENTAL MUSIC': 44,
 'PALI': 45,
 'PHYSICS': 46,
 'POLITICAL SCIENCE': 47,
 'SINHALA': 48,
 'TAMIL': 49,
 'WESTERN MUSIC': 50}

streamMap = {
 'ARTS': 1,
 'BIOLOGICAL SCIENCE': 2,
 'BIOSYSTEMS TECHNOLOGY': 3,
 'COMMERCE': 4,
 'ENGINEERING TECHNOLOGY': 5,
 'PHYSICAL SCIENCE': 7}

genderMap = {'female': 2, 'male': 3}

sysllabusMap = {'new': 0, 'old': 1}

gradeMap={'A': 0, 'B': 2, 'C': 3, 'F': 4, 'S': 5, 'Withheld': 6, 'Absent': 1}

with st.form("prediction Form"):


    stream=st.selectbox("select the Stream :", streamMap.keys())

    sysllabus=st.selectbox("select the Syllabus :", sysllabusMap.keys())


    sub1=st.selectbox("select the Subject 1 :", submap.keys())
    grade1=st.selectbox("Subject 1 grade :", gradeMap.keys())

    sub2=st.selectbox("select the Subject 2 :", submap.keys())
    grade2=st.selectbox("Subject 2 grade :", gradeMap.keys())

    sub3=st.selectbox("select the Subject 3 :", submap.keys())
    grade3=st.selectbox("Subject 3 grade :", gradeMap.keys())

    gender=st.selectbox("Gender: ", genderMap.keys())



    submitted = st.form_submit_button(label="Predict")

if submitted:
    
    input_data = np.array([[

        streamMap[stream],
        submap[sub1],
        gradeMap[grade1],
        submap[sub2],
        gradeMap[grade2],
        submap[sub3],
        gradeMap[grade3],
        sysllabusMap[sysllabus],
        genderMap[gender]

    ]])
    
    try:
        scaled = scaler.transform(input_data)
        poly = poly.transform(scaled)
        prediction = model.predict(poly)
        st.success(f"Predicted Z-Score: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"An error occured: {e}")
