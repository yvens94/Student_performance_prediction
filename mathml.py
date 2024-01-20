import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline



def predict_datapoint():
    st.title("MathML")
    st.header("Student Math Exam Performance Indicator")

    gender = st.selectbox("Gender", ["", "Male", "Female"], key='gender')
    race_ethnicity = st.selectbox("Race or Ethnicity", ["", "Group A", "Group B", "Group C", "Group D", "Group E"], key='race_ethnicity')
    parental_level_of_education = st.selectbox("Parental Level of Education", ["", "associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"], key='parental_level_of_education')
    lunch = st.selectbox("Lunch Type", ["", "free/reduced", "standard"], key='lunch')
    test_preparation_course = st.selectbox("Test Preparation Course", ["", "None", "Completed"], key='test_preparation_course')

    reading_score = st.number_input("Reading Score out of 100", min_value=0, max_value=100, key='reading_score')
    writing_score = st.number_input("Writing Score out of 100", min_value=0, max_value=100, key='writing_score')

    submitted = st.button("Predict your Math Score")

    if submitted:
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=float(reading_score),
            writing_score=float(writing_score)
        )

        pred_df = data.get_data_as_data_frame()
        st.write(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.subheader("The math_score prediction is")
        st.write(results[0])

if __name__ == "__main__":
    predict_datapoint()


