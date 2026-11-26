import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/student_data.csv")

@st.cache_resource
def train_model():
    data = load_data()
    features = ["study_hours", "attendance", "previous_marks", "sleep_hours"]
    X = data[features]
    y = data["final_marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, mean_absolute_error(y_test, predictions), r2_score(y_test, predictions), data

model, mae, r2, data = train_model()

st.title("🎓 Student Performance Predictor")
st.write("Predict a student's expected exam marks using a Machine Learning model.")
st.info("This is an educational demo trained on an illustrative sample dataset, not real student records.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Enter student details")
    study_hours = st.slider("Daily study hours", 0.5, 12.0, 4.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    previous_marks = st.slider("Previous marks (%)", 0, 100, 60)
    sleep_hours = st.slider("Daily sleep hours", 3.0, 12.0, 7.0, 0.5)
    if st.button("Predict marks", type="primary", use_container_width=True):
        sample = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "sleep_hours": sleep_hours
        }])
        result = max(0, min(100, float(model.predict(sample)[0])))
        st.session_state["prediction"] = result

with right:
    st.subheader("Prediction result")
    if "prediction" in st.session_state:
        score = st.session_state["prediction"]
        st.metric("Expected marks", f"{score:.1f} / 100")
        if score >= 75:
            st.success("Excellent predicted performance!")
        elif score >= 50:
            st.warning("Good effort. There is room to improve.")
        else:
            st.error("Consider improving study time and attendance.")
        st.progress(int(score))
    else:
        st.write("Enter the details and click **Predict marks**.")

st.divider()
st.subheader("Model performance")
c1, c2, c3 = st.columns(3)
c1.metric("Training records", len(data))
c2.metric("Mean Absolute Error", f"{mae:.2f}")
c3.metric("R² score", f"{r2:.2f}")

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Study hours vs final marks")
fig, ax = plt.subplots()
ax.scatter(data["study_hours"], data["final_marks"], alpha=0.65)
ax.set_xlabel("Daily study hours")
ax.set_ylabel("Final marks")
ax.set_title("Study hours and student marks")
st.pyplot(fig)
