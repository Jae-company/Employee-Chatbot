import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Employee Attendance Insights", layout="wide")

st.title("📊 Employee Attendance Chatbot")
st.markdown("""
Welcome to the **Employee Attendance Insights Bot**!  
Ask questions in natural language and get instant insights on employee attendance based on department, location, and other filters.

---

### 💡 How to use this app:
- Type questions like:
  - *"What is the average attendance by Employee Peer rating?"*
  - *"Which department has the heighest attendance?"*
  - *"Show attendance by roles."*
- Click **"Ask"** to view the results below.

---
""")



# Load dataset
df = pd.read_csv('Employee_Performance_Dataset.csv')  # Adjust filename if needed

# Clean data
df['Attendance (%)'] = df['Attendance (%)'].astype(float)

# Train a simple intent classifier
training_data = {
    "department_attendance": [
        "What is the average attendance by department?",
        "Attendance in each department?",
        "Show department wise attendance"
    ],
    "peer_rating_attendance": [
        "Attendance by peer rating",
        "Does peer rating affect attendance?",
        "Show rating wise attendance"
    ],
    "job_role_attendance": [
        "Attendance by job role",
        "Which job roles have better attendance?",
        "Show role wise attendance"
    ],
    "overall_attendance": [
        "Give overall attendance summary",
        "Attendance overview",
        "Describe the attendance data"
    ]
}

X = sum(training_data.values(), [])
y = [key for key, samples in training_data.items() for _ in samples]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X_vec, y)

# Classifier function
def classify_intent(user_input):
    vec = vectorizer.transform([user_input])
    return model.predict(vec)[0]

# Response functions
def department_attendance(df):
    return df.groupby("Department")["Attendance (%)"].mean().sort_values(ascending=False)

def peer_rating_attendance(df):
    return df.groupby("Peer Rating")["Attendance (%)"].mean().sort_index()

def job_role_attendance(df):
    return df.groupby("Job Role")["Attendance (%)"].mean().sort_values(ascending=False)

def overall_attendance(df):
    return df["Attendance (%)"].describe()

# Streamlit app
st.title("👨‍💼 Employee Attendance Insight Chatbot")
st.write("Ask me about attendance stats based on department, location, peer rating, or roles.")

user_query = st.text_input("Your question here:")

if user_query:
    intent = classify_intent(user_query)

    if intent == "department_attendance":
        st.write("📊 Department-wise Attendance:")
        st.dataframe(department_attendance(df))
    elif intent == "peer_rating_attendance":
        st.write("⭐ Peer Rating vs Attendance:")
        st.dataframe(peer_rating_attendance(df))
    elif intent == "job_role_attendance":
        st.write("💼 Job Role-wise Attendance:")
        st.dataframe(job_role_attendance(df))
    elif intent == "overall_attendance":
        st.write("📈 Overall Attendance Summary:")
        st.write(overall_attendance(df))
    else:
        st.warning("🤖 Sorry, I couldn't understand your question.")
