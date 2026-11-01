import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Employee Attrition Analytics", layout="wide")

st.title("Employee Attrition Prediction & Risk Scoring System")
st.caption("Machine Learning-Based HR Analytics Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("IBM-HR-Analytics-Employee-Attrition-and-Performance.csv")

@st.cache_resource
def load_model():
    return joblib.load("employee_attrition_model.pkl")

df = load_data()
model = load_model()

# Prepare features exactly as during training
X = df.drop(columns=[
    'Attrition', 'EmployeeCount',
    'EmployeeNumber', 'Over18', 'StandardHours'
], errors='ignore')

df["Risk_Score"] = model.predict_proba(X)[:, 1] * 100
df["Risk_Level"] = pd.cut(
    df["Risk_Score"],
    bins=[-1, 30, 60, 100],
    labels=["Low Risk", "Medium Risk", "High Risk"]
).astype(str)

# Sidebar filter
st.sidebar.header("Dashboard Filters")
departments = ["All"] + sorted(df["Department"].dropna().unique().tolist())
department = st.sidebar.selectbox("Select Department", departments)

filtered = df if department == "All" else df[df["Department"] == department]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Employees", len(filtered))
c2.metric("Employees Left", int((filtered["Attrition"] == "Yes").sum()))
c3.metric("Attrition Rate", f"{(filtered['Attrition'] == 'Yes').mean()*100:.1f}%")
c4.metric("High Risk Employees", int((filtered["Risk_Level"] == "High Risk").sum()))

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Attrition Distribution")
    chart = filtered["Attrition"].value_counts().reset_index()
    chart.columns = ["Attrition", "Employees"]
    st.plotly_chart(px.pie(chart, names="Attrition", values="Employees",
                           hole=0.4), use_container_width=True)

with right:
    st.subheader("Department-wise Attrition")
    dept = filtered.groupby("Department", as_index=False).agg(
        Employees=("Attrition", "size"),
        Left=("Attrition", lambda x: (x == "Yes").sum())
    )
    st.plotly_chart(px.bar(dept, x="Department", y="Left",
                           color="Department"), use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("Employee Risk Distribution")
    risk = filtered["Risk_Level"].value_counts().reset_index()
    risk.columns = ["Risk Level", "Employees"]
    st.plotly_chart(px.bar(risk, x="Risk Level", y="Employees",
                           color="Risk Level"), use_container_width=True)

with right:
    st.subheader("Monthly Income vs Risk")
    st.plotly_chart(px.scatter(
        filtered, x="MonthlyIncome", y="Risk_Score",
        color="Risk_Level", hover_data=["Age", "Department"]
    ), use_container_width=True)

st.subheader("Employee Risk Report")
columns = ["Age", "Department", "JobRole", "MonthlyIncome",
           "OverTime", "Risk_Score", "Risk_Level"]
st.dataframe(
    filtered[columns].sort_values("Risk_Score", ascending=False),
    use_container_width=True
)

st.download_button(
    "Download Risk Report",
    filtered[columns].to_csv(index=False),
    file_name="employee_attrition_risk_report.csv",
    mime="text/csv"
)

st.info("Risk scores are model-estimated probabilities, not certainty that an employee will leave. This educational dataset is synthetic and is not actual Palo Alto Networks employee data.")
