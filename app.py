import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Student Clustering App", layout="centered")

# Load the trained KMeans model
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Title
st.title("📊 Student Academic Clustering")

# Upload CSV
uploaded_file = st.file_uploader("Upload student_academic_data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Student_ID' in df.columns:
        df_clean = df.drop("Student_ID", axis=1)
    else:
        st.warning("Column 'Student_ID' not found, skipping drop.")
        df_clean = df.copy()

    # Predict clusters
    df['Cluster'] = kmeans.predict(df_clean)

    st.subheader("📋 Clustered Data")
    st.dataframe(df)

    # Scatter Plot (CGPA vs Lab Score)
    st.subheader("🔵 Scatter Plot (CGPA vs Lab Score)")
    fig1, ax1 = plt.subplots()
    scatter = ax1.scatter(df['CGPA'], df['Lab_Score'], c=df['Cluster'], cmap='viridis')
    ax1.set_xlabel("CGPA")
    ax1.set_ylabel("Lab Score")
    ax1.set_title("Student Clusters")
    st.pyplot(fig1)

    # Bar Chart - Cluster Size
    st.subheader("📊 Cluster Size Distribution")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color='skyblue')
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Number of Students")
    st.pyplot(fig2)

    # Pie Chart - Lab Score per Cluster
    st.subheader("🥧 Total Lab Score per Cluster")
    lab_score_sum = df.groupby('Cluster')['Lab_Score'].sum()
    fig3, ax3 = plt.subplots()
    ax3.pie(lab_score_sum, labels=lab_score_sum.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title("Total Lab Score per Cluster")
    st.pyplot(fig3)

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        right: 0;
        bottom: 0;
        padding: 10px;
        text-align: right;
        font-size: 14px;
        color: gray;
    }
    </style>
    <div class="footer">
        Developed by Kaumudi <br>
        <a href="mailto:kaumudigavkhadkar275@gmail.com">vprakashpate@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)
