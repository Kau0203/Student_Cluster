# 📊 Student Clustering App
🚀 Live App: https://studentcluster-zzbytqtpps3cc27hqs2vbn.streamlit.app/

---

**📌 Project Overview**

This Streamlit app performs unsupervised clustering of students based on their academic performance using the KMeans algorithm.

---

1.Upload a dataset with academic features

2.Automatically group students into clusters

3.Visualize clusters using scatter plot, bar chart, and pie chart

---


**🧠 Machine Learning Used**

Algorithm: KMeans Clustering

Input Features:
CGPA, Attendance_Percentage, Assignments_Completed, Lab_Score

Output: Cluster labels (0 to 3)

---

**📂 Sample Dataset Format**
| Student\_ID | CGPA | Attendance\_Percentage | Assignments\_Completed | Lab\_Score |
| ----------- | ---- | ---------------------- | ---------------------- | ---------- |
| 201         | 8.4  | 90                     | 10                     | 85         |
| 202         | 6.8  | 70                     | 7                      | 60         |
| 203         | 9.2  | 96                     | 10                     | 92         |
| 204         | 7.5  | 80                     | 8                      | 72         |


---

**🔵 Scatter Plot:**
Visualizes CGPA vs Lab Score colored by cluster

**📊 Bar Chart:**
Shows student count per cluster

**🥧 Pie Chart:**
Shows total Lab Score percentage per cluster
