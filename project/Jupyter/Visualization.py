import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ensure the 'data' DataFrame is preprocessed
data = preprocess_data(load_and_concat_data("train.csv", "test.csv"))

# 1. Scatter Plot: Age vs Fare, color by Survival
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Age", y="Fare", hue="Survived")
plt.title("Age vs Fare (Colored by Survival)")
plt.show()

# 2. Interactive Scatter Plot (Plotly): Age vs Fare, color by Pclass
fig = px.scatter(
    data, x="Age", y="Fare", color="Pclass", title="Age vs Fare by Passenger Class"
)
fig.show()

# 3. Line Plot (Multiple Lines): Average Age by Class and Gender
age_by_class_gender = data.groupby(["Pclass", "Sex"])["Age"].mean().unstack()
age_by_class_gender.plot(kind="line", marker="o")
plt.title("Average Age by Class and Gender")
plt.xlabel("Passenger Class")
plt.ylabel("Average Age")
plt.legend(title="Gender")
plt.show()

# 4. Interactive Line Plot (Plotly): Survival Rate by Family Size
survival_by_family = data.groupby("FamilySize")["Survived"].mean().reset_index()
fig = px.line(
    survival_by_family,
    x="FamilySize",
    y="Survived",
    title="Survival Rate by Family Size",
)
fig.show()

# 5. Histogram: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data["Age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# 6. Interactive Histogram (Plotly): Fare Distribution
fig = px.histogram(data, x="Fare", nbins=50, title="Fare Distribution")
fig.show()

# 7. Bar Chart: Survival Count by Embarked Port
survived_by_embarked = data.groupby("Embarked")["Survived"].sum().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(data=survived_by_embarked, x="Embarked", y="Survived")
plt.title("Survival Count by Embarked Port")
plt.show()

# 8. Horizontal Bar Chart: Average Fare by Passenger Class
avg_fare_by_class = data.groupby("Pclass")["Fare"].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(data=avg_fare_by_class, x="Fare", y="Pclass", orient="h")
plt.title("Average Fare by Passenger Class")
plt.show()

# 9. Pie Chart: Survival Rate
survival_counts = data["Survived"].value_counts()
fig = px.pie(
    values=survival_counts, names=["Did Not Survive", "Survived"], title="Survival Rate"
)
fig.show()

# 10. Box Plot: Fare by Class and Survival Status
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="Pclass", y="Fare", hue="Survived")
plt.title("Fare Distribution by Class and Survival")
plt.show()

# 11. 3D Scatter Plot: Age, Fare, and Family Size
fig = px.scatter_3d(
    data,
    x="Age",
    y="Fare",
    z="FamilySize",
    color="Survived",
    title="3D Scatter Plot of Age, Fare, and Family Size",
)
fig.show()

# 12. Sunburst Chart: Class, Gender, and Survival
fig = px.sunburst(
    data,
    path=["Pclass", "Sex", "Survived"],
    title="Sunburst Chart: Class, Gender, and Survival",
)
fig.show()

# 13. **(Bonus) Sankey Diagram**: Showing Survival Flow by Class and Gender
# Prepare data for Sankey diagram
sankey_data = (
    data.groupby(["Pclass", "Sex", "Survived"]).size().reset_index(name="count")
)
fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[
                    "1st Class",
                    "2nd Class",
                    "3rd Class",
                    "Male",
                    "Female",
                    "Survived",
                    "Did Not Survive",
                ],
            ),
            link=dict(
                source=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
                target=[3, 4, 5, 3, 4, 6, 3, 4, 6, 5, 6, 5, 6],
                value=sankey_data["count"],
            ),
        )
    ]
)
fig.update_layout(
    title_text="Sankey Diagram: Class, Gender, and Survival", font_size=10
)
fig.show()
