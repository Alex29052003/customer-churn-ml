import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# загружаем данные 
df = pd.read_csv("data/customers.csv")

# подключаем базу SQLite
conn = sqlite3.connect("database.db")

# сохраняем таблицу в базу данных
df.to_sql("customers", conn, if_exists="replace", index=False)

# SQL-запрос
query = "SELECT * FROM customers"
df_sql = pd.read_sql(query, conn)

# небольшой анализ данных
print("Churn distribution:\n", df_sql["churn"].value_counts())
print("Average tenure:", df_sql["tenure"].mean())
print("Median tenure:", df_sql["tenure"].median())
print("Average monthly charges:", df_sql["monthly_charges"].mean())
print("Average tenure by churn:\n", df_sql.groupby("churn")["tenure"].mean())

# подготовка данных
X = df_sql[["tenure", "monthly_charges"]]
y = df_sql["churn"]

# обучение модели
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Оценка модели
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Model coefficients (tenure, monthly_charges):", model.coef_)

# визуализация
df_sql["churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = stayed, 1 = left)")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.show()

# закрываем подключение 
conn.close()