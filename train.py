from sklearn.model_selection import train_test_split
import pandas as pd
from ft_linear_regression import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('./data.csv')
X = df[['km']].values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_scaled, y_train)

y_pred_line = reg.predict(X_test_scaled)
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color="red", s=30)
m2 = plt.scatter(X_test, y_test, color="blue", s=30)
plt.plot(X_test, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
