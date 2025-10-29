import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

data = {
    "Rainfall": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "Temperature": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    "CoffePrice": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
}

df = pd.DataFrame(data)

x = df [["Rainfall", "Temperature"]]
y = df [["CoffePrice"]]

model = LinearRegression()
model.fit(x, y)

df ["Predicted CoffePrice"] = model.predict(x)

joblib.dump(model, "linear_regression_model.pkl")
