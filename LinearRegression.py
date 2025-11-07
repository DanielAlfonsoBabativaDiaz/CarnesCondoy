import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

data = {
    "fecha_corte": ["2025-10-01", "2025-10-05", "2025-10-10"],
    "tipo_carne": ["Cabra", "Cabra", "Cabra"],
    "productos_utilizados": ["Sal", "Vinagre", "Condimentos"],
    "tipo_empaque": ["Vacío", "Vacío", "Atmósfera modificada"],
    "fecha_vencimiento": ["2025-10-15", "2025-10-20", "2025-10-25"]
}

df = pd.DataFrame(data)
df["dias_vencimiento"] = (pd.to_datetime(df["fecha_vencimiento"]) - pd.to_datetime(df["fecha_corte"])).dt.days
df["fecha_corte"] = pd.to_datetime(df["fecha_corte"]).map(lambda x: x.toordinal())

X = df[["fecha_corte", "tipo_carne", "productos_utilizados", "tipo_empaque"]]
y = df["dias_vencimiento"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(), ["tipo_carne", "productos_utilizados", "tipo_empaque"])
], remainder='passthrough')

model = Pipeline(steps=[
    ("pre", preprocessor),
    ("reg", RandomForestRegressor())
])

model.fit(X, y)
joblib.dump(model, "modelo_vencimiento.pkl")
