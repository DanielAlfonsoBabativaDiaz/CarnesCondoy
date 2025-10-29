import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import io, base64
import os

# Simulación de datos
np.random.seed(42)
n = 500
categorias = ['Ropa', 'Electrónica', 'Hogar']

edad = np.random.randint(18, 65, size=n)
genero = np.random.choice(['M', 'F'], size=n)
historial = np.random.poisson(lam=4, size=n).clip(1, 15)
tiempo = np.random.normal(loc=180, scale=60, size=n).clip(60, 600)
visitadas = np.random.poisson(lam=3, size=n).clip(1, 10)

def prob_categoria(e, g, h, t, v):
    base = h + v + (t / 100)
    if g == 'F' and h > 5: return 'Ropa'
    elif t > 300: return 'Electrónica'
    else: return np.random.choice(categorias)

compra = [prob_categoria(e, g, h, t, v) for e, g, h, t, v in zip(edad, genero, historial, tiempo, visitadas)]

df = pd.DataFrame({
    'Edad': edad,
    'Genero': genero,
    'HistorialCompras': historial,
    'TiempoEnSitio': tiempo.round(1),
    'CategoriasVisitadas': visitadas,
    'CategoriaObjetivo': compra
})

os.makedirs("datasets", exist_ok=True)
os.makedirs("PKL", exist_ok=True)
df.to_csv("datasets/ecommerce_data.csv", index=False)
print("✅ Dataset generado y guardado")

# Preprocesamiento
le_genero = LabelEncoder()
df['Genero'] = le_genero.fit_transform(df['Genero'])

le_target = LabelEncoder()
df['CategoriaObjetivo'] = le_target.fit_transform(df['CategoriaObjetivo'])

X = df.drop('CategoriaObjetivo', axis=1)
y = df['CategoriaObjetivo']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_scaled, y_train)

# Guardar modelo y transformadores
with open("PKL/knn_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("PKL/scaler_knn.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("PKL/encoder_knn.pkl", "wb") as f:
    pickle.dump(le_target, f)

print("✅ Modelo y scaler guardados")

# Evaluación
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy del modelo: {accuracy:.2f}")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_html = report_df.round(2).to_html(classes="table table-bordered table-striped", border=0)

def generar_matriz_confusion_base64(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return img_base64

conf_matrix_img = generar_matriz_confusion_base64(y_test, y_pred)
print("✅ Imagen de matriz de confusión generada en base64")

