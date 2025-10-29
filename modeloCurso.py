import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import io, base64


np.random.seed(42)
n = 500
niveles = ['Secundaria', 'Tecnico', 'Universitario']

horas = np.random.normal(loc=25, scale=10, size=n).clip(5, 60)
foros = np.random.poisson(lam=5, size=n).clip(0, 15)
nivel = np.random.choice(niveles, size=n)

def prob_aprobacion(h, f, n):
    base = 0.3 + 0.01*h + 0.05*f
    if n == 'Tecnico': base += 0.1
    elif n == 'Universitario': base += 0.2
    return np.random.rand() < min(base, 0.95)

aprobo = [ 'Si' if prob_aprobacion(h, f, n) else 'No' for h, f, n in zip(horas, foros, nivel) ]

df = pd.DataFrame({
    'HorasConexion': horas.round(1),
    'ForosRespondidos': foros,
    'NivelEducativo': nivel,
    'Aprobo': aprobo
})

df.to_csv('./dataseets/curso_aprobacion.csv', index=False)
print(" Dataset generado y guardado")

df['NivelEducativo'] = LabelEncoder().fit_transform(df['NivelEducativo'])
df['Aprobo'] = df['Aprobo'].map({'Si': 1, 'No': 0})

x = df.drop('Aprobo', axis=1)
y = df['Aprobo']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

with open('modelo_aprobacion.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler_aprobacion.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(" Modelo y scaler guardados")

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlación entre variables")
plt.show()

sns.countplot(data=df, x='NivelEducativo', hue='Aprobo')
plt.title("Aprobación por Nivel Educativo")
plt.show()

def cargar_datos():
    df = pd.read_csv('./dataseets/curso_aprobacion.csv')
    df['NivelEducativo'] = LabelEncoder().fit_transform(df['NivelEducativo'])
    df['Aprobo'] = df['Aprobo'].map({'Si': 1, 'No': 0})
    return df

def evaluar_modelo(model, scaler):
    df = cargar_datos()
    x = df.drop('Aprobo', axis=1)
    y = df['Aprobo']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test_scaled = scaler.transform(x_test)
    y_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_html = report_df.round(2).to_html(classes="table table-bordered table-striped", border=0)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicho 0', 'Predicho 1'],
                yticklabels=['Real 0', 'Real 1'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    conf_matrix_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return round(accuracy, 2), report_html, conf_matrix_img