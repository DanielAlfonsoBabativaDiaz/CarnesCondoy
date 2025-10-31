from flask import Flask
from flask import render_template
from flask import request
from datetime import datetime, timedelta
import LinearRegression
import joblib
import numpy as np
import pickle
import pandas as pd


app = Flask (__name__)
model = joblib.load("modelo_vencimiento.pkl")


def generar_etiqueta(fecha_corte, dias_predichos):
    fecha_corte_dt = datetime.strptime(fecha_corte, "%Y-%m-%d")
    fecha_vencimiento = fecha_corte_dt + timedelta(days=int(dias_predichos))
    return {
        "corte": fecha_corte_dt.strftime('%d/%m/%Y'),
        "vencimiento": fecha_vencimiento.strftime('%d/%m/%Y'),
        "mensaje": f"CECINA DE CABRA\nFecha de corte: {fecha_corte_dt.strftime('%d/%m/%Y')}\nFecha de vencimiento: {fecha_vencimiento.strftime('%d/%m/%Y')}"
    }



@app.route('/casos_de_uso_ML')
def casos_de_uso_ML():
    return render_template('casos_de_uso_ML.html')

@app.route('/navbar')
def navbar():
    return render_template('navbar.html')

@app.route('/')
def home():
    Myname = "Carnes Condoy"
    return render_template('index.html', name=Myname)

@app.route("/prediccion", methods=["GET", "POST"])
def predecir():
    etiqueta = None

    if request.method == "POST":
        fecha_corte = request.form['fecha_corte']
        tipo_carne = request.form['tipo_carne']
        productos = request.form['productos']
        empaque = request.form['empaque']

        entrada = pd.DataFrame([{
            "fecha_corte": pd.to_datetime(fecha_corte).toordinal(),
            "tipo_carne": tipo_carne,
            "productos_utilizados": productos,
            "tipo_empaque": empaque
        }])

        dias = model.predict(entrada)[0]
        etiqueta = generar_etiqueta(fecha_corte, dias)

    return render_template("prediccion.html", etiqueta=etiqueta)





if __name__== '__main__':
    app.run(debug=True)