from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import os
import base64
from io import BytesIO
import LinearRegression
import joblib
import numpy as np
import pickle
import pandas as pd

load_dotenv()

app = Flask (__name__)
app.secret_key = os.getenv("SECRET_KEY", "cambiame-por-secreto")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SQLALCHEMY_DATABASE_URI", "sqlite:///database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
model = joblib.load("modelo_vencimiento.pkl")

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
class Etiqueta(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tipo_carne = db.Column(db.String(100))
    productos = db.Column(db.String(100))
    empaque = db.Column(db.String(100))
    fecha_corte = db.Column(db.String(20))
    fecha_vencimiento = db.Column(db.String(20))
    imagen_base64 = db.Column(db.Text)  # Aquí guardaremos la imagen en formato Base64
    fecha_creacion = db.Column(db.DateTime, default=datetime.utcnow)    
    
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def generar_etiqueta(fecha_corte, dias_predichos):
    fecha_corte_dt = datetime.strptime(fecha_corte, "%Y-%m-%d")
    fecha_vencimiento = fecha_corte_dt + timedelta(days=int(dias_predichos))
    return {
        "corte": fecha_corte_dt.strftime('%d/%m/%Y'),
        "vencimiento": fecha_vencimiento.strftime('%d/%m/%Y'),
        "mensaje": f"CECINA DE CABRA\nFecha de corte: {fecha_corte_dt.strftime('%d/%m/%Y')}\nFecha de vencimiento: {fecha_vencimiento.strftime('%d/%m/%Y')}"
    }

def crear_etiqueta_imagen(etiqueta):
    ancho, alto = 400, 200
    imagen = Image.new("RGB", (ancho, alto), "white")
    draw = ImageDraw.Draw(imagen)

    try:
        fuente = ImageFont.truetype("arial.ttf", 20)
    except:
        fuente = ImageFont.load_default()

    texto = f"""
    CONDOY
    CECINA DE CABRA

    Fecha de corte: {etiqueta['corte']}
    Fecha de vencimiento: {etiqueta['vencimiento']}
    """

    draw.multiline_text((20, 20), texto.strip(), fill="black", font=fuente, spacing=5)

    ruta = "static/img/etiqueta_generada.png"
    imagen.save(ruta)
    return ruta

def crear_etiqueta_imagen(etiqueta):
    ancho, alto = 400, 200
    imagen = Image.new("RGB", (ancho, alto), "white")
    draw = ImageDraw.Draw(imagen)

    try:
        fuente = ImageFont.truetype("arial.ttf", 20)
    except:
        fuente = ImageFont.load_default()

    texto = f"""
    CONDOY
    CECINA DE CABRA

    Fecha de corte: {etiqueta['corte']}
    Fecha de vencimiento: {etiqueta['vencimiento']}
    """

    draw.multiline_text((20, 20), texto.strip(), fill="black", font=fuente, spacing=5)

    # Guardar imagen en disco
    ruta = "static/img/etiqueta_generada.png"
    imagen.save(ruta)

    # Convertir a Base64 para guardar en la BD
    buffer = BytesIO()
    imagen.save(buffer, format="PNG")
    imagen_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return ruta, imagen_base64

@app.route('/navbar')
def navbar():
    return render_template('navbar.html')

@app.route('/')
def home():
    Myname = "Carnes Condoy"
    return render_template('index.html', name=Myname)

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html", user=current_user)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # Validaciones básicas
        if not username or not email or not password:
            flash("Todos los campos son obligatorios.", "warning")
            return redirect(url_for("register"))

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Usuario o email ya registrado.", "danger")
            return redirect(url_for("register"))

        # Crear usuario
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)  # iniciar sesión automáticamente
        flash("Registro exitoso. Has iniciado sesión.", "success")
        return redirect(url_for("dashboard"))

    # GET
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email_or_username = request.form.get("email_or_username", "").strip().lower()
        password = request.form.get("password", "")

        # Buscar usuario por email o username
        user = User.query.filter(
            (User.email == email_or_username) | (User.username == email_or_username)
        ).first()

        if not user:
            flash("Usuario no existe.", "danger")
            return redirect(url_for("login"))

        if not user.check_password(password):
            flash("Contraseña incorrecta.", "danger")
            return redirect(url_for("login"))

        login_user(user)
        flash("Inicio de sesión exitoso.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(email=email).first()
        if not user:
            flash("No se encontró ninguna cuenta con ese correo.", "warning")
            return redirect(url_for("forgot_password"))

        flash("Se ha enviado un correo con instrucciones para restablecer la contraseña (simulado).", "info")
        return redirect(url_for("login"))

    return render_template("forgot_password.html")

@app.route("/me")
def me():
    if not current_user.is_authenticated:
        return jsonify({"authenticated": False}), 401
    return jsonify({
        "authenticated": True,
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at.isoformat()
    })

@app.route("/prediccion", methods=["GET", "POST"])
@login_required
def predecir():
    etiqueta = None
    ruta_imagen = None

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

        # Generar imagen + versión Base64
        ruta_imagen, imagen_base64 = crear_etiqueta_imagen(etiqueta)

        # Guardar en la base de datos
        nueva_etiqueta = Etiqueta(
            tipo_carne=tipo_carne,
            productos=productos,
            empaque=empaque,
            fecha_corte=etiqueta['corte'],
            fecha_vencimiento=etiqueta['vencimiento'],
            imagen_base64=imagen_base64
        )

        db.session.add(nueva_etiqueta)
        db.session.commit()

    return render_template("prediccion.html", etiqueta=etiqueta, ruta_imagen=ruta_imagen)


@app.route("/etiquetas")
@login_required
def etiquetas():
    todas = Etiqueta.query.order_by(Etiqueta.fecha_creacion.desc()).all()
    return render_template("etiquetas.html", etiquetas=todas)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get("message", "").lower()

    # Aquí programas respuestas básicas o conectas IA real
    if "hola" in user_message:
        response = "¡Hola! ¿En qué puedo ayudarte?"
    elif "precio" in user_message:
        response = "Los precios varían según el peso y el tipo de carne."
    else:
        response = "Lo siento, no entendí tu mensaje, ¿puedes reformular?"

    return jsonify({"response": response})

if __name__ == "__main__":
    def create_tables():
        with app.app_context():
            db.create_all()

    create_tables()
    app.run(debug=True)
