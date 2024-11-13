from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Função para verificar se a extensão é permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Função para aplicar Label Encoding
def apply_label_encoding(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Salve o LabelEncoder para futuras inversões ou predições
    return df, label_encoders

# Rota para página inicial e upload do CSV
@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('visualize_data', filename=file.filename))
    return render_template("index.html")

# Rota para visualização dos dados e análises
@app.route("/visualize/<filename>")
def visualize_data(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    # Transformação das variáveis categóricas para numéricas (Label Encoding)
    df, label_encoders = apply_label_encoding(df)
    
    # Gráfico de distribuição
    plt.figure(figsize=(10, 6))
    sns.histplot(df['avg_price_brl'], bins=30, kde=True)
    plt.title("Distribuição de Preços Médios de Carros")
    plt.xlabel("Preço Médio (BRL)")
    plt.ylabel("Frequência")
    plot_path = os.path.join("static", "plot.png")
    plt.savefig(plot_path)

    df_html = df.head().to_html(classes='table table-striped')
    return render_template("visualize.html", data=df_html, filename=filename, plot_path=plot_path)

# Rota para exibir gráficos
@app.route("/plot/<filename>")
def plot_data(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    # Transformação das variáveis categóricas para numéricas (Label Encoding)
    df, label_encoders = apply_label_encoding(df)
    
    # Gráfico de distribuição
    plt.figure(figsize=(10, 6))
    sns.histplot(df['avg_price_brl'], bins=30, kde=True)
    plt.title("Distribuição de Preços Médios de Carros")
    plt.xlabel("Preço Médio (BRL)")
    plt.ylabel("Frequência")
    plot_path = os.path.join("static", "plot.png")
    plt.savefig(plot_path)
    plt.close()  # Fecha o gráfico após salvar
    return render_template("visualize.html", plot_path=plot_path, filename=filename)

# Rota para configurar e treinar o modelo
@app.route("/train/<filename>", methods=['GET', 'POST'])
def train_model(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    # Transformação das variáveis categóricas para numéricas (Label Encoding)
    df, label_encoders = apply_label_encoding(df)
    
    if request.method == 'POST':
        target = request.form['target']
        features = request.form.getlist('features')
        
        X = df[features]
        y = df[target]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escolha de modelo
        model_choice = request.form['model']
        if model_choice == 'linear_regression':
            model = LinearRegression()
        elif model_choice == 'random_forest':
            model = RandomForestRegressor()
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        # Gráfico de performance do modelo
        y_pred = model.predict(X_test)
        
        # Gráfico de dispersão: valores reais vs predições
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.title("Valores Reais vs. Previsões")
        plt.xlabel("Valor Real")
        plt.ylabel("Previsão")
        plt.grid(True)
        performance_plot_path = os.path.join("static", "performance_plot.png")
        plt.savefig(performance_plot_path)
        plt.close()

        # Salvar o modelo
        model_path = os.path.join("models", f"{model_choice}.joblib")
        joblib.dump(model, model_path)
        
        return render_template("train_model.html", score=score, filename=filename, performance_plot_path=performance_plot_path)
    
    return render_template("train_model.html", columns=df.columns, filename=filename)

# Rota para fazer predições
@app.route("/predict/<filename>", methods=['GET', 'POST'])
def make_prediction(filename):
    model_path = os.path.join("models", "random_forest.joblib")  # exemplo de modelo carregado
    model = joblib.load(model_path)
    prediction = None

    if request.method == 'POST':
        features = []
        for i in range(len(request.form)):
            feature_value = request.form.get(f'feature_{i}')
            try:
                features.append(float(feature_value))  # Tenta converter para float
            except ValueError:
                features.append(0)  # Atribui 0 caso não consiga converter
        prediction = model.predict([features])[0]
        
        # Gráfico de predições
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(range(len(features))), y=features)
        plt.title("Previsões Realizadas")
        plt.xlabel("Índice das Features")
        plt.ylabel("Valor das Features")
        prediction_plot_path = os.path.join("static", "prediction_plot.png")
        plt.savefig(prediction_plot_path)

    return render_template("predictions.html", prediction=prediction, filename=filename, prediction_plot_path=prediction_plot_path)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)
