import numpy as np
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

    # Limpeza de dados
    print("Antes da limpeza:")
    print(df.isna().sum())  # Exibir a contagem de valores NaN
    df = df.drop(["fipe_code", "authentication", "gear", "year_model"], axis=1, errors="ignore")
    df = df.drop_duplicates()

    # Criar coluna de datas
    mapeamento = {'January': "1", 'February': "2", 'March': "3", 'April': "4", 'May': "5", 'June': "6",
                  'July': "7", 'August': "8", 'September': "9", 'October': "10", 'November': "11", 'December': "12"}
    if "month_of_reference" in df.columns and "year_of_reference" in df.columns:
        df["month_of_reference"] = df["month_of_reference"].map(mapeamento)
        df["year_of_reference"] = df["year_of_reference"].astype(str)
        df["date"] = df["month_of_reference"].str.cat(df["year_of_reference"], sep="/")
        df["date"] = pd.to_datetime(df["date"], format="%m/%Y").dt.strftime('%Y/%m')
        df = df.drop(columns=["month_of_reference", "year_of_reference"])

    print("Depois da limpeza:")
    print(df.isna().sum())  # Verificar novamente os valores NaN

    # Transformação das variáveis categóricas para numéricas (Label Encoding)
    df, label_encoders = apply_label_encoding(df)

    # Gráfico de distribuição
    plt.figure(figsize=(10, 6))
    if 'avg_price_brl' in df.columns:
        sns.histplot(df['avg_price_brl'], bins=30, kde=True)
        plt.title("Distribuição de Preços Médios de Carros")
        plt.xlabel("Preço Médio (BRL)")
        plt.ylabel("Frequência")
        plot_path = os.path.join("static", "plot.png")
        plt.savefig(plot_path)
        plt.close()  # Fecha o gráfico após salvar
    else:
        plot_path = None

    # Salvar dataset limpo
    clean_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"clean_{filename}")
    df.to_csv(clean_filepath, index=False)

    df_html = df.head().to_html(classes='table table-striped')
    return render_template("visualize.html", data=df_html, filename=filename, plot_path=plot_path)



# Rota para exibir gráficos
@app.route("/plot/<filename>")
def plot_data(filename, mcolors=None):
    clean_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"clean_{filename}")
    df = pd.read_csv(clean_filepath)

    # Transformação das variáveis categóricas para numéricas (Label Encoding)
    df, label_encoders = apply_label_encoding(df)

    if 'avg_price_brl' in df.columns and 'date' in df.columns:
        # Prepara os dados
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'avg_price_brl'])
        medias_data = df.groupby("date")["avg_price_brl"].mean()
        datas = medias_data.index.astype(str)
        valores = np.arange(len(datas))

        # Criar gráfico
        plt.figure(figsize=(12, 7))
        sns.scatterplot(x=datas, y=medias_data, c=valores, cmap="viridis")

        coeficientes = np.polyfit(valores, medias_data, 1)
        regressao = np.polyval(coeficientes, valores)
        plt.plot(datas, regressao, color="red", label="Regressão Linear")

        # Configurações do gráfico
        plt.title("Preço Médio dos Carros por Mês")
        plt.ylabel("Preço Médio")
        plt.xlabel("Data")
        plt.xticks(rotation=45)

        # Salvar o gráfico
        plot_path = os.path.join("static", "regression_plot.png")
        plt.savefig(plot_path)
        plt.close()

        return render_template("visualize_graficos.html", plot_path=plot_path, filename=filename)
    else:
        return "<p>Colunas necessárias não encontradas no dataset.</p>"



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

@app.route("/select_car/<filename>", methods=['GET', 'POST'])
def select_car(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    # Limpar nomes das colunas e remover espaços nos valores
    df.columns = df.columns.str.strip()  # Remover espaços em torno dos nomes das colunas
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()  # Remover espaços nos valores de strings apenas

    selected_cars_html = None
    columns = df.columns.tolist()  # Lista de colunas

    if request.method == 'POST':
        # Obter filtros preenchidos no formulário
        filters = {col: request.form.get(col).strip() for col in columns if request.form.get(col)}

        if filters:
            # Debug: Imprimir os filtros recebidos
            print("Filtros recebidos:", filters)

            try:
                # Aplicar os filtros manualmente para evitar erros no query
                filtered_df = df.copy()
                for col, value in filters.items():
                    filtered_df = filtered_df[filtered_df[col].astype(str) == value]

                # Debug: Verificar resultados
                print("DataFrame filtrado:\n", filtered_df)

                # Gerar HTML dos resultados
                if not filtered_df.empty:
                    selected_cars_html = filtered_df.to_html(classes='table table-striped', index=False)
                else:
                    selected_cars_html = "<p>Nenhum resultado encontrado com os filtros fornecidos.</p>"
            except Exception as e:
                print("Erro ao aplicar os filtros:", e)
                selected_cars_html = f"<p>Erro ao aplicar filtros: {e}</p>"

    return render_template(
        "select_car.html",
        columns=columns,
        selected_cars=selected_cars_html,
    )




if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)
