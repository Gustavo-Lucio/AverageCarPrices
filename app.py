import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Verifica extensões permitidas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Aplica Label Encoding às colunas categóricas
def apply_label_encoding(df, categorical_columns):
    label_encoders = {}
    original_model_values = {}  # Dicionário para armazenar os valores originais da coluna 'model'

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

        if col == 'model':
            # Armazenar os valores originais de 'model'
            original_model_values = dict(zip(le.transform(le.classes_), le.classes_))

    # Após o Label Encoding, converte de volta os valores de 'model' para os nomes originais
    if 'model' in df.columns:
        df['model'] = df['model'].map(original_model_values)

    return df, label_encoders

# Função para limpar e preparar o dataset
def clean_data(filepath):
    df = pd.read_csv(filepath)

    # Limpeza de dados
    print("Antes da limpeza:")
    print(df.isna().sum())  # Exibir a contagem de valores NaN

    # Eliminar colunas desnecessárias
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

    # Aplicar Label Encoding somente em colunas categóricas (não em 'date' 'fuel' 'brand')
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns[
        (categorical_columns != 'date') &
        (categorical_columns != 'fuel') &
        (categorical_columns != 'brand')]
    # Excluir 'date' 'fuel' 'brand'

    df, label_encoders = apply_label_encoding(df, categorical_columns)

    return df, label_encoders


@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Limpeza e preparação dos dados
            df, _ = clean_data(filepath)

            # Verificar se o arquivo já contém 'clean_' e evitar duplicação
            filename = file.filename
            if not filename.startswith("clean_"):
                filename = f"clean_{filename}"

            clean_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            df.to_csv(clean_filepath, index=False)
            
            # Redireciona para a visualização após upload
            return redirect(url_for('visualize_data', filename=filename))
    return render_template("index.html")

@app.route("/visualize/<filename>")
def visualize_data(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    # Exibe uma pré-visualização dos dados
    df_html = df.head().to_html(classes='table table-striped')

    # Gerar gráfico de distribuição de preços
    plot_path = None
    if 'avg_price_brl' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['avg_price_brl'], bins=30, kde=True)
        plt.title("Distribuição de Preços Médios de Carros")
        plt.xlabel("Preço Médio (BRL)")
        plt.ylabel("Frequência")
        plot_path = os.path.join("static", "plot.png")
        plt.savefig(plot_path)
        plt.close()  # Fecha o gráfico após salvar

    return render_template("visualize.html", data=df_html, filename=filename, plot_path=plot_path)


@app.route("/plot/<filename>")
def visualize_graphs(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    # Gráfico de Regressão (usando Matplotlib)
    if 'avg_price_brl' in df.columns and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'avg_price_brl'])

        # Agrupar por mês e ano
        df['year_month'] = df['date'].dt.to_period('M')
        medias_data = df.groupby("year_month")["avg_price_brl"].mean()

        # Transformar os dados de datas para string com formato 'YYYY-MM'
        datas = medias_data.index.astype(str)
        valores = np.arange(len(datas))

        # Gráfico de Regressão
        plt.figure(figsize=(12, 7))
        sns.scatterplot(x=datas, y=medias_data, c=valores, cmap="viridis")

        # Regressão Linear
        coeficientes = np.polyfit(valores, medias_data, 1)
        regressao = np.polyval(coeficientes, valores)
        plt.plot(datas, regressao, color="red", label="Regressão Linear")

        plt.title("Preço Médio dos Carros por Mês")
        plt.ylabel("Preço Médio")
        plt.xlabel("Ano-Mês")

        # Ajuste dos rótulos no eixo X para mostrar apenas ano e mês
        plt.xticks(rotation=45)

        # Salvar o gráfico
        regression_plot_path = os.path.join("static", "regression_plot.png")
        plt.savefig(regression_plot_path)
        plt.close()

    # Gráfico de Distribuição de Preços Médios
    if 'avg_price_brl' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['avg_price_brl'], bins=30, kde=True)
        plt.title("Distribuição de Preços Médios de Carros")
        plt.xlabel("Preço Médio (BRL)")
        plt.ylabel("Frequência")
        price_distribution_plot_path = os.path.join("static", "price_distribution_plot.png")
        plt.savefig(price_distribution_plot_path)
        plt.close()

    # Gráfico de Distribuição do Tamanho dos Motores
    if 'engine_size' in df.columns:
        # Definir as cores para cada combustível
        cores2 = [px.colors.qualitative.Plotly[0], px.colors.qualitative.Plotly[1], px.colors.qualitative.D3[2]]

        # Plotando o histograma de distribuição do tamanho do motor
        plt.figure(figsize=(12, 7))
        sns.histplot(df, x="engine_size", bins=30, hue="fuel", stat="probability", multiple="stack",
                     palette=cores2, alpha=0.7, kde=True)  # Histograma do tamanho dos motores

        plt.title("Distribuição do Tamanho dos Motores (cm³)")
        plt.ylabel("Frequência Relativa")
        plt.xlabel("Tamanho do Motor")

        # Adicionando a legenda
        Alcohol = mlines.Line2D([], [], color=px.colors.qualitative.D3[2], marker='s', linestyle='None',
                                markersize=12, label='Alcohol')
        Diesel = mlines.Line2D([], [], color=px.colors.qualitative.Plotly[1], marker='s', linestyle='None',
                               markersize=12, label='Diesel')
        Gasoline = mlines.Line2D([], [], color=px.colors.qualitative.Plotly[0], marker='s',
                                 linestyle='None', markersize=12, label='Gasoline')

        plt.legend(handles=[Alcohol, Diesel, Gasoline], title="Combustível", title_fontsize="x-large",
                   fontsize="x-large")

        # Salvar o gráfico
        engine_size_distribution_plot_path = os.path.join("static", "engine_size_distribution_plot.png")
        plt.savefig(engine_size_distribution_plot_path)
        plt.close()

    # Gráfico de Proporção de Combustíveis (usando Plotly)
    if 'fuel' in df.columns:
        df_combustivel = pd.DataFrame()
        df_combustivel["Combustível"] = df["fuel"].unique()  # Separando os tipos de combustíveis
        df_combustivel["Frequência"] = np.array(df["fuel"].value_counts())  # Frequência Absoluta

        # Seleção de cores para o gráfico
        fig = px.pie(df_combustivel, values="Frequência", names="Combustível",
                     title="Proporção de Combustíveis no DataSet", template="presentation",
                     color_discrete_sequence=cores2)  # Plot do gráfico de pizza
        fig.update_layout(width=800, height=500)
        fig.update_traces(textposition="inside", textinfo="percent+label")

        # Salvar o gráfico de pizza como imagem
        fuel_proportion_plot_path = os.path.join("static", "fuel_proportion_plot.png")
        fig.write_image(fuel_proportion_plot_path)

    # Renderização no Template
    return render_template("visualize_graficos.html",
                           regression_plot_path=regression_plot_path,
                           price_distribution_plot_path=price_distribution_plot_path,
                           engine_size_distribution_plot_path=engine_size_distribution_plot_path,
                           fuel_proportion_plot_path=fuel_proportion_plot_path,
                           filename=filename)


@app.route("/select_car_model/<filename>", methods=['GET', 'POST'])
def select_car_model(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Limpeza do dataset
    df, _ = clean_data(filepath)

    # Verifica se a coluna 'model' existe
    if 'model' not in df.columns:
        return "Erro: Coluna 'model' não encontrada no dataset.", 400

    # Obtém os modelos únicos
    car_models = df['model'].unique()

    if request.method == 'POST':
        selected_models = request.form.getlist('selected_models')
        if not selected_models:
            return "Erro: Nenhum modelo selecionado.", 400

        # Filtra os dados pelos modelos selecionados
        filtered_df = df[df['model'].isin(selected_models)]

        # Salva o dataset filtrado
        filtered_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"filtered_{filename}")
        filtered_df.to_csv(filtered_filepath, index=False)

        return redirect(url_for('train_model', filename=f"filtered_{filename}"))

    return render_template("select_car_model.html", car_models=car_models, filename=filename)

@app.route("/train/<filename>", methods=['GET', 'POST'])
def train_model(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return f"Erro: Arquivo {filename} não encontrado.", 404
    
    df, label_encoders = clean_data(filepath)

    # Transformação das colunas categóricas para numéricas, incluindo 'model'
    if 'model' in df.columns:
        le = LabelEncoder()
        df['model'] = le.fit_transform(df['model'])

    if request.method == 'POST':
        try:
            target = request.form['target']
            features = request.form.getlist('features')
            
            if target not in df.columns or not set(features).issubset(df.columns):
                return "Erro: Coluna alvo ou features inválidas.", 400
            
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_choice = request.form['model']
            if model_choice == 'linear_regression':
                model = LinearRegression()
            elif model_choice == 'random_forest':
                model = RandomForestRegressor()
            else:
                return "Erro: Modelo inválido selecionado.", 400

            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            y_pred = model.predict(X_test)
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.7, label="Previsões")
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", label="Ideal")
            plt.title("Valores Reais vs. Previsões")
            plt.xlabel("Valor Real")
            plt.ylabel("Previsão")
            plt.grid(True)
            plt.legend()
            performance_plot_path = os.path.join("static", "model_performance.png")
            plt.savefig(performance_plot_path)
            plt.close()

            os.makedirs('models', exist_ok=True)
            model_path = os.path.join("models", f"{model_choice}_{filename}.joblib")
            joblib.dump(model, model_path)

            return render_template(
                "train_model.html",
                score=score,
                filename=filename,
                performance_plot_path=performance_plot_path,
                columns=df.columns
            )
        except Exception as e:
            return f"Erro ao treinar o modelo: {e}", 500

    return render_template("train_model.html", columns=df.columns, filename=filename, score=None)

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
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
