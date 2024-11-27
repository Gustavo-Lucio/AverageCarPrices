import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
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

@app.route("/visualize_graphs_static/<filename>")
def visualize_graphs_static(filename):
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
    return render_template("visualize_graficos_estaticos.html",
                           regression_plot_path=regression_plot_path,
                           price_distribution_plot_path=price_distribution_plot_path,
                           engine_size_distribution_plot_path=engine_size_distribution_plot_path,
                           fuel_proportion_plot_path=fuel_proportion_plot_path,
                           filename=filename)


@app.route("/visualize_graphs_interactive/<filename>")
def visualize_graphs_interactive(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    # Gráfico de Preços Médios ao Longo do Tempo (Plotly)
    price_trend_dates = df['date'].dropna().unique()

    # Verifica se os dados em 'date' são strings ou datetime e converte para datetime se necessário
    if isinstance(price_trend_dates[0], str):  # Se os dados são strings
        price_trend_dates = pd.to_datetime(price_trend_dates, errors='coerce')  # Converte para datetime e ignora erros

    # Agora podemos aplicar o strftime para formatar corretamente
    price_trend_dates = [date.strftime('%Y-%m') for date in price_trend_dates if
                         pd.notnull(date)]  # Convertendo para strings

    price_trend_values = df.groupby('date')['avg_price_brl'].mean().values
    price_trend_values = price_trend_values.tolist()  # Convertendo para lista de valores numéricos

    # Gráfico Interativo de Preços Médios ao Longo do Tempo
    price_trend_data = go.Scatter(
        x=price_trend_dates,
        y=price_trend_values,
        mode='lines+markers',
        name='Preço Médio',
    )
    price_trend_layout = go.Layout(
        title='Evolução dos Preços Médios de Carros',
        xaxis={'title': 'Data'},
        yaxis={'title': 'Preço Médio (BRL)'},
        height=500,
        width=1000,
    )
    price_trend_fig = go.Figure(data=[price_trend_data], layout=price_trend_layout)

    # Gráfico da Relação entre Combustível e Tamanho do Motor (Plotly Express)
    df_combustivel = df.groupby("fuel")["engine_size"].mean().reset_index()
    df_combustivel["Tamanho do Motor"] = df_combustivel["engine_size"]

    # Criando o gráfico de barras
    fig_combustivel_motor = px.bar(
        df_combustivel,
        x="fuel",
        y="Tamanho do Motor",
        title="Média do Tamanho do Motor (cm³) por Combustível",
        color="fuel",
        color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]  # Cores para os combustíveis
    )
    fig_combustivel_motor.update_layout(width=800, height=500, showlegend=False, title_x=0.5)

    # Gráfico de Comparação de Preços Médios e Quantidade de Modelos por Marca
    marcas = np.array(
        ['BMW', 'Ferrari', 'Ford', 'GM - Chevrolet', 'Honda', 'Jeep', 'Mclaren', 'Mercedes-Benz', 'Toyota',
         'VW - VolksWagen'])  # Marcas escolhidas

    dfm = df[df['brand'].isin(marcas)]  # Filtra as marcas selecionadas
    df_marcas = pd.DataFrame()
    df_marcas['Marca'] = marcas
    df_marcas['Preço Médio'] = list(dfm.groupby('brand')['avg_price_brl'].mean())  # Média dos preços
    df_marcas["Modelos"] = list(dfm.groupby("brand")["model"].nunique())  # Quantidade de modelos diferentes por marca

    # Gráfico de subplots
    fig_comparacao = sp.make_subplots(rows=1, cols=2)

    # Preço Médio
    indices_ordenados1 = df_marcas["Preço Médio"].sort_values(ascending=False).index
    fig_comparacao.add_trace(go.Bar(
        x=df_marcas["Marca"].iloc[indices_ordenados1],
        y=df_marcas["Preço Médio"].sort_values(ascending=False),
        name="Preço Médio",
        marker_color=px.colors.qualitative.Alphabet[0]
    ), row=1, col=1)

    # Quantidade de Modelos
    indices_ordenados2 = df_marcas["Modelos"].sort_values(ascending=False).index
    fig_comparacao.add_trace(go.Bar(
        x=df_marcas["Marca"].iloc[indices_ordenados2],
        y=df_marcas["Modelos"].sort_values(ascending=False),
        name="Quantidade de Modelos",
        marker_color=px.colors.qualitative.Bold[8]
    ), row=1, col=2)

    fig_comparacao.update_layout(
        title_text="Comparação de Marcas Famosas: Preço Médio e Quantidade de Modelos",
        title_x=0.5,
        title_font=dict(size=20, color="black"),
        showlegend=False,
        margin=dict(t=120, b=60, l=60, r=60),  # Ajuste a margem superior para mais espaço

        annotations=[
            dict(
                x=0.12, y=1.05, xref='paper', yref='paper',  # Ajuste o valor de y para 1.05
                text='Preço Médio por Marca (Milhões de Reais)', showarrow=False,
                font=dict(size=15),
                xanchor='center',  # Ajusta o alinhamento das anotações
                yanchor='bottom'  # Ajuste o ancoramento para evitar sobreposição
            ),
            dict(
                x=0.88, y=1.05, xref='paper', yref='paper',  # Ajuste o valor de y para 1.05
                text='Quantidade de Modelos Diferentes por Marca', showarrow=False,
                font=dict(size=15),
                xanchor='center',  # Ajusta o alinhamento das anotações
                yanchor='bottom'  # Ajuste o ancoramento para evitar sobreposição
            )
        ],
        font=dict(size=15),
        height=700,  # Altura fixa
        width=1100,  # Largura fixa
    )

    fig_comparacao.update_xaxes(tickangle=40)

    # Preparando os dados para o gráfico de Proporções de Combustíveis por Marca
    df_marcas_combustivel = dfm.groupby("brand")["engine_size"].mean().reset_index()
    df_marcas_combustivel["Tamanho Médio dos Motores"] = df_marcas_combustivel[
        "engine_size"]  # Tamanho médio do motor por marca

    proporcoes = dfm.groupby(["brand", "fuel"]).size() / dfm.groupby(
        "brand").size()  # Calculando as proporções para cada marca
    df_proporcoes = proporcoes.unstack(level='fuel').reset_index().fillna(0)

    # Criando o gráfico de barras de proporções de combustíveis
    fig_prop = px.bar(df_proporcoes, "brand", ["Gasoline", "Diesel", "Alcohol"],
                      color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"])

    # Criando o gráfico combinado de Tamanho Médio dos Motores e Proporção de Combustíveis
    fig_comparacao_combustivel = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Tamanho Médio dos Motores", "Proporção de Combustíveis por Marca"),
        shared_yaxes=False
    )

    # Adicionando o gráfico de Tamanho Médio dos Motores
    fig_comparacao_combustivel.add_trace(go.Bar(
        x=df_marcas_combustivel["brand"], y=df_marcas_combustivel["Tamanho Médio dos Motores"],
        name="Tamanho Médio dos Motores", marker_color=px.colors.qualitative.D3[9], showlegend=False),
        row=1, col=1
    )

    # Adicionando os gráficos de Proporções de Combustíveis
    for combu in fig_prop.data:
        fig_comparacao_combustivel.add_trace(combu, row=1, col=2)

    # Atualizando o layout do gráfico combinado
    fig_comparacao_combustivel.update_layout(
        title_text="Comparação de Proporção de Combustíveis e Tamanho dos Motores por Marca",
        title_x=0.5,
        title_font=dict(size=20, color="black"),
        font=dict(size=15),
        barmode="stack",
        height=600,
        width=1200
    )
    fig_comparacao_combustivel.update_xaxes(tickangle=40)

    # Renderizando a template com os três gráficos
    return render_template(
        "visualize_graficos_interativos.html",
        filename=filename,
        price_trend_fig=price_trend_fig.to_html(full_html=False),
        fig_combustivel_motor=fig_combustivel_motor.to_html(full_html=False),
        fig_comparacao=fig_comparacao.to_html(full_html=False),
        fig_comparacao_combustivel=fig_comparacao_combustivel.to_html(full_html=False)
    )


def train_models(df):
    # Inicializando o LabelEncoder
    label_encoder = LabelEncoder()

    # Converter variáveis categóricas para numéricas
    categorical_cols = ['brand', 'model', 'fuel']  # Colunas categóricas
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Se houver colunas de datas, convertê-las para numérico
    if 'date' in df.columns:
        # Convertendo a coluna 'date' para datetime no formato 'YYYY/MM'
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m')

        # Calculando o número de meses desde uma data base, por exemplo, Janeiro de 2000
        base_date = pd.Timestamp('2000-01-01')
        df['months_since_base'] = (df['date'] - base_date) // pd.Timedelta(days=30)

        # Extraindo o ano e o mês como colunas numéricas
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Remover a coluna 'date' original, pois já extraímos as informações necessárias
        df = df.drop(columns=['date'])

    # Preparando os dados para o treinamento
    X = df.drop('avg_price_brl', axis=1)  # Features
    y = df['avg_price_brl']  # Target

    # Dividindo em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelos
    models = {
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    # Salvar os modelos para uso posterior
    for model_name, model in trained_models.items():
        joblib.dump(model, f'model_{model_name}.pkl')

    return trained_models


# Função para prever o preço com base nos dados de entrada
def predict_price(df, model_name, input_data):
    # Carregar o modelo treinado
    model = joblib.load(f'model_{model_name}.pkl')

    # Garantir que o LabelEncoder seja usado da mesma forma que foi durante o treinamento
    label_encoder = LabelEncoder()

    # Converter as variáveis categóricas em valores numéricos, assim como no treinamento
    categorical_cols = ['brand', 'model', 'fuel']  # As mesmas colunas categóricas usadas no treinamento
    for col in categorical_cols:
        input_data[col] = label_encoder.fit_transform([input_data[col]])[0]

    # Converte o input_data para um DataFrame (como foi feito durante o treinamento)
    input_data_df = pd.DataFrame([input_data])

    # Predição
    predicted_price = model.predict(input_data_df)

    return predicted_price[0]


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Recebe os dados do formulário
        brand = request.form.get('brand')
        model = request.form.get('model')
        fuel = request.form.get('fuel')
        engine_size = float(request.form.get('engine_size'))
        age_years = int(request.form.get('age_years'))  # Idade do veículo
        date = request.form.get('date')  # Data de fabricação (ano e mês)

        # Processar a data (transformando para datetime)
        manufacture_date = pd.to_datetime(date, format='%Y-%m')

        # Calculando o número de meses desde uma data base (exemplo: Janeiro de 2000)
        base_date = pd.Timestamp('2000-01-01')
        months_since_base = (manufacture_date - base_date) // pd.Timedelta(days=30)

        # Cria o dicionário de dados para predição
        input_data = {
            'brand': brand,
            'model': model,
            'fuel': fuel,
            'engine_size': engine_size,
            'age_years': age_years,
            'months_since_base': months_since_base,  # Dados numéricos da data
            'year': manufacture_date.year,  # Ano extraído da data
            'month': manufacture_date.month  # Mês extraído da data
        }

        # Carregar os dados limpos e treinar os modelos
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'clean_fipe_2022.csv')
        df, _ = clean_data(filepath)


        trained_models = train_models(df)

        # Prever com o modelo escolhido pelo usuário
        model_name = request.form.get('model_choice', 'Random Forest')
        predicted_price = predict_price(df, model_name, input_data)

        return render_template('prediction_result.html', predicted_price=predicted_price)

    return render_template("predict_form.html")

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
    
    # Carregar e processar os dados
    df = pd.read_csv(filepath)
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

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
            
            # Salvar gráfico no diretório static
            os.makedirs("static", exist_ok=True)
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
                performance_plot_path="model_performance.png",
                columns=df.columns
            )
        except Exception as e:
            return f"Erro ao treinar o modelo: {e}", 500

    return render_template("train_model.html", columns=df.columns, filename=filename, score=None)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
