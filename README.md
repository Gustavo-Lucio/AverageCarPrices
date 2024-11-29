
# Aplicativo Flask para Processamento de Dados e Aprendizado de Máquina

Este projeto é um aplicativo web baseado em Flask, projetado para preprocessamento de dados, visualização e treinamento de modelos de aprendizado de máquina. Ele permite o upload de arquivos CSV, limpeza e transformação de dados, visualização de insights e treinamento de modelos para tarefas de previsão.

---

## Funcionalidades

1. **Upload de Arquivos**: Permite o envio de arquivos CSV para análise.
2. **Limpeza de Dados**: Lida com valores ausentes, remove duplicatas e codifica colunas categóricas.
3. **Visualização**: Gera gráficos como histogramas e scatter plots para explorar distribuições e tendências.
4. **Aprendizado de Máquina**:
   - Treina modelos como Regressão Linear e Random Forest.
   - Avalia o desempenho usando métricas e visualizações.
5. **Predição**: Usa modelos treinados para fazer previsões com base nos dados fornecidos.
6. **Gerenciamento de Modelos**: Salva e carrega modelos treinados para reutilização.

---

## Tecnologias Utilizadas

- **Bibliotecas Python**:
  - `Flask`: Framework para aplicativos web.
  - `Pandas`: Manipulação e análise de dados.
  - `NumPy`: Computação numérica.
  - `Matplotlib` & `Seaborn`: Visualização de dados.
  - `Plotly` & `Kaleido`: Visualizações interativas e exportação de gráficos.
  - `scikit-learn`: Modelos e ferramentas de preprocessamento para aprendizado de máquina.
  - `Joblib`: Persistência de modelos.

- **Frontend**:
  - `HTML` e `CSS`: Para renderizar páginas web e estilização.
  - Classes do Bootstrap: Para uma interface responsiva.

---

## Pré-requisitos

- Python 3.8 ou superior.
- Instalar as bibliotecas necessárias:
  ```bash
  pip install flask pandas numpy matplotlib seaborn plotly kaleido scikit-learn joblib
  ```

---

## Uso

### 1. Executando o Aplicativo

1. Clone este repositório:
   ```bash
   git clone <url-do-repositorio>
   cd <diretorio-do-repositorio>
   ```
2. Crie os diretórios necessários:
   ```bash
   mkdir uploads static models
   ```
3. Execute o aplicativo Flask:
   ```bash
   python app.py
   ```
4. Acesse o aplicativo em [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

### 2. Fluxo de Trabalho

#### **Passo 1: Upload de Arquivo CSV**
- Navegue até a página inicial e envie um arquivo CSV.
- Certifique-se de que o arquivo contenha colunas relevantes para a análise.

#### **Passo 2: Limpeza de Dados**
- O aplicativo remove colunas desnecessárias, lida com valores ausentes e aplica codificação de rótulos (label encoding) em colunas categóricas.

#### **Passo 3: Visualização de Dados**
- Exibe resumos e gráficos como:
  - Histogramas de atributos numéricos.
  - Gráficos de dispersão para tendências ao longo do tempo.

#### **Passo 4: Treinamento de Modelos**
- Escolha as variáveis alvo e preditoras para treinar:
  - Regressão Linear.
  - Random Forest.
- Visualize métricas de avaliação e gráficos de desempenho.

#### **Passo 5: Fazer Previsões**
- Forneça valores de entrada ao modelo treinado para obter previsões.
- Visualize os resultados e gráficos de importância das variáveis.

---

### Estrutura do Projeto

```plaintext
├── app.py                 # Aplicação principal do Flask
├── templates/             # Templates HTML para as páginas web
│   ├── index.html         # Página de upload
│   ├── visualize.html     # Página de visualização de dados
│   ├── train_model.html   # Página de treinamento de modelo
│   ├── predictions.html   # Página de previsões
│   ├── select_car_model.html  # Página de seleção de modelos de carro
├── uploads/               # Pasta para os arquivos CSV enviados
├── models/                # Pasta para salvar modelos treinados
├── static/                # Pasta para gráficos gerados e arquivos estáticos
├── README.md              # Documentação do projeto
```

---

### Notas Importantes

1. **Validação de Arquivos**:
   - Apenas arquivos `.csv` são permitidos.
2. **Colunas Pré-definidas**:
   - O aplicativo espera colunas específicas (ex.: `avg_price_brl`, `date`) para um funcionamento adequado.
3. **Salvamento de Gráficos**:
   - Gráficos são salvos no diretório `static` e exibidos nas páginas correspondentes.

---

## Melhorias Futuras

1. Suporte a modelos adicionais de aprendizado de máquina.
2. Exportação de modelos treinados em múltiplos formatos.
3. Interface aprimorada para seleção de variáveis e preprocessamento.

---

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

--- 

### Alterações Realizadas
- Adição da biblioteca **Kaleido** como dependência para exportação de gráficos do Plotly.
