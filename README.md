# AverageCarPrices

1. Acesse a Aplicação
Inicie a aplicação Flask com o comando:
bash
Copiar código
python app.py
Acesse a interface da aplicação no navegador, geralmente disponível em http://127.0.0.1:5000.
2. Faça o Upload do Dataset
Na página principal da aplicação, você deve encontrar uma opção para Fazer Upload de um Arquivo.

Selecione um arquivo .csv que tenha dados estruturados como o exemplo do tema AverageCarPrices:

Exemplo de formato:

objectivec
Copiar código
year_of_reference,month_of_reference,fipe_code,authentication,brand,model,fuel,gear,engine_size,year_model,avg_price_brl,age_years
2022,January,038001-6,vwmrywl5qs,Acura,NSX 3.0,Gasoline,manual,3.0,1995,43779.0,28
Faça o upload do arquivo para que a aplicação carregue os dados. A interface deve exibir uma confirmação de que o arquivo foi carregado corretamente.

3. Visualize as Colunas e Selecione Variáveis para Análise
Após o upload do arquivo, a aplicação deve mostrar as colunas do dataset. Aqui, as colunas importantes para análise e previsão podem incluir:

year_of_reference: ano da referência
engine_size: tamanho do motor
year_model: ano do modelo
avg_price_brl: preço médio em BRL (reais)
age_years: idade do carro em anos
Selecione variáveis específicas para visualizar, como:

Distribuição de Preço por Ano do Modelo: visualiza como os preços variam com o ano do modelo.
Distribuição de Preço por Idade: visualiza o impacto da idade do carro no preço.
A interface deve fornecer opções para gráficos (barras, pizza, etc.) e permitir gerar visualizações baseadas nas colunas selecionadas. Clique nas opções desejadas para ver o gráfico.

4. Configure o Modelo de Machine Learning
A aplicação deve permitir a configuração do modelo. Você pode:

Escolher um tipo de modelo (por exemplo, Regressão Linear para previsão de preço).
Configurar parâmetros, como profundidade da árvore para modelos de árvore de decisão ou o número de vizinhos para K-Nearest Neighbors (KNN).
Escolha uma coluna de target (variável a ser prevista). Para este exemplo:

Target: avg_price_brl (preço médio)
Selecione as Features (variáveis preditivas):

Features recomendadas: year_of_reference, engine_size, year_model, age_years
5. Treine e Teste o Modelo
Clique em "Treinar Modelo". A aplicação deve exibir um indicador de progresso e, ao final, apresentar uma métrica de avaliação, como o erro médio absoluto ou a acurácia, para você verificar a precisão.
Após o treino, faça uma predição de teste:
Insira valores para as features que você usou no treino (por exemplo, insira 2022 para o ano, 3.0 para o tamanho do motor, etc.).
A aplicação deve fornecer uma previsão do valor para avg_price_brl.
6. Teste com um Novo Dataset ou Retreine o Modelo
Para testar a funcionalidade de re-treino, faça o upload de um novo dataset ou atualize o dataset existente.
Retorne à seção de Machine Learning e clique em "Retreinar Modelo" para ver como as previsões se ajustam com os novos dados.
7. Documentação e Verificação Final
Revise as análises e predições para garantir que os gráficos e previsões são consistentes com o dataset.
Consulte a documentação e, se necessário, ajuste os parâmetros para refinar as previsões e visualizar como diferentes modelos e parâmetros afetam o resultado.
Esses passos ajudam a testar todas as funcionalidades principais da interface da aplicação.