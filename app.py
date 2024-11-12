from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Função para verificar extensão do arquivo
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Rota para exibir o formulário de upload
@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Verificar se há um arquivo
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # Verificar se o arquivo é permitido
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Ler o CSV com pandas
            df = pd.read_csv(filepath)
            columns = df.columns.tolist()  # Obter lista de colunas
            
            # Redirecionar para a página de filtro, passando as colunas
            return render_template('filter.html', columns=columns, filename=filename)
    
    return render_template('index.html')

# Rota para exibir as colunas e permitir filtragem
@app.route("/filter", methods=['POST'])
def filter_data():
    filename = request.form.get('filename')
    selected_columns = request.form.getlist('columns')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Ler o CSV com pandas e filtrar as colunas selecionadas
    df = pd.read_csv(filepath)
    filtered_df = df[selected_columns]
    
    # Exibir os dados filtrados como HTML
    return render_template('filtered_data.html', tables=[filtered_df.to_html(classes='data', header="true")])

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
