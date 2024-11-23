
# Flask Data Processing and Machine Learning App

This project is a Flask-based web application designed for data preprocessing, visualization, and machine learning model training. It supports uploading CSV files, cleaning and transforming data, visualizing insights, and training models for prediction tasks.

---

## Features

1. **File Upload**: Upload CSV files for processing and analysis.
2. **Data Cleaning**: Handles missing values, removes duplicates, and encodes categorical features.
3. **Visualization**: Generates histograms and scatter plots to explore data distributions and trends.
4. **Machine Learning**:
   - Train models like Linear Regression and Random Forest.
   - Evaluate performance using R² scores and visualizations.
5. **Prediction**: Use trained models to make predictions with user-provided inputs.
6. **Model Management**: Save and load trained models for reuse.

---

## Technologies Used

- **Python Libraries**:
  - `Flask`: Web application framework.
  - `Pandas`: Data manipulation and analysis.
  - `NumPy`: Numerical computing.
  - `Matplotlib` & `Seaborn`: Data visualization.
  - `scikit-learn`: Machine learning models and preprocessing tools.
  - `Joblib`: Model persistence.
- **Frontend**:
  - `HTML` and `CSS`: For rendering web pages and styling.
  - Bootstrap classes: For a responsive UI.
- **Environment**:
  - Python 3.8+ recommended.

---

## Prerequisites

- Python 3.8+
- Install required libraries:
  ```bash
  pip install flask pandas numpy matplotlib seaborn scikit-learn joblib
  ```

---

## Usage

### 1. Running the Application

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Create the necessary folders:
   ```bash
   mkdir uploads static models
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Access the application at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

### 2. Functional Workflow

#### **Step 1: Upload CSV File**
- Navigate to the homepage and upload a CSV file.
- Ensure the file contains relevant data columns for analysis.

#### **Step 2: Data Cleaning**
- The app automatically removes unnecessary columns, handles missing values, and applies label encoding to categorical features.

#### **Step 3: Data Visualization**
- View data summaries and visualizations like:
  - Histograms of numerical features.
  - Scatter plots of trends over time.

#### **Step 4: Model Training**
- Select target variables and features to train:
  - Linear Regression.
  - Random Forest.
- View model evaluation metrics and performance plots.

#### **Step 5: Make Predictions**
- Provide input features to the trained model for predictions.
- View prediction results and visualizations of feature importance.

---

### File Structure

```plaintext
├── app.py                 # Main Flask application
├── templates/             # HTML templates for web pages
│   ├── index.html         # Upload page
│   ├── visualize.html     # Data visualization page
│   ├── train_model.html   # Model training page
│   ├── predictions.html   # Predictions page
│   ├── select_car_model.html  # Model selection page
├── uploads/               # Folder for uploaded CSV files
├── models/                # Folder for saving trained models
├── static/                # Folder for generated plots and static files
├── README.md              # Project documentation
```

---

### Important Notes

1. **File Validation**:
   - Only `.csv` files are allowed for upload.
2. **Predefined Columns**:
   - The app expects specific column names (e.g., `avg_price_brl`, `date`) for proper functionality.
   - Ensure column names align with application expectations.
3. **Graph Saving**:
   - Graphs are saved in the `static` directory and displayed on respective pages.

---

## Future Enhancements

1. Support for additional machine learning models.
2. Export trained models in multiple formats.
3. Enhanced UI for feature selection and preprocessing.

---

## License

This project is licensed under the MIT License.
