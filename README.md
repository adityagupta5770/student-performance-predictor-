# Student Performance Predictor 🎓

A beginner-friendly Machine Learning project that predicts expected student exam marks using study hours, attendance, previous marks, and sleep hours.

## Features
- Interactive Streamlit web app
- Linear Regression model (scikit-learn)
- Sample dataset included
- Model evaluation using MAE and R²
- Data preview and scatter plot

## Tech stack
Python, Pandas, scikit-learn, Streamlit, Matplotlib.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   ```

   Windows:
   ```bash
   venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

The app opens in your browser, usually at `http://localhost:8501`.

## How it works
The app loads `data/student_data.csv`, splits the records into training and test sets, trains a Linear Regression model, and predicts marks from the values entered by the user.

## Dataset note
The included dataset is synthetic and generated for demonstration and learning. It is not real student data. Predictions are illustrative and should not be used for academic decisions.

## Upload to GitHub

Create a new empty repository on GitHub (for example, `student-performance-predictor`). Then run these commands from this project folder:

```bash
git init
git add .
git commit -m "Initial commit: Student Performance Predictor"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-predictor.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## Future improvements
- Add more student-related input features
- Compare Linear Regression with Random Forest
- Use a verified real-world dataset
- Deploy the app using Streamlit Community Cloud
