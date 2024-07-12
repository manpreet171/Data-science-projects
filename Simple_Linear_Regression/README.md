# Simple Linear Regression

This project aims to build a Simple Linear Regression model to find the relationship between delivery time and sorting time for a logistics company. The delivery time is the target variable, and sorting time is the predictor variable. The project involves data loading, exploratory data analysis, model building, and evaluation.

## Project Structure

Simple_Linear_Regression/
├── data/
│ ├── delivery_time.csv
│
├── scripts/
│ ├── simple_linear_regression.py
│
├── results/
│
├── .gitignore
├── README.md
└── requirements.txt

markdown


## Steps

### 1. Data Collection and Loading

The dataset is loaded from a CSV file containing two columns: `Delivery Time` and `Sorting Time`.

### 2. Exploratory Data Analysis (EDA)

- Display basic statistics
- Check for missing values
- Plot histograms of the features
- Scatter plot of sorting time vs. delivery time
- Correlation matrix

### 3. Model Building

- Simple Linear Regression
- Log Transformation
- Exponential Transformation
- Polynomial Transformation

### 4. Model Evaluation and Tuning

- Root Mean Squared Error (RMSE)
- Model summary and coefficients

## Libraries Used

- pandas: For data manipulation
- matplotlib: For plotting
- seaborn: For statistical graphics
- numpy: For numerical operations
- statsmodels: For building statistical models
- sklearn: For polynomial transformations and linear regression

## How to Run the Project

1. Clone the Repository
    ```bash
    git clone https://github.com/your-username/Data-science-projects.git
    ```

2. Navigate to the Project Directory
    ```bash
    cd Data-science-projects/Simple_Linear_Regression
    ```

3. Install the Required Packages
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Script
    ```bash
    python scripts/simple_linear_regression.py
    ```

## Results

The results, including the RMSE values and model summaries, will be displayed in the terminal.

## Dependencies

The project requires the following Python packages:
- pandas
- matplotlib
- seaborn
- numpy
- statsmodels
- scikit-learn

Install them using:
```bash
pip install -r requirements.txt