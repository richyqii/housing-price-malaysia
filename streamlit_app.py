import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

# Set the title and favicon
st.set_page_config(
    page_title='Malaysia Housing Price Decision Tree',
    page_icon=':house:',
    layout='wide'
)

st.title('🏠 Malaysia Housing Price Prediction - Decision Tree Model')

# =========================
# Load and prepare data
# =========================
@st.cache_data
def load_and_prepare_data():
    DATA_FILENAME = Path(__file__).parent / 'data/malaysia_house_price_data_2025.csv'
    data = pd.read_csv(DATA_FILENAME)
    
    # Convert Median_Price into 3 categories
    low = data['Median_Price'].quantile(0.33)
    high = data['Median_Price'].quantile(0.66)
    
    def categorize(price):
        if price < low:
            return 0   # Low
        elif price < high:
            return 1   # Medium
        else:
            return 2   # High
    
    data['Price_Category'] = data['Median_Price'].apply(categorize)
    return data, low, high

data, low, high = load_and_prepare_data()

# Display dataset info
st.header('📊 Dataset Overview')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Total Records', len(data))
with col2:
    st.metric('Features', len(data.columns))
with col3:
    st.metric('Price Range', f"RM {data['Median_Price'].min():.0f} - RM {data['Median_Price'].max():.0f}")

# =========================
# Build and train model
# =========================
@st.cache_data
def train_model(data):
    # Define X and y
    X = data.drop(columns=['Median_Price', 'Price_Category'])
    y = data['Price_Category']
    
    # Define feature types
    categorical_features = ['Township', 'Area', 'State', 'Tenure', 'Type']
    numerical_features = ['Median_PSF', 'Transactions']
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('dt', DecisionTreeClassifier(random_state=42))
    ])
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, y_test_proba

model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, y_test_proba = train_model(data)

# =========================
# Model Performance Metrics
# =========================
st.header('📈 Model Performance')

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
total_acc = accuracy_score(
    pd.concat([y_train, y_test]),
    np.concatenate([y_train_pred, y_test_pred])
)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric('Total Accuracy', f'{total_acc:.4f}')
with col2:
    st.metric('Train Accuracy', f'{train_acc:.4f}')
with col3:
    st.metric('Test Accuracy', f'{test_acc:.4f}')
with col4:
    st.metric('MSE', f'{mse:.4f}')
with col5:
    st.metric('RMSE', f'{rmse:.4f}')

st.markdown('---')

# =========================
# Classification Report
# =========================
st.subheader('Classification Report')
class_report_df = pd.DataFrame(
    classification_report(
        y_test, y_test_pred,
        target_names=['Low', 'Medium', 'High'],
        output_dict=True
    )
).transpose()
st.dataframe(class_report_df, use_container_width=True)

# =========================
# Confusion Matrix
# =========================
st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(
    cm,
    index=['Actual Low', 'Actual Medium', 'Actual High'],
    columns=['Pred Low', 'Pred Medium', 'Pred High']
)
st.dataframe(cm_df, use_container_width=True)

# Confusion Report (TP, FP, FN, TN)
st.subheader('Confusion Report Details')
labels = ['Low', 'Medium', 'High']
confusion_details = []

for i, label in enumerate(labels):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    confusion_details.append([label, TP, FP, FN, TN])

confusion_report_df = pd.DataFrame(
    confusion_details,
    columns=['Class', 'TP', 'FP', 'FN', 'TN']
)
st.dataframe(confusion_report_df, use_container_width=True)

st.markdown('---')

# =========================
# Visualizations
# =========================
st.header('📊 Visualizations')

col1, col2 = st.columns(2)

# Classification Report Heatmap
with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        class_report_df.iloc[:-3, :-1],
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Classification Report Heatmap")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Classes")
    st.pyplot(fig)

# Confusion Matrix Heatmap
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Low', 'Medium', 'High'],
        yticklabels=['Low', 'Medium', 'High'],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Predicted Class Distribution
st.subheader('Predicted Class Distribution')
fig, ax = plt.subplots(figsize=(8, 4))
unique, counts = np.unique(y_test_pred, return_counts=True)
ax.bar(['Low', 'Medium', 'High'], counts, color=['lightcoral', 'lightyellow', 'lightgreen'])
ax.set_title("Distribution of Predicted Price Categories")
ax.set_xlabel("Price Category")
ax.set_ylabel("Count")
st.pyplot(fig)

# =========================
# ROC Curve and AUC
# =========================
st.subheader('ROC Curve Analysis')
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

macro_roc_auc = roc_auc_score(y_test_bin, y_test_proba, multi_class='ovr', average='macro')

# Display AUC scores
auc_col1, auc_col2, auc_col3, auc_col4 = st.columns(4)
with auc_col1:
    st.metric('AUC - Low', f'{roc_auc[0]:.4f}')
with auc_col2:
    st.metric('AUC - Medium', f'{roc_auc[1]:.4f}')
with auc_col3:
    st.metric('AUC - High', f'{roc_auc[2]:.4f}')
with auc_col4:
    st.metric('Macro Avg AUC', f'{macro_roc_auc:.4f}')

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['blue', 'green', 'red']

for i, color in zip(range(n_classes), colors):
    ax.plot(
        fpr[i], tpr[i], color=color, lw=2,
        label=f'ROC curve of class {labels[i]} (AUC = {roc_auc[i]:.2f})'
    )

ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Multiclass ROC Curve (Decision Tree)')
ax.legend(loc="lower right")
st.pyplot(fig)
