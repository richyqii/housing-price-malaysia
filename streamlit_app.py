import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
    page_title='Malaysia Housing Price ML Models',
    page_icon=':house:',
    layout='wide'
)

st.title('🏠 Malaysia Housing Price Prediction - ML Models Comparison')

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

st.markdown('---')

# =========================
# Prepare data for modeling
# =========================
@st.cache_data
def prepare_model_data(data):
    X = data.drop(columns=['Median_Price', 'Price_Category'])
    y = data['Price_Category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_model_data(data)

# =========================
# Helper function to create preprocessor
# =========================
def create_preprocessor():
    categorical_features = ['Township', 'Area', 'State', 'Tenure', 'Type']
    numerical_features = ['Median_PSF', 'Transactions']
    
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
    
    return preprocessor

# =========================
# Train Decision Tree Model
# =========================
@st.cache_data
def train_decision_tree(X_train, X_test, y_train, y_test):
    preprocessor = create_preprocessor()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('dt', DecisionTreeClassifier(random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    return model, y_train_pred, y_test_pred, y_test_proba

# =========================
# Train ANN Model
# =========================
@st.cache_data
def train_ann(X_train, X_test, y_train, y_test):
    preprocessor = create_preprocessor()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('ann', MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=300,
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    return model, y_train_pred, y_test_pred, y_test_proba

# =========================
# Train SVM Model with Hyperparameter Tuning
# =========================
@st.cache_data
def train_svm(X_train, X_test, y_train, y_test):
    preprocessor = create_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    param_grid = [
        {
            'svm__kernel': ['rbf'],
            'svm__C': [0.1, 1, 5, 10],
            'svm__gamma': ['scale', 0.1, 0.01, 0.001]
        },
        {
            'svm__kernel': ['linear'],
            'svm__C': [0.1, 1, 5, 10]
        },
        {
            'svm__kernel': ['poly'],
            'svm__C': [0.1, 1, 5],
            'svm__degree': [2, 3],
            'svm__gamma': ['scale', 0.01]
        }
    ]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    
    return best_model, y_train_pred, y_test_pred, y_test_proba, grid.best_params_, grid.best_score_

# Train all models
dt_model, dt_train_pred, dt_test_pred, dt_test_proba = train_decision_tree(X_train, X_test, y_train, y_test)
ann_model, ann_train_pred, ann_test_pred, ann_test_proba = train_ann(X_train, X_test, y_train, y_test)
svm_model, svm_train_pred, svm_test_pred, svm_test_proba, svm_best_params, svm_best_score = train_svm(X_train, X_test, y_train, y_test)

# =========================
# Create tabs for each model
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["🌳 Decision Tree", "🧠 Artificial Neural Network", "🤖 Support Vector Machine", "🔍 Price Filter"])

labels = ['Low', 'Medium', 'High']

# ============================================
# DECISION TREE TAB
# ============================================
with tab1:
    st.header("Decision Tree Classifier")
    
    # Performance Metrics
    st.subheader("Model Performance")
    dt_train_acc = accuracy_score(y_train, dt_train_pred)
    dt_test_acc = accuracy_score(y_test, dt_test_pred)
    dt_total_acc = accuracy_score(
        pd.concat([y_train, y_test]),
        np.concatenate([dt_train_pred, dt_test_pred])
    )
    dt_mse = mean_squared_error(y_test, dt_test_pred)
    dt_rmse = np.sqrt(dt_mse)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric('Total Accuracy', f'{dt_total_acc:.4f}')
    with col2:
        st.metric('Train Accuracy', f'{dt_train_acc:.4f}')
    with col3:
        st.metric('Test Accuracy', f'{dt_test_acc:.4f}')
    with col4:
        st.metric('MSE', f'{dt_mse:.4f}')
    with col5:
        st.metric('RMSE', f'{dt_rmse:.4f}')
    
    st.markdown('---')
    
    # Classification Report
    st.subheader('Classification Report')
    dt_class_report = pd.DataFrame(
        classification_report(
            y_test, dt_test_pred,
            target_names=labels,
            output_dict=True
        )
    ).transpose()
    st.dataframe(dt_class_report, use_container_width=True)
    
    # Confusion Matrix
    st.subheader('Confusion Matrix')
    dt_cm = confusion_matrix(y_test, dt_test_pred)
    dt_cm_df = pd.DataFrame(
        dt_cm,
        index=['Actual Low', 'Actual Medium', 'Actual High'],
        columns=['Pred Low', 'Pred Medium', 'Pred High']
    )
    st.dataframe(dt_cm_df, use_container_width=True)
    
    # Confusion Report Details
    st.subheader('Confusion Report Details')
    dt_confusion_details = []
    for i, label in enumerate(labels):
        TP = dt_cm[i, i]
        FN = dt_cm[i, :].sum() - TP
        FP = dt_cm[:, i].sum() - TP
        TN = dt_cm.sum() - (TP + FP + FN)
        dt_confusion_details.append([label, TP, FP, FN, TN])
    
    dt_confusion_report = pd.DataFrame(
        dt_confusion_details,
        columns=['Class', 'TP', 'FP', 'FN', 'TN']
    )
    st.dataframe(dt_confusion_report, use_container_width=True)
    
    st.markdown('---')
    
    # Visualizations
    st.header('Visualizations')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            dt_class_report.iloc[:-3, :-1],
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            ax=ax
        )
        ax.set_title("Classification Report Heatmap")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Classes")
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            dt_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    
    # Predicted Class Distribution
    st.subheader('Predicted Class Distribution')
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(dt_test_pred, return_counts=True)
    bars = ax.bar(labels, counts, color=['lightcoral', 'lightyellow', 'lightgreen'])
    ax.set_title("Distribution of Predicted Price Categories")
    ax.set_xlabel("Price Category")
    ax.set_ylabel("Count")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'{int(height)}',
                ha='center', va='center', fontweight='bold', color='black', fontsize=12)
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader('ROC Curve Analysis')
    dt_y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    dt_fpr, dt_tpr, dt_roc_auc = {}, {}, {}
    
    for i in range(3):
        dt_fpr[i], dt_tpr[i], _ = roc_curve(dt_y_test_bin[:, i], dt_test_proba[:, i])
        dt_roc_auc[i] = auc(dt_fpr[i], dt_tpr[i])
    
    dt_macro_roc_auc = roc_auc_score(dt_y_test_bin, dt_test_proba, multi_class='ovr', average='macro')
    
    auc_col1, auc_col2, auc_col3, auc_col4 = st.columns(4)
    with auc_col1:
        st.metric('AUC - Low', f'{dt_roc_auc[0]:.4f}')
    with auc_col2:
        st.metric('AUC - Medium', f'{dt_roc_auc[1]:.4f}')
    with auc_col3:
        st.metric('AUC - High', f'{dt_roc_auc[2]:.4f}')
    with auc_col4:
        st.metric('Macro Avg AUC', f'{dt_macro_roc_auc:.4f}')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'green', 'red']
    for i, color in zip(range(3), colors):
        ax.plot(
            dt_fpr[i], dt_tpr[i], color=color, lw=2,
            label=f'ROC curve of class {labels[i]} (AUC = {dt_roc_auc[i]:.2f})'
        )
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curve (Decision Tree)')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# ============================================
# ANN TAB
# ============================================
with tab2:
    st.header("Artificial Neural Network (ANN) Classifier")
    
    # Performance Metrics
    st.subheader("Model Performance")
    ann_train_acc = accuracy_score(y_train, ann_train_pred)
    ann_test_acc = accuracy_score(y_test, ann_test_pred)
    ann_total_acc = accuracy_score(
        pd.concat([y_train, y_test]),
        np.concatenate([ann_train_pred, ann_test_pred])
    )
    ann_mse = mean_squared_error(y_test, ann_test_pred)
    ann_rmse = np.sqrt(ann_mse)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric('Total Accuracy', f'{ann_total_acc:.4f}')
    with col2:
        st.metric('Train Accuracy', f'{ann_train_acc:.4f}')
    with col3:
        st.metric('Test Accuracy', f'{ann_test_acc:.4f}')
    with col4:
        st.metric('MSE', f'{ann_mse:.4f}')
    with col5:
        st.metric('RMSE', f'{ann_rmse:.4f}')
    
    st.markdown('---')
    
    # Classification Report
    st.subheader('Classification Report')
    ann_class_report = pd.DataFrame(
        classification_report(
            y_test, ann_test_pred,
            target_names=labels,
            output_dict=True
        )
    ).transpose()
    st.dataframe(ann_class_report, use_container_width=True)
    
    # Confusion Matrix
    st.subheader('Confusion Matrix')
    ann_cm = confusion_matrix(y_test, ann_test_pred)
    ann_cm_df = pd.DataFrame(
        ann_cm,
        index=['Actual Low', 'Actual Medium', 'Actual High'],
        columns=['Pred Low', 'Pred Medium', 'Pred High']
    )
    st.dataframe(ann_cm_df, use_container_width=True)
    
    # Confusion Report Details
    st.subheader('Confusion Report Details')
    ann_confusion_details = []
    for i, label in enumerate(labels):
        TP = ann_cm[i, i]
        FN = ann_cm[i, :].sum() - TP
        FP = ann_cm[:, i].sum() - TP
        TN = ann_cm.sum() - (TP + FP + FN)
        ann_confusion_details.append([label, TP, FP, FN, TN])
    
    ann_confusion_report = pd.DataFrame(
        ann_confusion_details,
        columns=['Class', 'TP', 'FP', 'FN', 'TN']
    )
    st.dataframe(ann_confusion_report, use_container_width=True)
    
    st.markdown('---')
    
    # Visualizations
    st.header('Visualizations')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            ann_class_report.iloc[:-3, :-1],
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            ax=ax
        )
        ax.set_title("Classification Report Heatmap (ANN)")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Classes")
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            ann_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (ANN)")
        st.pyplot(fig)
    
    # Confusion Detail Heatmap
    st.subheader('Confusion Report Heatmap')
    fig, ax = plt.subplots(figsize=(7, 5))
    confusion_heatmap_data = ann_confusion_report.set_index('Class')
    sns.heatmap(
        confusion_heatmap_data,
        annot=True,
        fmt='d',
        cmap='Oranges',
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Confusion Report Heatmap (ANN)")
    ax.set_xlabel("Metrics (TP, FP, FN, TN)")
    ax.set_ylabel("Classes")
    st.pyplot(fig)
    
    # Predicted Class Distribution
    st.subheader('Predicted Class Distribution')
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(ann_test_pred, return_counts=True)
    bars = ax.bar(labels, counts, color=['lightcoral', 'lightyellow', 'lightgreen'])
    ax.set_title("Distribution of Predicted Price Categories (ANN)")
    ax.set_xlabel("Price Category")
    ax.set_ylabel("Count")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'{int(height)}',
                ha='center', va='center', fontweight='bold', color='black', fontsize=12)
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader('ROC Curve Analysis')
    ann_y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    ann_fpr, ann_tpr, ann_roc_auc = {}, {}, {}
    
    for i in range(3):
        ann_fpr[i], ann_tpr[i], _ = roc_curve(ann_y_test_bin[:, i], ann_test_proba[:, i])
        ann_roc_auc[i] = auc(ann_fpr[i], ann_tpr[i])
    
    ann_macro_roc_auc = roc_auc_score(ann_y_test_bin, ann_test_proba, multi_class='ovr', average='macro')
    
    auc_col1, auc_col2, auc_col3, auc_col4 = st.columns(4)
    with auc_col1:
        st.metric('AUC - Low', f'{ann_roc_auc[0]:.4f}')
    with auc_col2:
        st.metric('AUC - Medium', f'{ann_roc_auc[1]:.4f}')
    with auc_col3:
        st.metric('AUC - High', f'{ann_roc_auc[2]:.4f}')
    with auc_col4:
        st.metric('Macro Avg AUC', f'{ann_macro_roc_auc:.4f}')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'green', 'red']
    for i, color in zip(range(3), colors):
        ax.plot(
            ann_fpr[i], ann_tpr[i], color=color, lw=2,
            label=f'ROC curve of class {labels[i]} (AUC = {ann_roc_auc[i]:.2f})'
        )
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curve (ANN)')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# ============================================
# SVM TAB
# ============================================
with tab3:
    st.header("Support Vector Machine (SVM) - Tuned")
    
    # Best Parameters
    st.subheader("Best Hyperparameters")
    st.write(f"**Best CV Accuracy:** {svm_best_score:.4f}")
    st.write("**Best Parameters:**")
    for param, value in svm_best_params.items():
        st.write(f"- {param}: {value}")
    
    st.markdown('---')
    
    # Performance Metrics
    st.subheader("Model Performance")
    svm_train_acc = accuracy_score(y_train, svm_train_pred)
    svm_test_acc = accuracy_score(y_test, svm_test_pred)
    svm_total_acc = accuracy_score(
        pd.concat([y_train, y_test]),
        np.concatenate([svm_train_pred, svm_test_pred])
    )
    svm_mse = mean_squared_error(y_test, svm_test_pred)
    svm_rmse = np.sqrt(svm_mse)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric('Total Accuracy', f'{svm_total_acc:.4f}')
    with col2:
        st.metric('Train Accuracy', f'{svm_train_acc:.4f}')
    with col3:
        st.metric('Test Accuracy', f'{svm_test_acc:.4f}')
    with col4:
        st.metric('MSE', f'{svm_mse:.4f}')
    with col5:
        st.metric('RMSE', f'{svm_rmse:.4f}')
    
    st.markdown('---')
    
    # Classification Report
    st.subheader('Classification Report')
    svm_class_report = pd.DataFrame(
        classification_report(
            y_test, svm_test_pred,
            target_names=labels,
            output_dict=True
        )
    ).transpose()
    st.dataframe(svm_class_report, use_container_width=True)
    
    # Confusion Matrix
    st.subheader('Confusion Matrix')
    svm_cm = confusion_matrix(y_test, svm_test_pred)
    svm_cm_df = pd.DataFrame(
        svm_cm,
        index=['Actual Low', 'Actual Medium', 'Actual High'],
        columns=['Pred Low', 'Pred Medium', 'Pred High']
    )
    st.dataframe(svm_cm_df, use_container_width=True)
    
    # Confusion Report Details
    st.subheader('Confusion Report Details')
    svm_confusion_details = []
    for i, label in enumerate(labels):
        TP = svm_cm[i, i]
        FN = svm_cm[i, :].sum() - TP
        FP = svm_cm[:, i].sum() - TP
        TN = svm_cm.sum() - (TP + FP + FN)
        svm_confusion_details.append([label, TP, FP, FN, TN])
    
    svm_confusion_report = pd.DataFrame(
        svm_confusion_details,
        columns=['Class', 'TP', 'FP', 'FN', 'TN']
    )
    st.dataframe(svm_confusion_report, use_container_width=True)
    
    st.markdown('---')
    
    # Visualizations
    st.header('Visualizations')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            svm_class_report.iloc[:-3, :-1],
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            ax=ax
        )
        ax.set_title("Classification Report Heatmap (Tuned SVM)")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Classes")
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            svm_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Tuned SVM)")
        st.pyplot(fig)
    
    # Confusion Detail Heatmap
    st.subheader('Confusion Report Heatmap')
    fig, ax = plt.subplots(figsize=(7, 5))
    confusion_heatmap_data = svm_confusion_report.set_index('Class')
    sns.heatmap(
        confusion_heatmap_data,
        annot=True,
        fmt='d',
        cmap='Oranges',
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Confusion Report Heatmap (Tuned SVM)")
    ax.set_xlabel("Metrics (TP, FP, FN, TN)")
    ax.set_ylabel("Classes")
    st.pyplot(fig)
    
    # Predicted Class Distribution
    st.subheader('Predicted Class Distribution')
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(svm_test_pred, return_counts=True)
    bars = ax.bar(labels, counts, color=['lightcoral', 'lightyellow', 'lightgreen'])
    ax.set_title("Distribution of Predicted Price Categories (Tuned SVM)")
    ax.set_xlabel("Price Category")
    ax.set_ylabel("Count")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'{int(height)}',
                ha='center', va='center', fontweight='bold', color='black', fontsize=12)
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader('ROC Curve Analysis')
    svm_y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    svm_fpr, svm_tpr, svm_roc_auc = {}, {}, {}
    
    for i in range(3):
        svm_fpr[i], svm_tpr[i], _ = roc_curve(svm_y_test_bin[:, i], svm_test_proba[:, i])
        svm_roc_auc[i] = auc(svm_fpr[i], svm_tpr[i])
    
    svm_macro_roc_auc = roc_auc_score(svm_y_test_bin, svm_test_proba, multi_class='ovr', average='macro')
    
    auc_col1, auc_col2, auc_col3, auc_col4 = st.columns(4)
    with auc_col1:
        st.metric('AUC - Low', f'{svm_roc_auc[0]:.4f}')
    with auc_col2:
        st.metric('AUC - Medium', f'{svm_roc_auc[1]:.4f}')
    with auc_col3:
        st.metric('AUC - High', f'{svm_roc_auc[2]:.4f}')
    with auc_col4:
        st.metric('Macro Avg AUC', f'{svm_macro_roc_auc:.4f}')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'green', 'red']
    for i, color in zip(range(3), colors):
        ax.plot(
            svm_fpr[i], svm_tpr[i], color=color, lw=2,
            label=f'ROC curve of class {labels[i]} (AUC = {svm_roc_auc[i]:.2f})'
        )
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curve (Tuned SVM)')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# ============================================
# PRICE FILTER & PREDICTION TAB
# ============================================
with tab4:
    st.header('🔍 Filter & Predict Housing Price')
    st.markdown('**Select property features below to get predictions**')
    st.markdown('---')
    
    # Step 1: User selects properties
    st.subheader('Step 1: Select Property Features')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_township = st.selectbox('Township', sorted(data['Township'].unique()), key='township_select')
    
    with col2:
        selected_area = st.selectbox('Area', sorted(data['Area'].unique()), key='area_select')
    
    with col3:
        selected_state = st.selectbox('State', sorted(data['State'].unique()), key='state_select')
    
    col4, col5 = st.columns(2)
    
    with col4:
        selected_tenure = st.selectbox('Tenure', sorted(data['Tenure'].unique()), key='tenure_select')
    
    with col5:
        selected_type = st.selectbox('Type', sorted(data['Type'].unique()), key='type_select')
    
    st.markdown('---')
    
    # Step 2: Create input dataframe for prediction
    st.subheader('Step 2: Predictions')
    
    # Get statistics for numerical features
    median_psf_mean = data['Median_PSF'].median()
    transactions_mean = data['Transactions'].median()
    
    # Create input data for prediction
    input_data = pd.DataFrame({
        'Township': [selected_township],
        'Area': [selected_area],
        'State': [selected_state],
        'Tenure': [selected_tenure],
        'Type': [selected_type],
        'Median_PSF': [median_psf_mean],
        'Transactions': [transactions_mean]
    })
    
    # Make predictions using all three models
    dt_pred_category = dt_model.predict(input_data)[0]
    ann_pred_category = ann_model.predict(input_data)[0]
    svm_pred_category = svm_model.predict(input_data)[0]
    
    # Get probability scores for ANN
    ann_pred_proba = ann_model.predict_proba(input_data)[0]
    
    category_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    dt_category_label = category_map[dt_pred_category]
    ann_category_label = category_map[ann_pred_category]
    svm_category_label = category_map[svm_pred_category]
    
    # Get actual median price for similar properties
    similar_properties = data[
        (data['Township'] == selected_township) &
        (data['Area'] == selected_area) &
        (data['State'] == selected_state) &
        (data['Tenure'] == selected_tenure) &
        (data['Type'] == selected_type)
    ]
    
    if len(similar_properties) > 0:
        predicted_median_price = similar_properties['Median_Price'].mean()
        predicted_median_psf = similar_properties['Median_PSF'].mean()
        predicted_transactions = similar_properties['Transactions'].mean()
        st.success(f'✅ Found {len(similar_properties)} matching property records')
    else:
        # If no exact match, use category statistics
        low = data['Median_Price'].quantile(0.33)
        high = data['Median_Price'].quantile(0.66)
        
        if ann_pred_category == 0:  # Low
            predicted_median_price = data[data['Price_Category'] == 0]['Median_Price'].mean()
        elif ann_pred_category == 1:  # Medium
            predicted_median_price = data[data['Price_Category'] == 1]['Median_Price'].mean()
        else:  # High
            predicted_median_price = data[data['Price_Category'] == 2]['Median_Price'].mean()
        
        predicted_median_psf = data['Median_PSF'].mean()
        predicted_transactions = data['Transactions'].mean()
        st.info('📊 No exact match found. Using category statistics for predictions.')
    
    st.markdown('---')
    
    # Display predictions in metric cards and details
    st.subheader('📈 Predicted Values')
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        st.metric('Median Price', f"RM {predicted_median_price:,.0f}")
    
    with col_metrics2:
        st.metric('Median PSF', f"RM {predicted_median_psf:.2f}")
    
    with col_metrics3:
        st.metric('Transactions', f"{predicted_transactions:.0f}")
    
    st.markdown('---')
    
    # Detailed predictions from all models
    st.subheader('🤖 Model Predictions - Price Category')
    
    # Determine best model based on test accuracy
    accuracies = {
        'Decision Tree': dt_test_acc,
        'ANN': ann_test_acc,
        'SVM': svm_test_acc
    }
    
    best_model_name = max(accuracies, key=accuracies.get)
    best_model_accuracy = accuracies[best_model_name]
    
    # Get best model's prediction
    if best_model_name == 'Decision Tree':
        best_pred_category = dt_pred_category
        best_category_label = dt_category_label
    elif best_model_name == 'ANN':
        best_pred_category = ann_pred_category
        best_category_label = ann_category_label
    else:  # SVM
        best_pred_category = svm_pred_category
        best_category_label = svm_category_label
    
    # Display best model prediction prominently
    col_best = st.columns(1)[0]
    with col_best:
        st.markdown(f'### 🏆 Best Model: **{best_model_name}** (Test Accuracy: {best_model_accuracy:.2%})')
        st.metric('Predicted Price Category', best_category_label)
    
    st.markdown('---')
    
    # Show all models for comparison
    st.subheader('📊 All Models Comparison')
    
    col_pred1, col_pred2, col_pred3 = st.columns(3)
    
    with col_pred1:
        st.markdown(f'**Decision Tree** {"✓ BEST" if best_model_name == "Decision Tree" else ""}')
        st.write(f"Prediction: **{dt_category_label}**")
        st.write(f"Accuracy: {dt_test_acc:.2%}")
    
    with col_pred2:
        st.markdown(f'**ANN** {"✓ BEST" if best_model_name == "ANN" else ""}')
        st.write(f"Prediction: **{ann_category_label}**")
        st.write(f"Accuracy: {ann_test_acc:.2%}")
        if best_model_name == 'ANN':
            st.write(f"**Confidence:** {ann_pred_proba[best_pred_category]:.2%}")
    
    with col_pred3:
        st.markdown(f'**SVM** {"✓ BEST" if best_model_name == "SVM" else ""}')
        st.write(f"Prediction: **{svm_category_label}**")
        st.write(f"Accuracy: {svm_test_acc:.2%}")
    
    st.markdown('---')
    
    # Display input features table
    st.subheader('📋 Selected Features Summary')
    
    features_summary = pd.DataFrame({
        'Feature': ['Township', 'Area', 'State', 'Tenure', 'Type'],
        'Selected Value': [selected_township, selected_area, selected_state, selected_tenure, selected_type]
    })
    
    st.dataframe(features_summary, use_container_width=True, hide_index=True)
    
    st.markdown('---')
    
    # Market Visualizations using Best Model's Prediction
    st.subheader(f'📊 Market Visualizations ({best_model_name} - Price Category: {best_category_label})')
    
    col_viz1, col_viz2 = st.columns(2)
    
    # Visualization 1: Price Category Distribution across market
    with col_viz1:
        st.markdown('**Market Price Category Distribution**')
        
        category_counts = data['Price_Category'].value_counts().sort_index()
        category_labels_viz = ['Low', 'Medium', 'High']
        
        # Highlight the predicted category
        colors = []
        for i in range(3):
            if i == best_pred_category:
                colors.append('#FF6B6B')  # Red for predicted
            else:
                colors.append('#B8B8B8')  # Gray for others
        
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(category_labels_viz, [category_counts.get(i, 0) for i in range(3)], 
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                   f'{int(height)}',
                   ha='center', va='center', fontweight='bold', color='white', fontsize=11)
        
        ax.set_title(f'Your Property Predicted: {best_category_label}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Properties')
        ax.set_xlabel('Price Category')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    # Visualization 2: Median Price by State (highlight selected state)
    with col_viz2:
        st.markdown('**Median Price by State**')
        
        # Filter to predicted price category
        category_data_viz = data[data['Price_Category'] == best_pred_category]
        state_prices = category_data_viz.groupby('State')['Median_Price'].median().sort_values(ascending=False).head(10)
        
        # Highlight selected state
        colors_state = []
        for state in state_prices.index:
            if state == selected_state:
                colors_state.append('#FF6B6B')  # Red for selected
            else:
                colors_state.append('#4472C4')  # Blue for others
        
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(range(len(state_prices)), state_prices.values, color=colors_state, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' RM {int(width):,}',
                   ha='left', va='center', fontweight='bold', fontsize=9)
        
        ax.set_yticks(range(len(state_prices)))
        ax.set_yticklabels(state_prices.index)
        ax.set_xlabel('Median Price (RM)')
        ax.set_title(f'{best_category_label} Category - Top 10 States', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown('---')
    
    # More visualizations
    col_viz3, col_viz4 = st.columns(2)
    
    # Visualization 3: Property Type Distribution
    with col_viz3:
        st.markdown('**Property Type Distribution**')
        
        category_data_viz = data[data['Price_Category'] == best_pred_category]
        type_counts = category_data_viz['Type'].value_counts()
        
        # Highlight selected type
        colors_type = []
        for ptype in type_counts.index:
            if ptype == selected_type:
                colors_type.append('#FF6B6B')
            else:
                colors_type.append('#70AD47')
        
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(range(len(type_counts)), type_counts.values, color=colors_type, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                   f'{int(height)}',
                   ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        
        ax.set_xticks(range(len(type_counts)))
        ax.set_xticklabels(type_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(f'{best_category_label} Category - Property Types', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    # Visualization 4: Tenure Distribution
    with col_viz4:
        st.markdown('**Tenure Distribution**')
        
        category_data_viz = data[data['Price_Category'] == best_pred_category]
        tenure_counts = category_data_viz['Tenure'].value_counts()
        
        # Highlight selected tenure
        colors_tenure = []
        for tenure in tenure_counts.index:
            if tenure == selected_tenure:
                colors_tenure.append('#FF6B6B')
            else:
                colors_tenure.append('#FFA500')
        
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(range(len(tenure_counts)), tenure_counts.values, color=colors_tenure, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                   f'{int(height)}',
                   ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        
        ax.set_xticks(range(len(tenure_counts)))
        ax.set_xticklabels(tenure_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(f'{best_category_label} Category - Tenure Types', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown('---')
    
    # Show similar properties if found
    if len(similar_properties) > 0:
        st.subheader('📊 Similar Properties in Dataset')
        st.dataframe(
            similar_properties[['Township', 'Area', 'State', 'Tenure', 'Type', 'Median_Price', 'Median_PSF', 'Transactions', 'Price_Category']].head(5),
            use_container_width=True
        )

