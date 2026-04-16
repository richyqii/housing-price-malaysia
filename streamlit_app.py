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
# Train Random Forest Model
# =========================
@st.cache_data
def train_random_forest(X_train, X_test, y_train, y_test):
    preprocessor = create_preprocessor()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=42))
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
rf_model, rf_train_pred, rf_test_pred, rf_test_proba = train_random_forest(X_train, X_test, y_train, y_test)
svm_model, svm_train_pred, svm_test_pred, svm_test_proba, svm_best_params, svm_best_score = train_svm(X_train, X_test, y_train, y_test)

# =========================
# Create tabs for each model
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["🌳 Decision Tree", "🌲 Random Forest", "🤖 Support Vector Machine", "🔍 Price Filter"])

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
    ax.bar(labels, counts, color=['lightcoral', 'lightyellow', 'lightgreen'])
    ax.set_title("Distribution of Predicted Price Categories")
    ax.set_xlabel("Price Category")
    ax.set_ylabel("Count")
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
# RANDOM FOREST TAB
# ============================================
with tab2:
    st.header("Random Forest Classifier")
    
    # Performance Metrics
    st.subheader("Model Performance")
    rf_train_acc = accuracy_score(y_train, rf_train_pred)
    rf_test_acc = accuracy_score(y_test, rf_test_pred)
    rf_total_acc = accuracy_score(
        pd.concat([y_train, y_test]),
        np.concatenate([rf_train_pred, rf_test_pred])
    )
    rf_mse = mean_squared_error(y_test, rf_test_pred)
    rf_rmse = np.sqrt(rf_mse)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric('Total Accuracy', f'{rf_total_acc:.4f}')
    with col2:
        st.metric('Train Accuracy', f'{rf_train_acc:.4f}')
    with col3:
        st.metric('Test Accuracy', f'{rf_test_acc:.4f}')
    with col4:
        st.metric('MSE', f'{rf_mse:.4f}')
    with col5:
        st.metric('RMSE', f'{rf_rmse:.4f}')
    
    st.markdown('---')
    
    # Classification Report
    st.subheader('Classification Report')
    rf_class_report = pd.DataFrame(
        classification_report(
            y_test, rf_test_pred,
            target_names=labels,
            output_dict=True
        )
    ).transpose()
    st.dataframe(rf_class_report, use_container_width=True)
    
    # Confusion Matrix
    st.subheader('Confusion Matrix')
    rf_cm = confusion_matrix(y_test, rf_test_pred)
    rf_cm_df = pd.DataFrame(
        rf_cm,
        index=['Actual Low', 'Actual Medium', 'Actual High'],
        columns=['Pred Low', 'Pred Medium', 'Pred High']
    )
    st.dataframe(rf_cm_df, use_container_width=True)
    
    # Confusion Report Details
    st.subheader('Confusion Report Details')
    rf_confusion_details = []
    for i, label in enumerate(labels):
        TP = rf_cm[i, i]
        FN = rf_cm[i, :].sum() - TP
        FP = rf_cm[:, i].sum() - TP
        TN = rf_cm.sum() - (TP + FP + FN)
        rf_confusion_details.append([label, TP, FP, FN, TN])
    
    rf_confusion_report = pd.DataFrame(
        rf_confusion_details,
        columns=['Class', 'TP', 'FP', 'FN', 'TN']
    )
    st.dataframe(rf_confusion_report, use_container_width=True)
    
    st.markdown('---')
    
    # Visualizations
    st.header('Visualizations')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            rf_class_report.iloc[:-3, :-1],
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            ax=ax
        )
        ax.set_title("Classification Report Heatmap (Random Forest)")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Classes")
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            rf_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Random Forest)")
        st.pyplot(fig)
    
    # Predicted Class Distribution
    st.subheader('Predicted Class Distribution')
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(rf_test_pred, return_counts=True)
    ax.bar(labels, counts, color=['lightcoral', 'lightyellow', 'lightgreen'])
    ax.set_title("Distribution of Predicted Price Categories (Random Forest)")
    ax.set_xlabel("Price Category")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader('ROC Curve Analysis')
    rf_y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    rf_fpr, rf_tpr, rf_roc_auc = {}, {}, {}
    
    for i in range(3):
        rf_fpr[i], rf_tpr[i], _ = roc_curve(rf_y_test_bin[:, i], rf_test_proba[:, i])
        rf_roc_auc[i] = auc(rf_fpr[i], rf_tpr[i])
    
    rf_macro_roc_auc = roc_auc_score(rf_y_test_bin, rf_test_proba, multi_class='ovr', average='macro')
    
    auc_col1, auc_col2, auc_col3, auc_col4 = st.columns(4)
    with auc_col1:
        st.metric('AUC - Low', f'{rf_roc_auc[0]:.4f}')
    with auc_col2:
        st.metric('AUC - Medium', f'{rf_roc_auc[1]:.4f}')
    with auc_col3:
        st.metric('AUC - High', f'{rf_roc_auc[2]:.4f}')
    with auc_col4:
        st.metric('Macro Avg AUC', f'{rf_macro_roc_auc:.4f}')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'green', 'red']
    for i, color in zip(range(3), colors):
        ax.plot(
            rf_fpr[i], rf_tpr[i], color=color, lw=2,
            label=f'ROC curve of class {labels[i]} (AUC = {rf_roc_auc[i]:.2f})'
        )
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curve (Random Forest)')
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
    ax.bar(labels, counts, color=['lightcoral', 'lightyellow', 'lightgreen'])
    ax.set_title("Distribution of Predicted Price Categories (Tuned SVM)")
    ax.set_xlabel("Price Category")
    ax.set_ylabel("Count")
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
    
    st.markdown('**Select township and system will auto-fill other properties based on available data**')
    st.markdown('---')
    
    # Step 1: User selects Township
    townships = sorted(data['Township'].unique())
    selected_township = st.selectbox('Township', townships, key='township_filter')
    
    # Filter data by township
    township_data = data[data['Township'] == selected_township]
    
    # Step 2: Get available Areas for this Township (Auto-fill)
    available_areas = sorted(township_data['Area'].unique())
    selected_area = available_areas[0] if len(available_areas) > 0 else None
    st.info(f'Area: **{selected_area}** (Auto-filled)')
    
    # Filter data by township and area
    township_area_data = township_data[township_data['Area'] == selected_area]
    
    # Step 3: Get available States for this Township+Area
    available_states = sorted(township_area_data['State'].unique())
    
    if len(available_states) == 1:
        selected_state = available_states[0]
        st.info(f'State: **{selected_state}** (Auto-filled)')
    else:
        selected_state = st.selectbox('State', available_states, key='state_filter')
    
    # Filter data by township, area, and state
    township_area_state_data = township_area_data[township_area_data['State'] == selected_state]
    
    # Step 4: Get available Tenures and Types
    available_tenures = sorted(township_area_state_data['Tenure'].unique())
    available_types = sorted(township_area_state_data['Type'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(available_tenures) == 1:
            selected_tenure = available_tenures[0]
            st.info(f'Tenure: **{selected_tenure}** (Auto-filled)')
        else:
            selected_tenure = st.selectbox('Tenure', available_tenures, key='tenure_filter')
    
    with col2:
        if len(available_types) == 1:
            selected_type = available_types[0]
            st.info(f'Type: **{selected_type}** (Auto-filled)')
        else:
            selected_type = st.selectbox('Type', available_types, key='type_filter')
    
    st.markdown('---')
    
    # Filter data based on all selections
    filtered_data = data[
        (data['Township'] == selected_township) &
        (data['Area'] == selected_area) &
        (data['State'] == selected_state) &
        (data['Tenure'] == selected_tenure) &
        (data['Type'] == selected_type)
    ]
    
    # Display results
    if len(filtered_data) > 0:
        st.success(f"✅ Found {len(filtered_data)} matching properties")
        st.markdown('---')
        
        # Get the aggregated values
        median_price = filtered_data['Median_Price'].iloc[0] if len(filtered_data) > 0 else 0
        median_psf = filtered_data['Median_PSF'].iloc[0] if len(filtered_data) > 0 else 0
        transactions = filtered_data['Transactions'].iloc[0] if len(filtered_data) > 0 else 0
        
        # Categorize price
        price_category = 'Low' if median_price < low else ('Medium' if median_price < high else 'High')
        
        # Choose color based on category
        if price_category == 'Low':
            color_indicator = '🟢'
            bg_color = '#90EE90'
        elif price_category == 'Medium':
            color_indicator = '🟡'
            bg_color = '#FFD700'
        else:
            color_indicator = '🔴'
            bg_color = '#FFB6C6'
        
        # Display price category prominently
        st.subheader(f'{color_indicator} Price Category: {price_category}')
        st.markdown(f'**Low Range:** RM 0 - RM {low:,.0f}')
        st.markdown(f'**Medium Range:** RM {low:,.0f} - RM {high:,.0f}')
        st.markdown(f'**High Range:** RM {high:,.0f}+')
        
        st.markdown('---')
        
        # Display metrics
        st.subheader('Property Metrics')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric('Median Price', f'RM {median_price:,.0f}', 
                     delta=f'Category: {price_category}')
        with col2:
            st.metric('Median PSF', f'RM {median_psf:,.2f}')
        with col3:
            st.metric('Total Transactions', f'{int(transactions)}')
        
        st.markdown('---')
        
        # Display visualizations
        st.subheader('📊 Market Visualizations')
        
        # Visualization 1: Price Category Distribution - Highlight Selected
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('**Price Category Distribution (Your Selection Highlighted)**')
            # Get price category breakdown across all market
            category_counts = data['Price_Category'].value_counts().sort_index()
            category_labels = ['Low', 'Medium', 'High']
            
            # Highlight selected category
            colors_dist = []
            for i, label in enumerate(category_labels):
                if (i == 0 and price_category == 'Low') or \
                   (i == 1 and price_category == 'Medium') or \
                   (i == 2 and price_category == 'High'):
                    colors_dist.append('#FF6B6B')  # Bright red for selected
                else:
                    colors_dist.append('#B8B8B8')  # Gray for others
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(category_labels, [category_counts.get(i, 0) for i in [0, 1, 2]], 
                          color=colors_dist, edgecolor='black', linewidth=2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'Your Property is in "{price_category}" Category', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Properties')
            ax.set_xlabel('Price Category')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown('**Median PSF by Type (Same Category - Highlighted)**')
            
            # Filter to same price category
            category_num = 0 if price_category == 'Low' else (1 if price_category == 'Medium' else 2)
            same_category_data = data[data['Price_Category'] == category_num]
            type_psf = same_category_data.groupby('Type')['Median_PSF'].median().sort_values(ascending=False)
            
            # Remove duplicates by getting unique values
            type_psf = type_psf[~type_psf.duplicated()]
            
            # Get color for line based on selected type
            line_color = '#FF6B6B' if selected_type in type_psf.index else '#4472C4'
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Plot line chart
            x_pos = range(len(type_psf))
            ax.plot(x_pos, type_psf.values, marker='o', color=line_color, linewidth=3, 
                   markersize=10, label=selected_type if selected_type in type_psf.index else 'Type')
            
            # Highlight selected type marker
            for i, (ptype, value) in enumerate(zip(type_psf.index, type_psf.values)):
                if ptype == selected_type:
                    ax.scatter([i], [value], color='#FF6B6B', s=200, zorder=5, edgecolors='black', linewidth=2)
                    # Add label only for selected point
                    ax.text(i, value, f'{value:.2f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=9)
            
            ax.set_ylabel('Median PSF (RM)', fontsize=11, fontweight='bold')
            ax.set_title(f'{price_category} Category - PSF by Type', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(type_psf.index, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            st.pyplot(fig)
        
        st.markdown('---')
        
        # Visualization 2: Transactions by Type - Highlight Selected
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown('**Transactions by Type (Same Category - Highlighted)**')
            
            # Filter to same price category
            category_num = 0 if price_category == 'Low' else (1 if price_category == 'Medium' else 2)
            same_category_data = data[data['Price_Category'] == category_num]
            type_transactions = same_category_data.groupby('Type')['Transactions'].sum().sort_values(ascending=False)
            
            # Highlight selected type
            colors_trans = []
            for ptype in type_transactions.index:
                if ptype == selected_type:
                    colors_trans.append('#FF6B6B')  # Bright red for selected
                else:
                    colors_trans.append('#70AD47')  # Green for others
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(range(len(type_transactions)), type_transactions.values, color=colors_trans,
                          edgecolor='black', linewidth=2)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.set_ylabel('Total Transactions')
            ax.set_title(f'{price_category} Category - Activity by Type', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(type_transactions)))
            ax.set_xticklabels(type_transactions.index, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        with col4:
            st.markdown('**Transactions by State (Same Category - Highlighted)**')
            
            # Filter to same price category
            category_num = 0 if price_category == 'Low' else (1 if price_category == 'Medium' else 2)
            same_category_data = data[data['Price_Category'] == category_num]
            state_transactions = same_category_data.groupby('State')['Transactions'].sum().sort_values(ascending=False)
            
            # Highlight selected state
            colors_state_trans = []
            for state in state_transactions.index:
                if state == selected_state:
                    colors_state_trans.append('#FF6B6B')  # Bright red for selected
                else:
                    colors_state_trans.append('#70AD47')  # Green for others
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(range(len(state_transactions)), state_transactions.values, color=colors_state_trans,
                          edgecolor='black', linewidth=2)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            ax.set_ylabel('Total Transactions')
            ax.set_title(f'{price_category} Category - Activity by State', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(state_transactions)))
            ax.set_xticklabels(state_transactions.index, rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        st.markdown('---')
        
        # Visualization 3: Median Price by State - Highlight Selected (Same Category)
        st.markdown('**Median Price by State (Same Category - Highlighted)**')
        
        # Filter to same price category
        category_num = 0 if price_category == 'Low' else (1 if price_category == 'Medium' else 2)
        same_category_data = data[data['Price_Category'] == category_num]
        state_prices = same_category_data.groupby('State')['Median_Price'].median().sort_values(ascending=False)
        
        # Highlight selected state
        colors_state = []
        for state in state_prices.index:
            if state == selected_state:
                colors_state.append('#FF6B6B')  # Bright red for selected
            else:
                colors_state.append('#4472C4')  # Blue for others
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(state_prices.index, state_prices.values, color=colors_state, 
                       edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' RM {width:,.0f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Median Price (RM)', fontsize=11, fontweight='bold')
        ax.set_title(f'{price_category} Category - Price by State', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        
        st.markdown('---')
        
        # Display detailed filtered data
        st.subheader('Detailed Information')
        display_cols = ['Township', 'Area', 'State', 'Tenure', 'Type', 'Median_Price', 'Median_PSF', 'Transactions']
        st.dataframe(filtered_data[display_cols], use_container_width=True)
        
        st.markdown('---')
        
        # Summary section
        st.subheader('📊 Summary')
        summary_text = f"""
        **Property Location Summary:**
        - **Township:** {selected_township}
        - **Area:** {selected_area}
        - **State:** {selected_state}
        
        **Property Details:**
        - **Tenure:** {selected_tenure}
        - **Type:** {selected_type}
        
        **Market Analysis:**
        - **Median Price:** RM {median_price:,.0f}
        - **Price Category:** {price_category}
        - **Price per Square Foot:** RM {median_psf:,.2f}
        - **Market Activity:** {int(transactions)} transactions
        
        **Interpretation:**
        This property falls in the **{price_category}** price range for {selected_state}.
        With a median price of RM {median_price:,.0f}, it represents a
        **{price_category.lower()} value** option in this market segment.
        """
        st.info(summary_text)
        
    else:
        st.warning('❌ No matching properties found for the selected criteria.')

