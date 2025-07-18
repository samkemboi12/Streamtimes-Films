# ------------------------- LIBRARY IMPORTS -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# ------------------------- DATAFRAME SUMMARY -------------------------
# Summarizes shape, structure, and statistics of the dataframe
def summary(df):
    print('-------- Dataframe Shape -------\n')
    print(f'The dataframe has {df.shape[0]} rows and {df.shape[1]} columns\n')

    print('-------- Dataframe Info --------\n')
    print(df.info())  # Includes data types and non-null values
    print('\n')

    print('-------- Dataframe Descriptive Statistics --------\n')
    print(df.describe())  # Summary stats for numeric columns
    print('\n')

# ------------------------- MISSING & DUPLICATE CHECK -------------------------
# Displays count of missing and duplicate values
def data_integrity(df):
    print('-------- DataFrame Missing Values --------\n')
    print(df.isna().sum())  # Missing values per column
    print('\n')

    print('-------- Dataframe Duplicate Values --------\n')
    print(df.duplicated().value_counts())  # Count of duplicated rows

# ------------------------- CATEGORICAL VALUE COUNTS -------------------------
# Prints frequency of values in each categorical column
def column_values(df):
    cat_cols = df.select_dtypes('object').columns
    for col in cat_cols:
        print(f"\n--- Value counts for column: {col} ---")
        print(df[col].value_counts())
        print(f'The column has {len(df[col].unique())} unique values')

# ------------------------- NUMERICAL DISTRIBUTION PLOTS -------------------------
# KDE plots for numerical features
def num_data_distribution(df, num_cols, ncols=3, style='whitegrid',
                          hue=None, palette='rocket', title=None,
                          labelsize=12, fontsize=12):
    nrows = math.ceil(len(num_cols) / ncols)
    width = ncols * 5.65
    height = nrows * 4.2
    figsize = (width, height)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    sns.set_style(style)

    if title:
        fig.suptitle(title)

    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.kdeplot(
            x=col,
            data=df,
            ax=axes[i],
            hue=hue,
            fill=True,
            common_norm=False,
            palette=palette
        )

        axes[i].set_title(col, fontsize=16)
        axes[i].set_xlabel('', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=fontsize)
        axes[i].tick_params(axis='x', labelsize=labelsize)
        axes[i].tick_params(axis='y', labelsize=labelsize)

    # Remove unused subplots
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# ------------------------- CATEGORICAL DISTRIBUTION PLOTS -------------------------
# Count plots for categorical features
def cat_data_distribution(df, cat_cols, ncols=3, style='whitegrid',
                          hue=None, palette='rocket', title=None,
                          labelsize=12, fontsize=12):
    nrows = math.ceil(len(cat_cols) / ncols)
    width = ncols * 5.65
    height = nrows * 4.2
    figsize = (width, height)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    sns.set_style(style)

    if title:
        fig.suptitle(title)

    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.countplot(
            data=df,
            x=col,
            hue=hue,
            ax=axes[i],
            edgecolor='black',
            palette=palette
        )

        axes[i].set_title(col, fontsize=16)
        axes[i].set_ylabel('Count', fontsize=fontsize)
        axes[i].tick_params(axis='x', labelrotation=90, labelsize=labelsize)
        axes[i].tick_params(axis='y', labelsize=labelsize)

    # Remove empty axes
    for j in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# ------------------------- CORRELATION MATRIX -------------------------
def correlation_matrix(df, cols=None):
    # Calculate the correlation matrix for numeric columns in the DataFrame
    if cols:
        correlation_matrix = df[cols].corr()
    else:
        correlation_matrix = df.corr()

    # Create a mask for the upper triangle of the matrix
    # This avoids showing duplicate correlations since the matrix is symmetric
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Set the figure size for better readability
    plt.figure(figsize=(20, 10))
    sns.set_style('white')

    # Plot a heatmap of the correlation matrix
    sns.heatmap(correlation_matrix, mask=mask, # - hide redundant info
                annot=True, fmt=".3f", # format the annotations to 3 decimal places
                annot_kws={"fontsize": 10}, # set the font size of the annotations
                cmap="RdBu", vmin=-1, vmax=1) # set the color scale range from -1 (perfect negative) to 1 (perfect positive)

    # Display the heatmap
    plt.show()

# ------------------------- DATA CLEANING (IMPUTATION) -------------------------
# Imputes missing values using strategies for numeric and categorical columns
def cleaner(df, num_cols, cat_cols, num_strategy='median', cat_strategy='most_frequent'):
    # Only keep columns present in the dataframe
    used_num_cols = [col for col in num_cols if col in df.columns]
    used_cat_cols = [col for col in cat_cols if col in df.columns]

    # Save original dtypes
    original_dtypes = df[used_num_cols + used_cat_cols].dtypes

    # Define imputers
    num_imputer = SimpleImputer(strategy=num_strategy)
    cat_imputer = SimpleImputer(strategy=cat_strategy)

    # Combine into a transformer
    transformer = ColumnTransformer(transformers=[
        ('num', num_imputer, used_num_cols),
        ('cat', cat_imputer, used_cat_cols)
    ])

    # Apply transformations
    cleaned_array = transformer.fit_transform(df)

    # Build cleaned dataframe
    output_cols = transformer.get_feature_names_out()
    cleaned_df = pd.DataFrame(cleaned_array, columns=output_cols, index=df.index)

    # Remove transformer prefixes in column names
    cleaned_df.columns = [name.split('__')[-1] for name in cleaned_df.columns]

    # Restore original data types where possible
    for col in cleaned_df.columns:
        try:
            cleaned_df[col] = cleaned_df[col].astype(original_dtypes.get(col, cleaned_df[col].dtype))
        except (ValueError, TypeError):
            pass

    return cleaned_df

# ------------------------- OUTLIER REMOVAL -------------------------
# Removes outliers based on z-score or IQR
def remove_outliers(df, num_cols, method='zscore', threshold=3):
    if method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(df[num_cols], nan_policy='omit'))
        outlier_mask = (z_scores > threshold).any(axis=1)
        cleaned = df[~outlier_mask]
    
    elif method == 'iqr':
        # IQR method
        cleaned = df.copy()
        for col in num_cols:
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
    else:
        raise ValueError("Method must be 'zscore' or 'iqr'.")

    return cleaned

# ------------------------- MODEL EVALUATION -------------------------
# Evaluates classification model using metrics, plots and AUC
def evaluate_model(model, X, y_true, model_name='Model', class_labels=None):
    # Generate predictions
    y_pred = model.predict(X)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    disp.plot(ax=ax, cmap='Reds')
    ax.grid(False)
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve and AUC
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X)[:, 1]
    else:
        y_scores = model.decision_function(X)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Chance')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right', frameon=True, bbox_to_anchor=[1.015, -0.02])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print classification report and AUC score
    print('-------------- Classification Report --------------\n')
    print(classification_report(y_true, y_pred, target_names=class_labels))
    print('\n-------------- AUC Score --------------\n')
    print(f'AUC Score: {roc_auc:.4f}')


def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    
    def map_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # fallback
    
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, map_pos(pos)) for word, pos in tagged]