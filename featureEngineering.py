import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_train = pd.read_excel("cosine_similarity_GVL_train_results.xlsx")
df_val = pd.read_excel("cosine_similarity_GVL_validation_results.xlsx")
df_test = pd.read_excel("cosine_similarity_GVL_test_results.xlsx")

# Insert a marker which datapoint belongs to which dataset
df_train['dataset'] = 'train'
df_val['dataset'] = 'val'
df_test['dataset'] = 'test'

# Combine the datasets
combined = pd.concat([df_train, df_val, df_test], axis=0)

def normalizeYearSinCosTransformation():
    # Normalize the year and create sine/cosine transformations
    combined['Aufnahmejahr_sin'] = np.sin(2 * np.pi * combined['right_Aufnahmejahr'] / combined['right_Aufnahmejahr'].max())
    combined['Aufnahmejahr_cos'] = np.cos(2 * np.pi * combined['right_Aufnahmejahr'] / combined['right_Aufnahmejahr'].max())

def frequencyEncoding():
    # Frequency encoding for categorical variables
    combined['Mainartist_frequency'] = combined['right_Mainartist'].map(combined['right_Mainartist'].value_counts())
    combined['Komponist_frequency'] = combined['right_Komponist'].map(combined['right_Komponist'].value_counts())

def rationAndLogTransforamtion():
    # Ratio and log transformation
    combined['EAN_ISRC_ratio'] = combined['right_EAN'] / (combined['right_ISRC'] + 1e-5)
    combined['EAN_ISRC_log'] = np.log1p(combined['EAN_ISRC_ratio'])

def targetEncodingForGenre():
    # Target encoding for Genre
    combined['Genre_target_mean'] = combined.groupby('right_Genre')['label'].transform('mean')

def sanityCheck():
    #Sanity check
    assert train_transformed.shape[0] == df_train.shape[0], "Train split size mismatch!"
    assert val_transformed.shape[0] == df_val.shape[0], "Validation split size mismatch!"
    assert test_transformed.shape[0] == df_test.shape[0], "Test split size mismatch!"
    
def separateToDatasets():
    # Separate the datasets based on the 'dataset' column
    train_transformed = combined[combined['dataset'] == 'train'].drop('dataset', axis=1)
    val_transformed = combined[combined['dataset'] == 'val'].drop('dataset', axis=1)
    test_transformed = combined[combined['dataset'] == 'test'].drop('dataset', axis=1)
    sanityCheck()
    return train_transformed,val_transformed,test_transformed

def toExcel():
    train_transformed.to_excel("cosine_similarity_GVL_train_feature_engineered.xlsx", index=False)
    val_transformed.to_excel("cosine_similarity_GVL_validation_feature_engineered.xlsx", index=False)
    test_transformed.to_excel("cosine_similarity_GVL_test_feature_engineered.xlsx", index=False)
    
def generateHeatMap(df):
    # Assuming df is your dataset
    correlation_matrix = combined.corr(method='pearson')  # Change to 'spearman' if needed

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    
normalizeYearSinCosTransformation()
frequencyEncoding()
rationAndLogTransforamtion()
targetEncodingForGenre()
train_transformed,val_transformed,test_transformed = separateToDatasets()
#toExcel()

