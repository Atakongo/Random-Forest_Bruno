# Load the sentence transformer model
from sentence_transformers import SentenceTransformer
from sklearn.discriminant_analysis import StandardScaler

import Model
import Trainer
from ScoringMetric import ScoringMetric
from Trainer import GridSearchTrainer
from Parameter_Grids_Store import ParameterGridStore_Manager
from Model import RandomForestClassifierModel, RandomForestRegressorModel
from DataAdapter import DataAdapter
from Data_PreProcessing import Data_Preprocessor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

adapter = DataAdapter()
path_test_data = "cosine_similarity_GVL_test_results.xlsx"
path_validation_data = "cosine_similarity_GVL_validation_results.xlsx"
path_train_data = "cosine_similarity_GVL_train_results.xlsx"

test_data_DF = adapter.readDataFromPath(path_test_data)
validation_data_DF = adapter.readDataFromPath(path_validation_data)
train_data_DF = adapter.readDataFromPath(path_train_data)

data_to_predict = adapter.readDataFromPath("cosine_similarity_GVL_toPredictData_classic_january.xlsx") #default

transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
scoring = ScoringMetric.NEGATIVE__MEAN_SQUARED_ERROR.value #use for regression; ScoringMetric.RECALL.value use for classifier


dataPreprocessor = Data_Preprocessor(transformer)
param_store_manager = ParameterGridStore_Manager()
param_grid = param_store_manager.get_grid('RandomForest')
cv=10
scaler = MinMaxScaler()
#gridSearchTrainer = GridSearchTrainer(cv,adapter,param_store_manager,test_data_DF,train_data_DF,validation_data_DF, scaler)
trainer = GridSearchTrainer(cv,adapter,param_store_manager,test_data_DF,train_data_DF,validation_data_DF,scaler)
mainModel: Model = RandomForestRegressorModel(param_grid) #default
#classifier_model = RandomForestClassifierModel(param_grid)
#regressor_model = RandomForestRegressorModel(param_grid)
columns_to_keep = [
    "right_Titel",
    "right_Titelzusatz",
    "right_Mainartist",
    "right_Komponist",
    "right_Aufnahmejahr",
    "right_Genre",
    "right_ISRC",
    "right_EAN",
    "right_UPC",
    "right_Katalognummer",
]


def predictionConcatAndOutput(df_data, y_pred, nameToExcel): #example nameToExcel can be "df_pred_january_classic_Data_scaled.xlsx"
    # Ensure y_pred is a NumPy array
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Debug: Check type and shape
    print("Type of y_pred:", type(y_pred))
    print("Shape of y_pred:", y_pred.shape)

    # Convert y_pred to a Series or DataFrame based on its shape
    if len(y_pred.shape) == 1:  # 1D array
        y_pred_series = pd.Series(y_pred, name="Predictions")
    elif len(y_pred.shape) == 2:  # 2D array
        y_pred_series = pd.DataFrame(y_pred, columns=[f"Prediction_{i}" for i in range(y_pred.shape[1])])
    else:
        raise ValueError("y_pred must be a 1D or 2D array")

    # Ensure row counts match
    if len(y_pred_series) != len(df_data):
        raise ValueError(f"Mismatch in row counts: df_data has {len(df_data)} rows, but y_pred has {len(y_pred_series)} rows.")

    # Concatenate predictions with the original DataFrame
    df_data_with_prediction = pd.concat([df_data, y_pred_series], axis=1)

    # Save to Excel
    df_data_with_prediction.to_excel(nameToExcel, index=True, engine="openpyxl")

    print("DataFrame with predictions saved successfully!")


def predictData(model: Model, dataFrameToPredict, scaler, columns_to_keep):
    print("Prediction process of the model is started")
    
    # Filter the DataFrame to keep only the columns to keep
    filtered_df = dataFrameToPredict[columns_to_keep]
    filtered_df = filtered_df.fillna(0)  # Replace NaN with 0 
    filtered_df_scaled = scaler.fit_transform(filtered_df)
    y_pred = model.getModel().predict(filtered_df_scaled) #classifier_model.model.predict(filtered_df_scaled)

    # Convert filtered_df_scaled (numpy ndarray) to pandas DataFrame
    filtered_df_scaled = pd.DataFrame(filtered_df_scaled,columns=columns_to_keep)

    # Convert y_pred_january_classic_Data to pandas Series (if it isn't already)
    y_pred_series = pd.Series(y_pred)

    # Check lengths for debugging (they should match)
    #print("Length of filtered_df_scaled:", len(filtered_df_scaled))
    #print("Length of y_pred_series:", len(y_pred_series))

    # Add the predictions as a new column
    filtered_df_scaled['Predictions'] = y_pred_series
    return filtered_df_scaled

def vectorizeRawData(toVectorizeDataFrame, nameDataCosineDataFrame):
      cosineDF =  dataPreprocessor.transformRawDFToCosineDF(toVectorizeDataFrame)
      adapter.toExcel(cosineDF,nameDataCosineDataFrame)
        
def build_and_train_classifier_model(train_Df, validation_DF, nameToSaveModel):
    gridSearchTrainer = GridSearchTrainer(cv,adapter,param_store_manager,test_data_DF, train_Df ,validation_DF,scaler)
    # RandomForestClassifierModel usage
    mainModel = RandomForestClassifierModel(param_grid)
    mainModel.build_model()
    mainModel.tuneModel(gridSearchTrainer,scoring)
    mainModel.saveModel(nameToSaveModel) #example "RandomForest_Classifier_MinMaxScaler_Recall_TrainedNormal"

def build_and_train_regression_model(train_Df, validation_DF, nameToSaveModel):
    gridSearchTrainer = GridSearchTrainer(cv,adapter,param_store_manager,test_data_DF, train_Df ,validation_DF,scaler)
    # RandomForestRegressorModel usage
    mainModel = RandomForestRegressorModel(param_grid)
    mainModel.build_model()
    mainModel.tuneModel(gridSearchTrainer,scoring)
    mainModel.saveModel(nameToSaveModel)  #example: "RandomForest_Classifier_MinMaxScaler_Recall_Trained_on_Normal_data"

def continue_training(model: Model, trainer: Trainer, scoring: ScoringMetric): #important build new trainer with new data first if data has been changed
    model.tuneModel(trainer, scoring)
    
def evaluateModel(mainModel: Model, df_test):
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]
    X_test = X_test.fillna(0)  # Replace NaN with 0 
    X_test_scaled = scaler.fit_transform(X_test)
    mainModel.evaluate(X_test_scaled,y_test)
    
def setTrainingData_Df(pathToTrainingData):
    train_data_DF = adapter.readDataFromPath(path_train_data)
    
def setValidationData_Df(pathToValidationData):
    validation_data_DF = adapter.readDataFromPath(path_validation_data)
    
def setTestData_DF(pathToTestData):
    test_data_DF = adapter.readDataFromPath(path_test_data)
    
def setToPredictData(pathToPredictData):
    data_to_predict =  adapter.readDataFromPath(pathToPredictData)

def setScaler(newScaler): #can be MinMax or StandardScaler; Default is min max
    scaler = newScaler
    
def buildGridSearchTrainer():
    trainer = GridSearchTrainer(cv,adapter,param_store_manager,test_data_DF, train_data_DF ,validation_data_DF, scaler)

#build_and_train_regression_model(test_data_DF,validation_data_DF,"test_updated_bruno_master_regressionModel_v1")
mainModel.loadModel("test_updated_bruno_master_regressionModel_v1")
y_pred_for_test_updatedBruno = predictData(mainModel,data_to_predict,scaler,columns_to_keep)
predictionConcatAndOutput(data_to_predict,y_pred_for_test_updatedBruno,"test_updated_bruno_master_predicted_Data_v1")