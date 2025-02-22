import pandas as pd
from abc import ABC, abstractmethod
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from Parameter_Grids_Store import ParameterGridStore_Manager
from DataAdapter import DataAdapter
from sklearn.preprocessing import MinMaxScaler

class Trainer(ABC):
    """
    Abstract Trainer class enforcing the implementation of findBestModel.
    """
    def __init__(self, dataAdapter: DataAdapter, paramGridStore_Manager: ParameterGridStore_Manager, testData_DF, trainData_DF, validationData_DF,scaler):
        self.dataAdapter = dataAdapter
        self.parameterStoreManager = paramGridStore_Manager
        self.param_grid_rf = self.parameterStoreManager.get_grid('RandomForest_Advanced')
        self.df_train = trainData_DF
        self.df_val = validationData_DF
        self.df_test = testData_DF
        # 'match' column is the renamed to 'label' as the target column
        self.X_train = self.df_train.drop(columns=["label"])
        self.y_train = self.df_train["label"]

        self.X_val = self.df_val.drop(columns=["label"])
        self.y_val = self.df_val["label"]

        self.X_test = self.df_test.drop(columns=["label"])
        self.y_test = self.df_test["label"]

        #handle missing values
        self.X_train = self.X_train.fillna(0)  # Replace NaN with 0 
        self.X_val = self.X_val.fillna(0)  # Replace NaN with 0 
        self.X_test = self.X_test.fillna(0)  # Replace NaN with 0 

        #normalize the data
        self.scaler = scaler
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.fit_transform(self.X_val)
        self.X_test_scaled = self.scaler.fit_transform(self.X_test)
        print("Trainer initialisation: DONE")
       


    @abstractmethod
    def findBestModel(self, model, scoring):
        """
        Abstract method to find the best model based on the given scoring metric.
        Must be implemented by subclasses.

        Args:
            model: The machine learning model to train.
            scoring: The scoring metric to optimize.
        """
        pass
    
    @abstractmethod
    def get_best_params(self):
        """
        Abstract method to get the best parameters based on the model performance in the trainin scoring metric.
        Must be implemented by subclasses.
        """
        pass

class GridSearchTrainer(Trainer):
    """
    Trainer class that uses GridSearchCV to find the best model.
    """

    def __init__(self, cv,dataAdapter: DataAdapter, paramGridStore_Manager: ParameterGridStore_Manager, testData_DF, trainData_DF, validationData_DF,scaler):
        """
        Initialize the GridSearchTrainer.

        Args:
            param_grid (dict): Parameter grid for GridSearchCV.
            cv (int): Number of cross-validation folds.
        """
        super().__init__(dataAdapter,paramGridStore_Manager, testData_DF, trainData_DF, validationData_DF,scaler)
        self.cv = cv
        self.grid_search = None

    def findBestModel(self, model, scoring):
        """
        Perform a grid search to find the best model.

        Args:
            model: The machine learning model to train.
            scoring: The scoring metric to optimize.

        Returns:
            GridSearchCV object with the best model.
        """
        model.fit(self.X_train_scaled, self.y_train)
        cv_scores = cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=self.cv, scoring=scoring) 
        print(f"Cross-validated {scoring}: {cv_scores.mean():.2f}") 
        best_model = self.trainWithGridSearch(model, scoring, self.X_val_scaled, self.y_val, self.param_grid_rf)
        return best_model

    
    def trainWithGridSearch(self,model, scoring, X_val, y_val, param_grid):
        # Instantiate the grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid= param_grid,
            scoring= scoring,  # Metric to optimize
            cv=10,                # Number of cross-validation folds || was 10 but for xgboost 5 is tested
            n_jobs=-1,            # Use all available cores
            verbose=2             # Print progress
        )
    
        # Fit the model to the validation data
        grid_search.fit(X_val, y_val)
    
        # Best hyperparameters and precision
        print("Best Parameters:", grid_search.best_params_)
        print("Best {scoring}  Score:", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        validation_metric = best_model.score(X_val, y_val)
        print("Grid Search {scoring} Score on valdition data of the best model:", validation_metric)
        self.set_GridSearch(grid_search)
        return best_model
    
    def get_best_params(self):
        """
        Retrieve the best parameters from grid search.
        """
        if self.grid_search is None:
            raise ValueError("Grid search has not been performed.")
        return self.grid_search.best_params_
    
    def set_GridSearch(self,gridSearch):
        self.grid_search=gridSearch
        


   
    
    




    



    
