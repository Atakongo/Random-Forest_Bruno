from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from Trainer import Trainer
from joblib import dump,load


class BaseModel(ABC):
    def __init__(self, param_grid):
        """
        Initialize the model with a parameter grid for hyperparameter tuning.
        """
        self.param_grid = param_grid
        self.model = None  # Placeholder for the model instance
        self.tuned = False  # Boolean if model was tuned
        self.trainer = None
        

    @abstractmethod
    def build_model(self):
        """
        Build the model instance (e.g., RandomForestClassifier or RandomForestRegressor).
        """
        pass


    def tuneModel(self, trainer: Trainer, scoring):
        """
        Hyperparameter tuning the model on the validation data set.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call build_model() first.")
        self.trainer = trainer
        self.model= trainer.findBestModel(self.model,scoring)
        self.tuned=True

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        """
        if self.tuned is False:
            print("Model has not been tuned.")
        else:
            print("Model has been tuned.")
            
    def saveModel(self,modelName):
        if(self.model != None):
            dump(self.model,modelName)
        else:
            print("There exists no model to save")
    
    @abstractmethod
    def loadModel(patheToModel):
        pass
    
    def getModel(self):
        return self.model
    
   
        
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


class RandomForestClassifierModel(BaseModel):
    def build_model(self):
        """
        Build the RandomForestClassifier instance.
        """
        self.model = RandomForestClassifier()
    
    def tuneModel(self, trainer: Trainer, scoring):
        super().tuneModel(trainer, scoring)
        

    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier on the test set.
        """
        super().evaluate(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        self.calcMetrics(y_test,y_pred)
        
    def calcMetrics(self, y_test, y_pred):
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1Score = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1Score:.2f}")

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)
        
    def loadModel(self,pathToModel):
        model = load(pathToModel)
        if isinstance(model, RandomForestClassifier):
             self.model = model
        else: 
            print("The model to be loaded is not a RandomForestClassifier")
       
    


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class RandomForestRegressorModel(BaseModel):
    def build_model(self):
        """
        Build the RandomForestRegressor instance.
        """
        self.model = RandomForestRegressor()
        
    def tuneModel(self,trainer: Trainer, scoring):
        super().tuneModel(trainer,scoring)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the regressor on the test set.
        """
        super().evaluate(X_test,y_test)
        y_pred = self.model.predict(X_test)
        return mean_squared_error(y_test, y_pred)
    
    def loadModel(self,pathToModel):
        model = load(pathToModel)
        if isinstance(model, RandomForestRegressor):
            self.model = model
        else: 
            print("The model to be loaded is not a RandomForestRegressor")
