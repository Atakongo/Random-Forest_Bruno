
class ParameterGridStore:
    _instance = None  # Class-level variable to store the single instance #Singleton

    
    def __init__(self):
        # Store grids in a dictionary
        self.grids = {}
        
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def add_grid(self, name, grid):
        """Add a parameter grid with a name."""
        self.grids[name] = grid

    def get_grid(self, name):
        """Retrieve a grid by name."""
        return self.grids.get(name, None)

    def all_grids(self):
        """Return all grids."""
        return self.grids


class ParameterGridStore_Manager:
    _instance = None  # Class-level variable to store the single instance #Singleton

    def __init__(self):
        pass
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    param_store = ParameterGridStore()

    grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
    }
    
    grid_rf_advanced = {
    'n_estimators': [300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }
    
    param_store.add_grid('RandomForest', grid_rf)
    param_store.add_grid('RandomForest_Advanced', grid_rf)
    #param_store.add_grid('SVM', grid_svm)

    print("RF Grid:", param_store.get_grid('RandomForest'))
    #print("SVM Grid:", param_store.get_grid('SVM'))
    
    def get_grid(self, name):
        """Retrieve a grid by name."""
        return self.param_store.get_grid(name)
