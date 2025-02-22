import pandas as pd
import os
class DataAdapter: 
    def __init__(self):
        pass
        
    def toExcel(self,df, fileName):
        df.to_excel(fileName,index=True)
    
    def toCSV(self,df, fileName):
        df.to_csv(fileName,index=True)
        
    def readDataFromPath(self,file_path):
        if file_path.lower().endswith('.xlsx'):
           return self.fromExcel(file_path)
        elif file_path.lower().endswith('.csv'):
           return self.fromCSV(file_path)
            
    def fromExcel(self, file_path):
        # Ensure that a file path is provided
        assert file_path, "File path must be specified."
        # Ensure the file has a .csv extension
        assert file_path.lower().endswith('.xlsx'), "File must have a .xlsx extension."
        # Ensure the file exists
        print(file_path)
        assert os.path.isfile(file_path), f"No such file: {file_path}"
        # Read the .xlsx file into a DataFrame
        df = pd.read_excel(file_path)
        return df
    
    def fromCSV(self, file_path):
        # Ensure that a file path is provided
        assert file_path, "File path must be specified."
        # Ensure the file has a .csv extension
        assert file_path.lower().endswith('.csv'), "File must have a .csv extension."
        # Ensure the file exists
        assert os.path.isfile(file_path), f"No such file: {file_path}"

        # Read the .csv file into a DataFrame
        df = pd.read_csv(file_path)
        return df
     
        