import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm



class Data_Preprocessor:
    transformer = None
     
    def __init__(self, transformer: SentenceTransformer):
        self.transformer = transformer
    
    def transformRawDFToCosineDF(self, rawDF: pd.DataFrame) -> pd.DataFrame:
        print("transformation to cosine data frame started")
        if self.transformer == None:
            return pd.DataFrame()
        
        rawDF.fillna(0)
        #self.validateRawDF(rawDF)
        # Select "left" columns for df1
        df1 = rawDF.filter(like="left")

        # Select "right" columns and create a copy to avoid the warning
        df2 = rawDF.filter(like="right").copy()

        # Add the "label" column explicitly
       # df2["label"] = rawDF["label"]

        # Create an empty DataFrame for storing cosine similarity results
        cosine_similarity_df = pd.DataFrame(columns=df2.columns)

        # Iterate through each column index
        total_amount = len(df1) * len(df1.columns)
        with tqdm(total=total_amount, desc="Processing elements") as pbar:

            for col_index in range(len(df1.columns)):
                similarities = []
                for i in range(len(df1)):

                    # Get the column value for the current row and column index
                    value_df1 = str(df1.iloc[i, col_index])
                    value_df2 = str(df2.iloc[i, col_index])

                    # Generate embeddings
                    embedding1 = self.transformer.encode(value_df1, convert_to_tensor=True)
                    embedding2 = self.transformer.encode(value_df2, convert_to_tensor=True)

                    # Compute cosine similarity
                    cosine_similarity = util.cos_sim(embedding1, embedding2).item()
                    similarities.append(cosine_similarity)
                    pbar.update(1)

            # Assign similarity scores to the corresponding column in the result DataFrame
                cosine_similarity_df[df2.columns[col_index]] = similarities

        # Add the "label" column to the new DataFrame
        #cosine_similarity_df["label"] = df2["label"]
        print("Transformation from raw data to cosine similarity was successfull")
        return cosine_similarity_df

    
    def validateRawDF(self, rawDF: pd.DataFrame):
    
        required_columns = ["label"]
        left_columns_exist = any(col for col in rawDF.columns if "left" in col)
        right_columns_exist = any(col for col in rawDF.columns if "right" in col)

        # Check for required columns
        if not all(col in rawDF.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {set(required_columns) - set(rawDF.columns)}")
        if not left_columns_exist or not right_columns_exist:
            raise ValueError("Missing 'left' or 'right' columns in the dataframe.")
        
        # Check number of rows
        if len(rawDF) < 1:
            raise ValueError("DataFrame does not contain enough rows for processing.")
        
        #self.checkContainsNullValues(rawDF)

        # Check index uniqueness
        if not rawDF.index.is_unique:
            raise ValueError("DataFrame index is not unique.")
        
        #self.checkContainsOnlyStringValues(rawDF)
        
        # Check column count consistency
        if len(rawDF.filter(like="left").columns) != len(rawDF.filter(like="right").columns):
            raise ValueError("Mismatch in the number of 'left' and 'right' columns.")
        
        
        
        print("DataFrame validation successful.")

        
    def checkEmptyStrings(self, rawDF: pd.DataFrame): #currenly not in use
        # Check for empty strings
        if rawDF.filter(like="left").applymap(lambda x: len(str(x).strip()) == 0).any().any():
            raise ValueError("Some 'left' columns contain empty strings or whitespace.")
        if rawDF.filter(like="right").applymap(lambda x: len(str(x).strip()) == 0).any().any():
            raise ValueError("Some 'right' columns contain empty strings or whitespace.")
    
    
    def checkContainsNullValues(rawDF: pd.DataFrame):
        # Check for null values
        if rawDF.isnull().any().any():
            raise ValueError("DataFrame contains null values.")
    
    def checkContainsOnlyStringValues(rawDF: pd.DataFrame):
        # Check data types in "left" and "right" columns
        left_right_columns = rawDF.filter(like="left").columns.tolist() + rawDF.filter(like="right").columns.tolist()
        for col in left_right_columns:
            if not rawDF[col].apply(lambda x: isinstance(x, str)).all():
                raise ValueError(f"Column '{col}' contains non-string values.")
        
    def setTransformer(self, transformer):
        self.transformer = transformer