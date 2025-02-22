import pandas as pd
from sentence_transformers import SentenceTransformer, util

from data_preprocessing import getDF1, getDF2

df1 = getDF1()
df2 = getDF2()

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Exclude the last column ("match") in df2 from processing
columns_to_process = list(range(len(df2.columns) - 1))  # All indices except the last one

# Initialize the new DataFrame with the same column names as df2
new_df = pd.DataFrame(columns=df2.columns)

# Iterate through each column index
for col_index in columns_to_process:
    similarities = []
    for i in range(len(df1)):  # Process each row
        value_df1 = str(df1.iloc[i, col_index])  # Value from df1 using index
        value_df2 = str(df2.iloc[i, col_index])  # Value from df2 using index

        # Generate embeddings
        embedding1 = model.encode(value_df1, convert_to_tensor=True)
        embedding2 = model.encode(value_df2, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_similarity = util.cos_sim(embedding1, embedding2).item()
        similarities.append(cosine_similarity)

    # Add similarity scores to the new DataFrame
    new_df.iloc[:, col_index] = similarities

# Copy the "match" column from df2 to new_df
new_df["match"] = df2["match"]
# Convert "match" column to binary labels
new_df["match"] = new_df["match"].map({"j": 1, "n": 0})

# Display the new DataFrame
print("\nNew DataFrame with Cosine Similarity:")
print(new_df)

# Save to an Excel file
new_df.to_excel("cosine_similarity_results.xlsx", index=False)


