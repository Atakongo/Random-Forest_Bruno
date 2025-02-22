import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the Excel file
file_path = "test_klassik.xlsx"  # Update with your file path
df = pd.read_excel(file_path)

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Select "left" columns for df1
df1 = df.filter(like="left")

# Select "right" columns and create a copy to avoid the warning
df2 = df.filter(like="right").copy()

# Add the "label" column explicitly
df2["label"] = df["label"]

# Ensure column names align for processing
print("df1 Columns:", df1.columns)
print("df2 Columns:", df2.columns)

# Create an empty DataFrame for storing cosine similarity results
cosine_similarity_df = pd.DataFrame(columns=df2.columns)

# Iterate through each column index
for col_index in range(len(df1.columns)):
    similarities = []
    for i in range(len(df1)):
        # Get the column value for the current row and column index
        value_df1 = str(df1.iloc[i, col_index])
        value_df2 = str(df2.iloc[i, col_index])

        # Generate embeddings
        embedding1 = model.encode(value_df1, convert_to_tensor=True)
        embedding2 = model.encode(value_df2, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_similarity = util.cos_sim(embedding1, embedding2).item()
        similarities.append(cosine_similarity)

    # Assign similarity scores to the corresponding column in the result DataFrame
    cosine_similarity_df[df2.columns[col_index]] = similarities

# Add the "label" column to the new DataFrame
cosine_similarity_df["label"] = df2["label"]

# Display the resulting DataFrame
print("\nMerged DataFrame with Cosine Similarities:")
print(cosine_similarity_df)

# Save the new DataFrame to a file for reuse
#cosine_similarity_df.to_csv("cosine_similarity_results.csv", index=False)

# Optionally save as Excel
cosine_similarity_df.to_excel("cosine_similarity_GVL_test_klassik_results.xlsx", index=False)
