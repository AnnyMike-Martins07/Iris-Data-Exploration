import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Iris dataset from Seaborn
# This dataset contains measurements of iris flowers (e.g., sepal length, width)
df = sns.load_dataset("iris")
print("Here are the first few rows of the Iris dataset:")
print(df.head())  # Display the first 5 rows for a quick look at the data

# 2. Show summary statistics
# This will help us understand the basic statistics of the dataset.
print("\nSummary Statistics for the Dataset:")
print(df.describe())  # Shows basic statistical details
print("\nMissing values in each column:")
print(df.isnull().sum())  # Check for missing data

# 3. Plotting: Visualize Sepal Length Distribution
# Let's take a look at how the Sepal Length is distributed.
# Do you notice any clusters or patterns?
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal_length'], kde=True, color='skyblue')
plt.title("Distribution of Sepal Length: What does it tell us about flower size?")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Plotting: Count the number of flowers for each species
# We can see how many flowers are in each species.
plt.figure(figsize=(6, 4))
sns.countplot(x='species', data=df, palette='pastel')
plt.title("How many flowers are there in each species?")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

# 5. Summary and Reflection
# Let’s reflect on what we saw:
print("\nConclusion:")
print("From the summary statistics, we can see that the average sepal length is approximately 5.8 cm.")
print("The histogram suggests that the sepal lengths are fairly evenly distributed.")
print("From the count plot, we see that Setosa is the most common species.")
