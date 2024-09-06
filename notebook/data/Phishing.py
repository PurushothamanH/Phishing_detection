# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv('Phishing.csv')
df.head()

# %%
## check null values
df.isnull().sum()

# %%
df.columns

# %%
df.drop('id',axis=1,inplace=True)
df

# %%
df.info()

# %%
df.shape

# %%
df.duplicated().sum()

# %%
# finding duplicates
df[df.duplicated(keep=False)]

# %%
df.drop_duplicates(keep='first',inplace=True)
df.shape

# %%
plt.figure(figsize=(40,20))

for i, column in enumerate(df.select_dtypes(include=['float64', 'int64']).columns, start=1):
    plt.subplot(len(df.select_dtypes(include=['float64', 'int64']).columns), 2, i * 2 - 1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()

# %%
df.shape

# %%
# Function to detect outliers using IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Create a Boolean DataFrame indicating outliers
outlier_df = pd.DataFrame()

for column in df.select_dtypes(include=['float64', 'int64']).columns:
    outlier_df[column] = detect_outliers_iqr(df, column)

# Consider rows with outliers in more than one column as problematic
rows_with_multiple_outliers = outlier_df.sum(axis=1) > 1  # Adjust threshold as needed

# Remove rows with outliers in multiple columns
df_cleaned = df[~rows_with_multiple_outliers]

print(df_cleaned)

# %%
df_cleaned

# %%
df_cleaned.head()

# %%
df_cleaned.isnull().sum()

# %%
df_cleaned.corr()

# %%
x = df_cleaned.drop("CLASS_LABEL",axis=1)
y = df_cleaned["CLASS_LABEL"]

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
x_train.shape, x_test.shape

# %%
x_train.corr()

# %%
x_train.isna().sum()

# %%
x_train.head(20)

# %%
from sklearn.feature_selection import VarianceThreshold

# Initialize VarianceThreshold to remove features with zero variance
selector = VarianceThreshold(threshold=0)  # threshold=0 removes all zero-variance features

# Apply to your training data
x_train_reduced = selector.fit_transform(x_train)
x_test_reduced = selector.transform(x_test)

# %%
x_train_reduced.shape

# %%
x_test_reduced.shape

# %%
x_train

# %%
x_test

# %%
from sklearn.feature_selection import VarianceThreshold

# Initialize VarianceThreshold to remove features with zero variance
selector = VarianceThreshold(threshold=0)  # threshold=0 removes all zero-variance features

# Apply to your training data
x_train_reduced = selector.fit_transform(x_train)
selected_columns = x_train.columns[selector.get_support()]
x_train_reduced_df = pd.DataFrame(x_train_reduced, columns=selected_columns)
x_test_reduced = selector.transform(x_test)
x_test_reduced_df = pd.DataFrame(x_test_reduced, columns=selected_columns)

# %%
x_train_reduced_df

# %%
x_test_reduced_df

# %%
x_train_reduced_df.corr()

# %%
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(x_train_reduced_df)
# X_test_scaled = scaler.transform(x_test_reduced_df)
# x_train_scaled_df = pd.DataFrame(X_train_scaled,columns=x_train_reduced_df.columns)
# x_test_scaled_df = pd.DataFrame(X_test_scaled,columns=x_test_reduced_df.columns)

# %%
x_train_reduced_df.head()

# %%
x_test_reduced_df.head()

# %%
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # You can use any classifier

# Variables to store the best k and its score
best_k = None
best_score = -1  # Initialize with a low score

# Define the pipeline with SelectKBest and a classifier
for k in range(5, 20):  # Try different values for k
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif, k=k)),
        ('classifier', RandomForestClassifier())
    ])
    scores = cross_val_score(pipeline, x_train_reduced_df, y_train, cv=5)  # 5-fold cross-validation
    mean_score = scores.mean()
    
    print(f"k={k}, mean accuracy={mean_score:.4f}")
    
    # Check if the current mean score is the best
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

# Print the best k and its score
print(f"\nBest k: {best_k}, with a mean accuracy of: {best_score:.4f}")

# %%
from sklearn.feature_selection import SelectKBest, f_classif

# Perform feature selection based on the ANOVA F-test for classification
selector = SelectKBest(score_func=f_classif, k=19)
original_columns = x_train_reduced_df.columns
X_train_selected = selector.fit_transform(x_train_reduced_df, y_train)
X_test_selected = selector.transform(x_test_reduced_df)

# %%
X_train_selected

# %%
X_test_selected

# %%
original_columns = x_train_reduced_df.columns
selected_columns_mask = selector.get_support()
selected_columns = original_columns[selected_columns_mask]
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_columns)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_columns)

# %%
X_train_selected_df.shape

# %%
X_test_selected_df.shape

# %%
X_train_selected_df

# %%
y_train

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected_df)
X_test_scaled = scaler.transform(X_test_selected_df)
x_train_scaled_df = pd.DataFrame(X_train_scaled,columns=X_train_selected_df.columns)
x_test_scaled_df = pd.DataFrame(X_test_scaled,columns=X_test_selected_df.columns)

# %%
x_train_scaled_df

# %%
x_test_scaled_df

# %%


# %%
x_train_scaled_df.shape

# %%
y_train.shape

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential()
model.add(Dense(64, input_dim=19, activation='relu'))  # Input layer with 64 neurons
model.add(Dense(32, activation='relu'))  # Hidden layer with 32 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron (binary classification)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = model.fit(x_train_scaled_df,y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test_scaled_df,y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')


# %%
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# %%
# Overfitting: If you notice that:

# The training loss continues to decrease, but the validation loss increases after some epochs.
# The training accuracy improves, but the validation accuracy plateaus or worsens.
# This indicates that the model is overfitting, learning patterns specific to the training data that do not generalize to unseen data.

# Good Fit: The training and validation loss/accuracy should both improve steadily and stay close to each other.

# %%
# Predict the output for the test data
y_pred = model.predict(x_test_scaled_df)

# If you need binary classification, convert probabilities to class labels (0 or 1)
y_pred_classes = (y_pred > 0.5).astype(int)

# Display predictions
print(y_pred_classes[0:10])

# %%
y_test.head(10)

# %%
### thus the phishing detection is done.. with the value of Loss: 0.1151, Accuracy: 0.9589


