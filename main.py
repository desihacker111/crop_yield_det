import pandas as pd
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
plt.style.use("ggplot")

import pandas as pd

df = pd.read_csv("yield_df.csv")  # replace with actual filename if it's different
print(df.head())
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df.shape
df.describe()
df_numeric = df.select_dtypes(include=['number'])
correlation_matrix = df_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
df.corr(numeric_only=True)

#data visualization
len(df['Area'].unique())
len(df['Item'].unique())
plt.figure(figsize=(15,20))
sns.countplot(y= df['Area'])
plt.show()
plt.figure(figsize=(15,20))
sns.countplot(y= df['Item'])
plt.show()
(df['Area'].value_counts() < 400).sum()

country=df['Area'].unique()
yield_per_country=[]
for state in country:
    yield_per_country.append(df[df['Area']==state]['hg/ha_yield'])
df['hg/ha_yield'].sum()
yield_per_country
print(type(yield_per_country), len(yield_per_country))
print(type(country), len(country))

print(yield_per_country[:5])
print(country[:5])

flat_yield = pd.concat(yield_per_country).reset_index(drop=True)

# Country list tumne diya hai, assume it's a flat list too
# Check if lengths match
print(len(flat_yield), len(country))  # They should be equal

print(df.columns)

country_yield = df.groupby('Area')['hg/ha_yield'].mean().sort_values(ascending=False)

# Plot
plt.figure(figsize=(15, 20))
sns.barplot(x=country_yield.values, y=country_yield.index)
plt.xlabel("Average Yield (hg/ha)")
plt.ylabel("Country")
plt.title("Average Crop Yield per Country")
plt.show()
plt.figure(figsize=(15, 20))
sns.barplot(x=country_yield.values, y=country_yield)
plt.show()
crops=df['Item'].unique()
yield_per_crop=[]
for crop in crops:
    yield_per_crop.append(df[df['Item']==crop]['hg/ha_yield'].sum())
plt.figure(figsize=(15,20))
sns.barplot(y=crops,x=yield_per_crop)
plt.show()
df.head()
df.columns
col = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes', 'avg_temp','Area', 'Item', 'hg/ha_yield']
df = df[col]
df.head()
X = df.drop('hg/ha_yield', axis = 1)
y = df['hg/ha_yield']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0, shuffle=True)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

ohe = OneHotEncoder(drop = 'first')
scale = StandardScaler()

preprocesser = ColumnTransformer(
    transformers = [
        ('StandardScale', scale, [0,1,2,3]),
        ('OneHotEncode', ohe, [4,5])
    ], 
    remainder = 'passthrough'
) 
X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy  = preprocesser.fit_transform(X_test)
preprocesser.get_feature_names_out(col[:-1])

from sklearn.linear_model import LinearRegression,Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score


import scipy
print(scipy.version)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(max_iter=10000),
    'Ridge': Ridge(solver='lsqr'),
    'Decision Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor()
}


from sklearn.metrics import mean_absolute_error, r2_score

for name, model in models.items():
    model.fit(X_train_dummy, y_train)
    y_pred = model.predict(X_test_dummy)
    print(f"{name}: MAE = {mean_absolute_error(y_test, y_pred):.2f}, R² = {r2_score(y_test, y_pred):.2f}")

dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy,y_train)
dtr.predict(X_test_dummy)
df.columns
df.head()
# Predictive System
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X = df.drop(columns='hg/ha_yield')
y = df['hg/ha_yield']

# Numerical and categorical columns
num_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
cat_features = ['Area', 'Item']

# Column transformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the crop yield dataset
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")
            
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # Verify required columns exist
        required_columns = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Select features for prediction
        features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        X = df[features]
        y = df['hg/ha_yield']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create directory for models if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the scaler
        with open('models/preprocessor.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def main():
    """
    Main function to prepare the data
    """
    try:
        # Fix: Change file to __file__
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, "crop_yield_data.csv")
        
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_file)
        print("Data preprocessing completed successfully!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Save processed data
        np.save('models/X_train.npy', X_train)
        np.save('models/X_test.npy', X_test)
        np.save('models/y_train.npy', y_train)
        np.save('models/y_test.npy', y_test)
        print("Processed data saved successfully!")
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()

from sklearn.ensemble import RandomForestRegressor

model_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

def prediction(year, rain, pesticide, temp, area, crop):
    input_df = pd.DataFrame([{
        'Year': year,
        'average_rain_fall_mm_per_year': rain,
        'pesticides_tonnes': pesticide,
        'avg_temp': temp,
        'Area': area,
        'Item': crop
    }])
    return model_pipeline.predict(input_df)[0]

result = prediction(1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
print(f"Predicted Crop Yield: {result}")

result = prediction(1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
print(result)
import pickle

pickle.dump(model_pipeline, open("model_pipeline.pkl", "wb"))

import pandas as pd

test_data = pd.read_csv("yield_df.csv")  # Make sure the filename is correct

import pickle

model_pipeline = pickle.load(open("model_pipeline.pkl", "rb"))

predictions = model_pipeline.predict(test_data)
predictions = model_pipeline.predict(test_data)


import pandas as pd

submission = pd.DataFrame({
    'Id': test_data.index,       # This will just use row numbers as ID
    'Prediction': predictions
})

submission.to_csv("submission.csv", index=False)

# Check the first few rows of the submission
submission.head()

submission.to_csv("submission.csv", index=False)

#modl traning and evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train_dummy, y_train)

y_pred = model.predict(X_test_dummy)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R² Score:", r2)

# Visualize Prediction Performance
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")
plt.show()

#Feature Importance (if using tree-based models)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train_dummy, y_train)

importances = rf.feature_importances_
features = preprocesser.get_feature_names_out(col[:-1])

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12,10))
sns.barplot(data=importance_df.head(20), x='Importance', y='Feature')
plt.title("Top 20 Feature Importances")
plt.show()


result = prediction(2000, 2485.0, 128.0, 20.37, 'Albania', 'Potatoes')
print(f"Predicted Crop Yield: {result}")