import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

training_set = pd.read_csv('~/project/TSA/student_scores/training_scores.csv')
testing_set = pd.read_csv('~/project/TSA/student_scores/testing_data.csv')

print(training_set.columns)
print(testing_set.columns)

categorical_features = ['gender', 'school_type', 'lunch', 'school_setting']

column_transformer = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')

X_train = training_set[['school_setting', 'school_type', 'n_student', 'gender', 'lunch']]
y_train = training_set['posttest']  

X_test = testing_set[['school_setting', 'school_type', 'n_student', 'gender', 'lunch']]
y_test = testing_set["posttest"]  


X_train_transformed = column_transformer.fit_transform(X_train)
X_test_transformed = column_transformer.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_transformed, y_train)

y_pred = model.predict(X_test_transformed)

for predicted in y_pred[:9]:
    print(f"Predicted: {predicted:.2f}")
