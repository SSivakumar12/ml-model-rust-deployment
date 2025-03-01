import warnings
warnings.filterwarnings("ignore")
import time
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/titanic-dataset.csv")
df = df.loc[:, ['Survived', 'Age', 'Sex', 'Pclass']]
df = pd.get_dummies(df, columns=['Sex', 'Pclass'])
df.dropna(inplace=True)

X_train, X_test, y_train, y_test =\
    train_test_split(df.drop('Survived', axis=1), 
                     df['Survived'], 
                     test_size=0.2, 
                     stratify=df['Survived'],
                     random_state=0)
# Save test data
with open("data/testing_dataset.json", "w") as file:
    json.dump({"x_test": X_test.to_dict("records"), 
               "y_test": y_test.to_list()}, 
              file, 
              indent=4)


start_time = time.time()
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
end_time = time.time()
print("total time for training", end_time - start_time)


start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()
print("total time to make predictions in seconds", end_time - start_time)

overall_metrics = f"""
    - classification report: \n
    {classification_report(y_test, y_pred)}
"""
print(overall_metrics)

# Save Model weights to json so that the optimal model can be recreated in Rust 
with open("logistic_regression_architecture.json", "w") as file:
    json.dump({"weight": model.coef_.flatten().tolist(), 
               "bias": model.intercept_.tolist()}, file, indent=4)