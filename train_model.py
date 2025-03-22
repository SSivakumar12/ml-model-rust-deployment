import warnings
warnings.filterwarnings("ignore")
import os
import time
import json
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def load_split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    loads data performs basic pre-processing in preparation for model training
    """
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
    return X_train, X_test, y_train, y_test

def identify_save_strategy(model: sklearn, 
                           feature_names: list[str]) -> None:
    """
    Based on the repr of model architecture, 
    evaluates the best saving strategy to save model to run in rust
    
    currently only logisticregression and DecisionTreeClassifier are supported
    """
    supported_models = ["LogisticRegression", "DecisionTreeClassifier"]
    
    if "LogisticRegression" in repr(model):
        file_path = "model_weights/logistic_regression_architecture.json"
        if not os.path.isfile(file_path):
            with open(file_path, "w") as file:
                json.dump({"weight": model.coef_.flatten().tolist(), 
                           "bias": model.intercept_.tolist()}, file, indent=4)

    elif "DecisionTreeClassifier" in repr(model):
        file_path = "model_weights/decision_tree_classifier_architecture.json"
        if not os.path.isfile(file_path):
            with open(file_path, "w") as file:
                json.dump(save_tree_based_classifier(model.tree_, feature_names),
                          file, 
                          indent=4)

    elif "RandomForestClassifier" in repr(model):
        file_path = "model_weights/random_forest_classifier_architecture.json"
        if not os.path.isfile(file_path):
            with open(file_path, "w") as file:
                json.dump([save_tree_based_classifier(tree.tree_, feature_names)
                           for tree in model.estimators_],
                          file, 
                          indent=4)
    else:
        raise NotImplementedError(
            f"""{repr(model)} currently is not implemented yet. 
            The only supported model are {supported_models}""")


def train_model_architecture(model: sklearn, 
                             X_train: pd.DataFrame, 
                             y_train: pd.Series,
                             X_test: pd.DataFrame,
                             y_test: pd.DataFrame
                             ) -> None:
    start_time = time.time()
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

    identify_save_strategy(model, list(X_train.columns))    

def save_tree_based_classifier(tree: sklearn, feature_names: list[str]) -> dict:
    """
    recursively goes down a tree to define the decisions in dictionary format
    this can the be saved as a JSON and picked up by rust
    """
    def recurse(node, condition=None):
        if tree.children_left[node] == -1:
            return {"condition": condition, "value": tree.value[node].tolist()}
        return {
            "condition": condition,
            "feature": feature_names[tree.feature[node]],
            "threshold": float(tree.threshold[node]),
            "left": recurse(tree.children_left[node], f"{feature_names[tree.feature[node]]} <= {tree.threshold[node]:.4f}"),
            "right": recurse(tree.children_right[node], f"{feature_names[tree.feature[node]]} > {tree.threshold[node]:.4f}"),
        }

    return recurse(0)

if __name__ == "__main__":
    # define models that can be trained in python and savved in a format
    # which can be implemented in rust
    models = [
        LogisticRegression(random_state=0),
        DecisionTreeClassifier(random_state=0, max_depth=3),
        RandomForestClassifier(random_state=0, max_depth=3, n_estimators=10)
    ]
    X_train, X_test, y_train, y_test = load_split_data()

    for model in models:
        train_model_architecture(model, X_train, y_train, X_test, y_test)
