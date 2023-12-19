import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class data_analysis():
    def __init__(self):
        self.df = pd.read_csv(r"C:\Users\Dell\Desktop\project\ML Project\titanic\train.csv")
    
        # Drop unnecessary columns
        self.df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

        # Handle missing values
        self.df["Age"].fillna(self.df["Age"].mean(), inplace=True)
        self.df["Embarked"].fillna(self.df["Embarked"].mode()[0], inplace=True)

        # Convert categorical variables to numerical using LabelEncoder
        label_encoder = LabelEncoder()
        self.df["Sex"] = label_encoder.fit_transform(self.df["Sex"])
        self.df["Embarked"] = label_encoder.fit_transform(self.df["Embarked"])

        # Split the dataset into features and target
        self.X = self.df.drop("Survived", axis=1)
        self.y = self.df["Survived"]
        #print(self.X)
        # Random Forest parameters
        self.n_estimators = 100
        self.max_features = "auto"
        self.bootstrap = True
        self.random_state = 42

        # Initialize an empty list to store the decision trees
        self.trees = []

        # Random Forest training
        for _ in range(self.n_estimators):
            # Bootstrap sampling with replacement
            self.X_boot, self.y_boot = resample(self.X, self.y, replace=True, random_state=self.random_state)

            # Create a decision tree
            tree = DecisionTreeClassifier(max_features=self.max_features, random_state=self.random_state)

            # Fit the decision tree on the bootstrapped dataset
            tree.fit(self.X_boot, self.y_boot)

            # Append the decision tree to the list of trees
            self.trees.append(tree)

        # Random Forest predictio


def predict1(X):
        obj = data_analysis()
        # Initialize an empty array for storing the predictions
        predictions = np.zeros((X.shape[0],), dtype=np.int)

        # Make predictions using each decision tree
        for tree in obj.trees:
            tree_predictions = tree.predict(X)
            predictions += tree_predictions

        # Majority vote to determine the final prediction
        predictions = np.where(predictions > (obj.n_estimators / 2), 1, 0)

        return predictions    

def accuray():
    obj1 = data_analysis()
# Calculate accuracy
    predictions = predict1(obj1.X)
    accuracy = accuracy_score(obj1.y,predictions)
    print("Accuracy:", accuracy)
    return accuracy


def Random_forest_analysis(str):
    obj1 = data_analysis()
    # User input for prediction
    #user_input = input("Enter the values for prediction (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked): ")
    #accuray()
    input_values = list(map(float, str.split(",")))

    # Create a DataFrame with user input
    input_df = pd.DataFrame([input_values], columns=obj1.X.columns)

    # Make prediction on user input
    user_prediction = predict1(input_df)
    if(user_prediction == 0):
        return "Dead"
    elif(user_prediction == 1):
        return "alive"
    else:
        return "No predictions could be done"
