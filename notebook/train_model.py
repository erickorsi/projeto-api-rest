'''
This file contains the training, testing and validation of the model,
as well as the model dump after training.
'''

import os
import sys
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from .model_settings import ModelSettings

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(project_dir, 'notebook'))
sys.path.append(os.path.abspath("notebook"))

class TrainModel():
    '''
    Class for training the model.
    '''
    def __init__(
            self,
            seed,
            model_name,
            scoring,
            run_evaluation
        ) -> None:
        self.seed = seed
        self.model_name = model_name
        self.scoring = scoring
        self.run_evaluation = run_evaluation

        self.wine_data = load_wine()
        self.model = None

        # All possible models
        self.models = {}
        self.models['SupportVectorClassifier'] = SVC
        self.models['StochasticGradientDecentC'] = SGDClassifier
        self.models['RandomForestClassifier'] = RandomForestClassifier
        self.models['DecisionTreeClassifier'] = DecisionTreeClassifier
        self.models['GaussianNB'] = GaussianNB
        self.models['KNeighborsClassifier'] = KNeighborsClassifier
        self.models['AdaBoostClassifier'] = AdaBoostClassifier
        self.models['LogisticRegression'] = LogisticRegression

        # Saving results
        self.results = []
        self.names = []

        # Run model evaluations
        if self.run_evaluation == "1":
            self.evaluate_models()

        # Train model
        if self.model_name:
            self.train()
            self.save_model()

    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Gets the x_train and y_train data.

        Returns:
            (x_train, y_train)
        '''
        print("Getting training data.")
        x, y = self.wine_data.data, self.wine_data.target

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)

        return (x_train, y_train)

    def evaluate_models(self) -> None:
        '''
        Evaluates all mapped models and saves metrics.
        '''
        print("Running models evaluation.")
        x_train, y_train = self.get_train_data()
        # Evaluate each model
        for name, model in self.models.items():
            model = model()
            kfold = KFold(n_splits=10, random_state=self.seed, shuffle=True)
            cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=self.scoring)
            self.results.append({
                    'Model': name,
                    'Mean Accuracy': cv_results.mean(),
                    'Standard Deviation': cv_results.std()
                })
        # Save results
        # Ideally, this could be saved in a database
        df_results = pd.DataFrame(self.results)
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Ensure the directory exists before saving
        os.makedirs('notebook/model_evaluations', exist_ok=True)

        df_results.to_csv(f'notebook/model_evaluations/model_evaluation_results_{current_time}.csv', index=False)

    def train(self) -> None:
        '''
        Trains the specified model.
        '''
        print("Training.")
        x_train, y_train = self.get_train_data()
        # Train model
        # May need to implement different proceedures for different models
        self.model = self.models[self.model_name](random_state=self.seed)
        self.model.fit(x_train, y_train)

    def save_model(self) -> None:
        '''
        Saves the model pickle.
        '''
        print("Saving model.")
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Ensure the directory exists before saving
        os.makedirs('notebook/models', exist_ok=True)

        with open(f'notebook/models/wine_model_{current_time}.pkl', 'wb') as model_file:
            pickle.dump(self.model, model_file)

if __name__ == "__main__":
    seed = ModelSettings.SEED
    model_name = ModelSettings.MODEL
    scoring = ModelSettings.SCORING
    run_evaluation = ModelSettings.RUN_EVALUATION

    print("Current Working Directory:", os.getcwd())

    model_trainer = TrainModel(
        seed,
        model_name,
        scoring,
        run_evaluation
    )
