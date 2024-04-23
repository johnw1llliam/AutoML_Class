import pandas as pd
from math import isnan
import itertools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from itertools import combinations
import optuna
import numpy as np
from niapy.task import Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmOptimization
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
import os


def random_forest(df, columns, target, test_size):
    # SPLITTING DATA FOR TRAINING AND TESTING (PRE-NIAPY)
    X = df[columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # FEATURE SELECTION USING NIAPY
    class RandomForestFeatureSelection(Problem):
        def __init__(self, X_train, y_train, alpha=0.99):
            super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
            self.X_train = X_train
            self.y_train = y_train
            self.alpha = alpha

        def _evaluate(self, x):
            selected = x > 0.5
            num_selected = selected.sum()
            if num_selected == 0:
                return 1.0
            accuracy = cross_val_score(RandomForestClassifier(), self.X_train.iloc[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
            num_features = self.X_train.shape[1]
            return self.alpha * (1 - accuracy) + (1 - self.alpha) * (num_selected / num_features)

    problem_rf = RandomForestFeatureSelection(X_train, y_train)
    task_rf = Task(problem_rf, max_iters=100)
    algorithm_rf = ParticleSwarmOptimization(population_size=10, seed=1234)

    best_features_rf, best_fitness_rf = algorithm_rf.run(task_rf)

    selected_features_rf = best_features_rf > 0.5
    selected_columns_rf = X_train.columns[selected_features_rf]
    
    # SPLITTING DATA FOR TRAINING AND TESTING (POST-NIAPY)
    X = df[selected_columns_rf]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # HYPERPARAMETER TUNING WITH OPTUNA
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 300)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_trial = study.best_trial
    results = best_trial.params

    # TRAINING
    model = RandomForestClassifier(n_estimators=results['n_estimators'], max_depth=results['max_depth'], min_samples_split=results['min_samples_split'], min_samples_leaf=results['min_samples_leaf'], random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # EVALUATION
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    result_df = pd.DataFrame(report).transpose()
    result_df.insert(0, 'Stats', result_df.index)
    result_df = result_df.reset_index(drop=True)

    all_running_time = []

    for j in range(10):
        start_time = time.time()
        first_row = X_test.iloc[j]
        first_row = pd.DataFrame([first_row.values], columns=first_row.index)
        predictions = model.predict(first_row)
        end_time = time.time()

        running_time = end_time - start_time
        all_running_time.append(running_time)

    avg_time = sum(all_running_time)/len(all_running_time)
    result_df.loc[len(result_df.index)] = ['running time avg', avg_time, avg_time, avg_time, avg_time]
    
    folder = fr"{os.getcwd()}\results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    result_df.to_csv(fr"{folder}\rf_results.csv", index=False)

def decision_tree(df, columns, target, test_size):
    # SPLITTING DATA FOR TRAINING AND TESTING (PRE-NIAPY)
    X = df[columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # FEATURE SELECTION USING NIAPY
    class DecisionTreeFeatureSelection(Problem):
        def __init__(self, X_train, y_train, alpha=0.99):
            super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
            self.X_train = X_train
            self.y_train = y_train
            self.alpha = alpha

        def _evaluate(self, x):
            selected = x > 0.5
            selected_indices = np.where(selected)[0]
            num_selected = selected_indices.shape[0]
            
            if num_selected == 0:
                return 1.0
    
            accuracy = accuracy = cross_val_score(DecisionTreeClassifier(), self.X_train.iloc[:, selected_indices], self.y_train, cv=2, n_jobs=-1).mean()
            num_features = self.X_train.shape[1]
            return self.alpha * (1 - accuracy) + (1 - self.alpha) * (num_selected / num_features)

    problem_dt = DecisionTreeFeatureSelection(X_train, y_train)
    task_dt = Task(problem_dt, max_iters=100)
    algorithm_dt = ParticleSwarmOptimization(population_size=10, seed=1234)

    best_features_dt, best_fitness_dt = algorithm_dt.run(task_dt)

    selected_features_dt = best_features_dt > 0.5
    selected_columns_dt = X_train.columns[selected_features_dt]
    
    # SPLITTING DATA FOR TRAINING AND TESTING (POST-NIAPY)
    X = df[selected_columns_dt]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                                                
    # HYPERPARAMETER TUNING WITH OPTUNA
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 100, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_trial = study.best_trial
    results = best_trial.params
                      
    # TRAINING
    model = DecisionTreeClassifier(
        max_depth=results['max_depth'],
        min_samples_split=results['min_samples_split'],
        min_samples_leaf=results['min_samples_leaf'],
        random_state=42
    )

    model.fit(X_train, y_train)
                                                        
    # EVALUATION
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    result_df = pd.DataFrame(report).transpose()
    result_df.insert(0, 'Stats', result_df.index)
    result_df = result_df.reset_index(drop=True)

    all_running_time = []

    for j in range(10):
        start_time = time.time()
        first_row = X_test.iloc[j]
        first_row = pd.DataFrame([first_row.values], columns=first_row.index)
        predictions = model.predict(first_row)
        end_time = time.time()

        running_time = end_time - start_time
        all_running_time.append(running_time)

    avg_time = sum(all_running_time)/len(all_running_time)
    result_df.loc[len(result_df.index)] = ['running time avg', avg_time, avg_time, avg_time, avg_time]
    
    folder = fr"{os.getcwd()}\results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    result_df.to_csv(fr"{folder}\dt_results.csv", index=False)

def xgboost(df, columns, target, test_size):
    # SPLITTING DATA FOR TRAINING AND TESTING (PRE-NIAPY)
    X = df[columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # FEATURE SELECTION USING NIAPY
    class XGBoostFeatureSelection(Problem):
        def __init__(self, X_train, y_train, alpha=0.99):
            super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
            self.X_train = X_train
            self.y_train = y_train
            self.alpha = alpha

        def _evaluate(self, x):
            selected = x > 0.5
            
            selected_indices = np.where(selected)[0]
            num_selected = selected_indices.shape[0]
            
            if num_selected == 0:
                return 1.0
            
            accuracy = cross_val_score(XGBClassifier(), self.X_train.iloc[:, selected_indices], self.y_train, cv=2, n_jobs=-1).mean()
            num_features = self.X_train.shape[1]

            return self.alpha * (1 - accuracy) + (1 - self.alpha) * (num_selected / num_features)

    problem_xgb = XGBoostFeatureSelection(X_train, y_train)
    task_xgb = Task(problem_xgb, max_iters=100)
    algorithm_xgb = ParticleSwarmOptimization(population_size=10, seed=1234)

    best_features_xgb, best_fitness_xgb = algorithm_xgb.run(task_xgb)

    selected_features_xgb = best_features_xgb > 0.5
    selected_columns_xgb = X_train.columns[selected_features_xgb]
    
    # SPLITTING DATA FOR TRAINING AND TESTING (POST-NIAPY)
    X = df[selected_columns_xgb]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # HYPERPARAMETER TUNING USING OPTUNA
    def objective(trial):
        params = {
                'max_depth': trial.suggest_int('max_depth', 2, 100),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
                'gamma': trial.suggest_float('gamma', 0.01, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500)
        }

        clf = XGBClassifier(**params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    best_trial = study.best_trial
    results = best_trial.params
    
    # TRAINING
    model = XGBClassifier( 
        max_depth=results["max_depth"],
        min_child_weight=results["min_child_weight"],
        subsample=results["subsample"],
        colsample_bytree=results["colsample_bytree"],
        learning_rate=results["learning_rate"],
        gamma=results["gamma"],
        n_estimators=results["n_estimators"],
    )

    model.fit(X_train, y_train)
    
    # EVALUATION
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    result_df = pd.DataFrame(report).transpose()
    result_df.insert(0, 'Stats', result_df.index)
    result_df = result_df.reset_index(drop=True)

    all_running_time = []

    for j in range(10):
        start_time = time.time()
        first_row = X_test.iloc[j]
        first_row = pd.DataFrame([first_row.values], columns=first_row.index)
        predictions = model.predict(first_row)
        end_time = time.time()

        running_time = end_time - start_time
        all_running_time.append(running_time)

    avg_time = sum(all_running_time)/len(all_running_time)
    result_df.loc[len(result_df.index)] = ['running time avg', avg_time, avg_time, avg_time, avg_time]
    
    folder = fr"{os.getcwd()}\results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    result_df.to_csv(fr"{folder}\xgb_results.csv", index=False)

def logistic_regression(df, columns, target, test_size, solver_type='lbfgs'):
    # SPLITTING DATA FOR TRAINING AND TESTING (PRE-NIAPY)
    X = df[columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # FEATURE SELECTION USING NIAPY
    class LogisticRegressionFeatureSelection(Problem):
        def __init__(self, X_train, y_train, alpha=0.99):
            super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
            self.X_train = X_train
            self.y_train = y_train
            self.alpha = alpha
        
        def _evaluate(self, x):
            selected = x > 0.5
            selected_indices = np.where(selected)[0]  
            num_selected = selected_indices.shape[0]
            
            if num_selected == 0:
                return 1.0
            
            accuracy = cross_val_score(LogisticRegression(), self.X_train.iloc[:, selected_indices], self.y_train, cv=2, n_jobs=-1).mean()
            score = 1 - accuracy
            num_features = self.X_train.shape[1]
            return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)
        
    problem = LogisticRegressionFeatureSelection(X_train, y_train)
    task = Task(problem, max_iters=100)
    algorithm = ParticleSwarmOptimization(population_size=10, seed=1234)
    best_features, best_fitness = algorithm.run(task)

    selected_features = best_features > 0.5
    selected_columns = X_train.columns[selected_features]
    
    # SPLITTING DATA FOR TRAINING AND TESTING (POST-NIAPY)
    X = df[selected_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
     # HYPERPARAMETER TUNING USING OPTUNA
    def objective(trial):
        params = {
                'max_iter': trial.suggest_int('max_iter', 1000, 10000),
                'C': trial.suggest_float('C', 0.1, 1.0),
        }

        clf = LogisticRegression(**params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    best_trial = study.best_trial
    results = best_trial.params
    
    # TRAINING
    model = LogisticRegression( 
        max_iter=results["max_iter"],
        C=results["C"],
        solver=solver_type,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    
    # EVALUATION
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    result_df = pd.DataFrame(report).transpose()
    result_df.insert(0, 'Stats', result_df.index)
    result_df = result_df.reset_index(drop=True)

    all_running_time = []

    for j in range(10):
        start_time = time.time()
        first_row = X_test.iloc[j]
        first_row = pd.DataFrame([first_row.values], columns=first_row.index)
        predictions = model.predict(first_row)
        end_time = time.time()

        running_time = end_time - start_time
        all_running_time.append(running_time)

    avg_time = sum(all_running_time)/len(all_running_time)
    result_df.loc[len(result_df.index)] = ['running time avg', avg_time, avg_time, avg_time, avg_time]
    
    folder = fr"{os.getcwd()}\results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    result_df.to_csv(fr"{folder}\lr_results.csv", index=False)

def support_vector(df, columns, target, test_size, kernel_type='rbf'):
    # SPLITTING DATA FOR TRAINING AND TESTING (PRE-NIAPY)
    X = df[columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # FEATURE SELECTION USING NIAPY
    class SVCFeatureSelection(Problem):
        def __init__(self, X_train, y_train, alpha=0.99):
            super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
            self.X_train = X_train
            self.y_train = y_train
            self.alpha = alpha
        
        def _evaluate(self, x):
            selected = x > 0.5
            selected_indices = np.where(selected)[0]  
            num_selected = selected_indices.shape[0]
            
            if num_selected == 0:
                return 1.0
            
            accuracy = cross_val_score(SVC(), self.X_train.iloc[:, selected_indices], self.y_train, cv=2, n_jobs=-1).mean()
            score = 1 - accuracy
            num_features = self.X_train.shape[1]
            return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)
        
    problem = SVCFeatureSelection(X_train, y_train)
    task = Task(problem, max_iters=100)
    algorithm = ParticleSwarmOptimization(population_size=10, seed=1234)
    best_features, best_fitness = algorithm.run(task)

    selected_features = best_features > 0.5
    selected_columns = X_train.columns[selected_features]
    
    # SPLITTING DATA FOR TRAINING AND TESTING (POST-NIAPY)
    X = df[selected_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
     # HYPERPARAMETER TUNING USING OPTUNA
    def objective(trial):
        params = {
                'C': trial.suggest_float('C', 0.1, 1.0),
                'degree': trial.suggest_int('degree', 1, 20),
        }

        clf = SVC(**params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    best_trial = study.best_trial
    results = best_trial.params
    
    # TRAINING
    model = SVC( 
        C=results["C"],
        degree=results["degree"],
        kernel=kernel_type
    )

    model.fit(X_train, y_train)
    
    # EVALUATION
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    result_df = pd.DataFrame(report).transpose()
    result_df.insert(0, 'Stats', result_df.index)
    result_df = result_df.reset_index(drop=True)

    all_running_time = []

    for j in range(10):
        start_time = time.time()
        first_row = X_test.iloc[j]
        first_row = pd.DataFrame([first_row.values], columns=first_row.index)
        predictions = model.predict(first_row)
        end_time = time.time()

        running_time = end_time - start_time
        all_running_time.append(running_time)

    avg_time = sum(all_running_time)/len(all_running_time)
    result_df.loc[len(result_df.index)] = ['running time avg', avg_time, avg_time, avg_time, avg_time]
    
    folder = fr"{os.getcwd()}\results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    result_df.to_csv(fr"{folder}\svc_results.csv", index=False)

def knn(df, columns, target, test_size, corr_method='pearson', weights_type='uniform', algorithm_type='auto'):
    # SPLITTING DATA FOR TRAINING AND TESTING (PRE-NIAPY)
    X = df[columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # CREATE FEATURE SELECTION USING CORRELATION CHECK COMBINED W/ BRUTE FORCE
    corr = df[columns].corrwith(df["CLASS_LABEL"], method=corr_method)
    corr = corr.to_dict()
    corr = {k: corr[k] for k in corr if not isnan(corr[k])}
    corr = {key: abs(val) for key, val in corr.items()}
    sorted_corr = sorted(corr.items(), key=lambda x:x[1], reverse=True)
    sorted_corr = dict(sorted_corr)
    corr = dict(itertools.islice(sorted_corr.items(), 10)) 
    chosen_cols = list(corr.keys())
    X_train = X_train[chosen_cols]
    X_test = X_test[chosen_cols]
    
    # FEATURE SELECTION USING BRUTE FORCE
    # NUMBER OF SELECTED FEATURES = nC2 + nC3 + ... + nCn
    # TAKE APPROX. 8 MILLION YEARS TO FINISH USING THIS LAPTOP IF ALL COLS USED
    n_features = X_train.shape[1]
    best_score = 0
    best_features = None

    for r in range(1, n_features + 1):
        for combo in combinations(range(n_features), r):
            if len(combo) > 1:
                chosen_features = [chosen_cols[idx] for idx in combo]
                X_subset = X_train[chosen_features]
                scores = cross_val_score(KNeighborsClassifier(), X_subset, y_train, cv=2, n_jobs=-1)
                avg_score = scores.mean()

                if avg_score > best_score:
                    best_score = avg_score
                    best_features = chosen_features
    
    # SPLITTING DATA FOR TRAINING AND TESTING (POST-FEATURE SELECTION)
    X = df[best_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
     # HYPERPARAMETER TUNING USING OPTUNA
    def objective(trial):
        params = {
                'p': trial.suggest_int('p', 1, 2),
                'n_neighbors': trial.suggest_int('n_neighbors', 2, 100),
                'leaf_size': trial.suggest_int('leaf_size', 10, 500),
        }

        clf = KNeighborsClassifier(**params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    best_trial = study.best_trial
    results = best_trial.params
    
    # TRAINING
    model = KNeighborsClassifier( 
        p=results["p"],
        n_neighbors=results["n_neighbors"],
        leaf_size=results["leaf_size"],
        weights=weights_type,
        algorithm=algorithm_type
    )

    model.fit(X_train, y_train)
    
    # EVALUATION
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    result_df = pd.DataFrame(report).transpose()
    result_df.insert(0, 'Stats', result_df.index)
    result_df = result_df.reset_index(drop=True)

    all_running_time = []

    for j in range(10):
        start_time = time.time()
        first_row = X_test.iloc[j]
        first_row = pd.DataFrame([first_row.values], columns=first_row.index)
        predictions = model.predict(first_row)
        end_time = time.time()

        running_time = end_time - start_time
        all_running_time.append(running_time)

    avg_time = sum(all_running_time)/len(all_running_time)
    result_df.loc[len(result_df.index)] = ['running time avg', avg_time, avg_time, avg_time, avg_time]
    
    folder = fr"{os.getcwd()}\results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    result_df.to_csv(fr"{folder}\knn_results.csv", index=False)

def naive_bayes(df, columns, target, test_size):
    # SPLITTING DATA FOR TRAINING AND TESTING (PRE-NIAPY)
    X = df[columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # FEATURE SELECTION USING NIAPY
    class NBFeatureSelection(Problem):
        def __init__(self, X_train, y_train, alpha=0.99):
            super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
            self.X_train = X_train
            self.y_train = y_train
            self.alpha = alpha
        
        def _evaluate(self, x):
            selected = x > 0.5
            selected_indices = np.where(selected)[0]  
            num_selected = selected_indices.shape[0]
            
            if num_selected == 0:
                return 1.0
            
            accuracy = cross_val_score(GaussianNB(), self.X_train.iloc[:, selected_indices], self.y_train, cv=2, n_jobs=-1).mean()
            score = 1 - accuracy
            num_features = self.X_train.shape[1]
            return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)
        
    problem = NBFeatureSelection(X_train, y_train)
    task = Task(problem, max_iters=100)
    algorithm = ParticleSwarmOptimization(population_size=10, seed=1234)
    best_features, best_fitness = algorithm.run(task)

    selected_features = best_features > 0.5
    selected_columns = X_train.columns[selected_features]
    
    # SPLITTING DATA FOR TRAINING AND TESTING (POST-NIAPY)
    X = df[selected_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
     # HYPERPARAMETER TUNING USING OPTUNA
    def objective(trial):
        params = {
                'var_smoothing': trial.suggest_float('var_smoothing', 0.00000001, 0.9),
        }

        clf = GaussianNB(**params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    best_trial = study.best_trial
    results = best_trial.params
    
    # TRAINING
    model = GaussianNB( 
        var_smoothing=results["var_smoothing"]
    )

    model.fit(X_train, y_train)
    
    # EVALUATION
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    result_df = pd.DataFrame(report).transpose()
    result_df.insert(0, 'Stats', result_df.index)
    result_df = result_df.reset_index(drop=True)

    all_running_time = []

    for j in range(10):
        start_time = time.time()
        first_row = X_test.iloc[j]
        first_row = pd.DataFrame([first_row.values], columns=first_row.index)
        predictions = model.predict(first_row)
        end_time = time.time()

        running_time = end_time - start_time
        all_running_time.append(running_time)

    avg_time = sum(all_running_time)/len(all_running_time)
    result_df.loc[len(result_df.index)] = ['running time avg', avg_time, avg_time, avg_time, avg_time]
    
    folder = fr"{os.getcwd()}\results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    result_df.to_csv(fr"{folder}\nb_results.csv", index=False)

def all(df, columns, target, test_size, lr_solver_type='lbfgs', svc_kernel_type='rbf', knn_corr_method='pearson', knn_weights_type='uniform', knn_algorithm_type='auto'):
    random_forest(df, columns, target, test_size)
    decision_tree(df, columns, target, test_size)
    xgboost(df, columns, target, test_size)
    logistic_regression(df, columns, target, test_size, solver_type=lr_solver_type)
    support_vector(df, columns, target, test_size, kernel_type=svc_kernel_type)
    knn(df, columns, target, test_size, corr_method=knn_corr_method, weights_type=knn_weights_type, algorithm_type=knn_algorithm_type)
    naive_bayes(df, columns, target, test_size)

    rf_df = pd.read_csv("rf_results.csv")
    rf_accuracy = rf_df.iloc[2]
    rf_accuracy = rf_accuracy['precision']

    dt_df = pd.read_csv("dt_results.csv")
    dt_accuracy = dt_df.iloc[2]
    dt_accuracy = dt_accuracy['precision']

    xgb_df = pd.read_csv("xgb_results.csv")
    xgb_accuracy = xgb_df.iloc[2]
    xgb_accuracy = xgb_accuracy['precision']

    lr_df = pd.read_csv("lr_results.csv")
    lr_accuracy = lr_df.iloc[2]
    lr_accuracy = lr_accuracy['precision']

    svc_df = pd.read_csv("svc_results.csv")
    svc_accuracy = svc_df.iloc[2]
    svc_accuracy = svc_accuracy['precision']

    knn_df = pd.read_csv("knn_results.csv")
    knn_accuracy = knn_df.iloc[2]
    knn_accuracy = knn_accuracy['precision']

    nb_df = pd.read_csv("nb_results.csv")
    nb_accuracy = nb_df.iloc[2]
    nb_accuracy = nb_accuracy['precision']    

    models = ['Random Forest', 'Decision Tree', 'XGBoost', 'Logistic Regression', 'SVM', 'KNN', 'Naive Bayes']
    accuracies = [rf_accuracy, dt_accuracy, xgb_accuracy, lr_accuracy, svc_accuracy, knn_accuracy, nb_accuracy]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color='blue')
    plt.xlabel('Models')
    plt.ylabel('Precision')
    plt.title('Precision of Different Models')
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{accuracy:.2f}', ha='center', color='white', fontsize=9)

    # Show plot
    plt.show()
    

df = pd.read_csv("Phishing_Legitimate_full.csv")
cols = df.columns.to_list()
cols.remove('id')
cols.remove('CLASS_LABEL')

all(df, cols, 'CLASS_LABEL', 0.2)



