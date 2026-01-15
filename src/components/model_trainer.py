import os
import sys
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
)
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomExecption
from src.logger import logging
from src.utlis import save_object,evaluate_model


from dataclasses import dataclass

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()
    
    def intiate_model_trainer(self,train_array,test_array):
        """
        this function is responsible for data modelling
        """
        try:
             logging.info("split trainning and test input data")
             X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
             models={
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "xgboost":XGBRegressor(),
                "CatBoost":CatBoostRegressor()
                }
             params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "Ridge":{},
                "Lasso":{},
                "KNeighborsRegressor":{},
                "xgboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
             model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,
                                            X_test=X_test,y_test=y_test,models=models,param=params)
             best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict 
             best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
             best_model = models[best_model_name]
             if best_model_score<0.6:
                 raise CustomExecption("No best model found")
             logging.info(f"Best found model on both training and testing dataset")
             save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
             predicted=best_model.predict(X_test)
             r2_square = r2_score(y_test, predicted)
             return r2_square


        
            

        except:
            pass
