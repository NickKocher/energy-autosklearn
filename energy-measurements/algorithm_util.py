from sklearn import linear_model as lm
from sklearn import ensemble as en
from autosklearn.pipeline.components import classification as c_spaces

spaces = {"adaboostc": c_spaces.adaboost.AdaboostClassifier}
algorithm_list = [(en.AdaBoostClassifier, "adaboostc")]
algorithm_backup = [(lm.LinearRegression), lm.Ridge]
