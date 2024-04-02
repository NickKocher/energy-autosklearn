from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import openml
import time


def main():
    ###### Get Dataset
    dataset = openml.datasets.get_dataset(61)

    X, y, *_ = dataset.get_data(target=dataset.default_target_attribute)

    print(X)
    print(y)

    ######

    ##### Build classifiers
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1
    )

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder=f"/home/kocher/energy-autosklearn/energy-measurements/autosklearn_classification_example_tmp_{time.clock_gettime(time.CLOCK_MONOTONIC)}",
        delete_tmp_folder_after_terminate=False,
        memory_limit=16000
    )

    automl.fit(X_train, y_train, dataset_name="iris")
    ######

    ###### Get Results
    print(automl.leaderboard())

    pprint(automl.show_models(), indent=4)

    predictions = automl.predict(X_test)
    print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))

    #######
    return 0


if __name__ == "__main__":
    main()
