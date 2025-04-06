"""
SVM Training Script

This script trains a Support Vector Machine (SVM) model on quantum data using fidelity kernel.
The script performs the following steps:
1. Specify SVM experimental conditions
2. Load the dataset
3. Shuffle and split the dataset into training and test data
4. Train SVM using OneVsRestClassifier for multi-class classification
5. Make predictions on training and test data
6. Calculate and print accuracy
7. Save the trained model and related data
"""

from dataclasses import asdict

import joblib  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore

from misc import util
from misc.dataset_loader import load_dataset
from svm_util.svm_utility import (
    get_fidelity_kernel_matrix,
    get_svm_parameters,
    select_svm_condition,
)
from svm_util.SVMDictType import SVMParametersDict, SVMSaveDataDict

if __name__ == "__main__":
    # 1) Specify SVM experimental conditions
    condition = select_svm_condition()

    # 2) Load the dataset
    X, y = load_dataset(
        ds_name=condition.ds_name,
        labels=condition.labels,
        n_data_by_label=condition.n_data_by_label,
        fidelity=condition.fidelity,
        n_qubits=condition.n_qubits,
        start_index=condition.start_index,
        use_cache=True,
    )

    # 3) Shuffle the dataset and split into training and test data (using stratify option)
    X, y = util.data_shuffle(
        X=X, y=y, seed=condition.seed
    )  # TODO: Enable seed value. Currently disabled.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=condition.test_size, random_state=condition.seed, stratify=y
    )

    # 4. Train SVM
    ovr_clf = OneVsRestClassifier(
        SVC(
            kernel=get_fidelity_kernel_matrix,
            C=condition.C,
            random_state=condition.seed,
            class_weight=None,
        )
    )
    ovr_clf.fit(X_train, y_train)

    # 5. Prediction
    predict_train = ovr_clf.predict(X_train)
    predict_test = ovr_clf.predict(X_test)

    # 6. Calculate accuracy
    acc_train = accuracy_score(y_train, predict_train)
    acc_test = accuracy_score(y_test, predict_test)
    print("Accuracy (training data):", acc_train)
    print("Accuracy (test data):", acc_test)

    # 7. Save
    datas = []
    for i, clf in enumerate(ovr_clf.estimators_):
        data: SVMParametersDict = get_svm_parameters(clf=clf, X_train=X_train)
        datas.append(data)

    save_data_dict: SVMSaveDataDict = {
        # SVM conditions
        "condition": asdict(condition),
        # SVM results
        "datas": datas,
        # Used dataset
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        # Accuracy
        "acc_train": acc_train,
        "acc_test": acc_test,
        # Predicted values
        "predict_train": predict_train,
        "predict_test": predict_test,
    }

    joblib.dump(save_data_dict, condition.get_save_path())
