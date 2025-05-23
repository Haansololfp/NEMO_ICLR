import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import datetime
import os
from pathlib import Path


def save_results(results_dict, save_path):
    today = datetime.datetime.now().strftime("%d_%b")
    file_name = os.path.join(save_path, f"results_{today}.pkl")

    with open(file_name, "wb") as fobj:
        pickle.dump(results_dict, fobj)
    return file_name


def classifier_probe_train_val(
    training_data: np.ndarray,
    testing_data: np.ndarray,
    train_index: list,
    model: str = "linear",
    upsample_imbalanced: bool = False,
    learning_rate: float = 1e-4,
    c: float = 1.0,
):

    # sm = SMOTE(random_state=42)
    # training_data, train_index = sm.fit_resample(training_data, train_index)

    # Define the model
    if model == "linear":
        if upsample_imbalanced:
            ros = RandomOverSampler(random_state=42)
            training_data, train_index = ros.fit_resample(training_data, train_index)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, tol=1e-5, class_weight="balanced", C=c),
        )
    else:
        raise ValueError("Model not recognized")

    # Train the model
    clf.fit(training_data, train_index)

    # Make predictions
    predictions = clf.predict(testing_data)
    proba_predictions = clf.predict_proba(testing_data)

    all_predictions = predictions
    all_proba_predictions = proba_predictions

    return all_predictions, all_proba_predictions


def linear_probe_train_val(
    save_folder: str,
    training_data: np.ndarray,
    testing_data: np.ndarray,
    train_label: np.ndarray,
    test_label: np.ndarray,
    labelling: dict,
    c: float = 1.0,
):
    train_index = [labelling[label] for label in train_label]
    test_index = [labelling[label] for label in test_label]
    all_predictions, all_proba_predictions = classifier_probe_train_val(
        training_data,
        testing_data,
        train_index,
        model="linear",
        upsample_imbalanced=False,
        c=c,
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_true_labels = test_index

    # Calculate overall metrics
    overall_accuracy = balanced_accuracy_score(all_true_labels, all_predictions)
    overall_f1 = f1_score(all_true_labels, all_predictions, average="macro")
    overall_cm = confusion_matrix(all_true_labels, all_predictions, normalize="true")

    results_dict = {
        "f1_scores": [overall_f1],
        "accuracy": overall_accuracy,
        "true_targets": all_true_labels,
        "predicted_classes": all_predictions,
    }

    cv_path = save_results(results_dict, save_folder)

    # Plot overall confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        overall_cm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labelling.keys(),
        yticklabels=labelling.keys(),
    )
    plt.title(f"Overall - Accuracy: {overall_accuracy:.2f}, F1 Score: {overall_f1:.2f}")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    # Save the plot, make the name reflect the feature and whether use raw
    save_name = "validation_linear_probe.png"
    img_path = os.path.join(Path(save_folder), save_name)

    plt.savefig(img_path)

    return cv_path, img_path
