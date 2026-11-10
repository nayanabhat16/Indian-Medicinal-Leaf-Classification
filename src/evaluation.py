
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from IPython.display import display


def evaluate_and_save_model(
    model_path,
    model_name,
    test_ds,
    y_true,
    class_names,
    results_folder,
    figures_folder
):

    metrics_file = results_folder / f"{model_name}_metrics.csv"
    pred_file = results_folder / f"{model_name}_predictions.npz"
    report_file = results_folder / f"{model_name}_classification_report.csv"

    # Skip evaluation if results already exist
    if (
        metrics_file.exists()
        and pred_file.exists()
        and report_file.exists()
    ):
        print(model_name, "evaluation already completed.")
        return

    print("Loading model:", model_path)

    model = tf.keras.models.load_model(model_path)

    y_prob = model.predict(test_ds, verbose=1)

    num_classes = len(class_names)

    if num_classes == 2:

        y_prob_for_auc = y_prob.ravel()
        y_pred = (y_prob_for_auc >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_true, y_prob_for_auc)

        precision = precision_score(
            y_true, y_pred, zero_division=0
        )

        recall = recall_score(
            y_true, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )

        np.savez(
            pred_file,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob_for_auc
        )

    else:

        y_pred = np.argmax(y_prob, axis=1)

        roc_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

        precision = precision_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )

        np.savez(
            pred_file,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )

    accuracy = accuracy_score(y_true, y_pred)

    metrics_df = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    }])

    metrics_df.to_csv(metrics_file, index=False)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(report_file)

    print("\nSaved metrics:")
    display(metrics_df)

    print("\nSaved classification report:")
    display(report_df)

    # Clear model from memory
    del model
    tf.keras.backend.clear_session()

    import gc
    gc.collect()
