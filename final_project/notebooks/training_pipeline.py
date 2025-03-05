import h2o
import matplotlib.pyplot as plt
from h2o.automl import H2OAutoML
import pandas as pd

def auto_training_pipeline(
    df: pd.DataFrame,
    target_col: str,
    # --- H2O cluster config ---
    ip: str = "127.0.0.1",
    port: int = 54321,
    nthreads: int = -1,
    max_mem_size: str = "2G",
    max_runtime_secs: int = 300,
    max_models: int = 20,
    sort_metric: str = "AUTO",
    balance_classes: bool = False,
    seed: int = 1234,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    print_metrics: bool = True,
    print_confusion_matrix: bool = False,
    show_scoring_history: bool = True,
    show_variable_importance: bool = True,
    top_k_important_variables: int = 10,
    shutdown_h2o_after_train: bool = False
):
    """
    Automatically trains a classification model using H2OAutoML.
    Meaning, H2O will try several algorithms and return the best model found
    You can also print and plot some insights about the training process.

    Parameters
    ----------
    df:pd.DataFrame
        Input dataset.
    target_col:str
        Target feature.
    ip :str
        IP address for the H2O cluster.
    port: int
        Port for the H2O cluster.
    nthreads: int
        Number of CPU threads to use (-1 to use all).
    max_mem_size: str
        Maximum memory size for the H2O cluster.
    max_runtime_secs: int
        Maximum time to run AutoML.
    max_models : int
        Maximum number of models to try for the training in AutoML.
    sort_metric : str
        Metric used to sort models in the leaderboard.
    balance_classes :bool
        If classes are unbalanced, automatically balance them.
    seed:int
        Random seed for reproducibility.
    train_ratio:float
        Proportion of data used for training (between 0 and 1).
    valid_ratio:float
        Proportion of data used for validation, whatever data remains is used to testing.
    print_metrics: bool
        Wether to print model performance metrics.
    print_confusion_matrix: bool
        Wether to print model confusion matrix.
    show_scoring_history: bool
        Tries to plot the scoring history for the final model if available.
    show_variable_importance: bool
        Tries ploting variable (feature) importance for the best model.
    top_k_important_variables: bool
        If show_variable_importance is true, specify how many top k important variables to plot.
    shutdown_h2o_after_train:bool
        Shuts down the H2O cluster after training is done.

    Returns
    -------
    performance
        The performance metrics object on the test set.
    best_model
        The best model found by H2O AutoML (a.k.a. `leader`).
    automl_leaderboard
        The entire AutoML leaderboard.
    """
    h2o.init(ip=ip, port=port, nthreads=nthreads, max_mem_size=max_mem_size)
    df[target_col] = df[target_col].astype(str)
    col_types = {}
    for c in df.columns:
        if c == target_col:
            col_types[c] = "string"
    h2o_df = h2o.H2OFrame(df, column_types=col_types)

    features = [col for col in h2o_df.columns if col != target_col]
    print('features', features)

    h2o_df[target_col] = h2o_df[target_col].asfactor()

    train, valid, test = h2o_df.split_frame(
        ratios=[train_ratio, valid_ratio],
        seed=seed
    )

    automl = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=max_models,
        sort_metric=sort_metric,
        balance_classes=balance_classes,
        seed=seed
    )
    automl.train(
        x=features,
        y=target_col,
        training_frame=train,
        validation_frame=valid
    )

    best_model = automl.leader
    automl_leaderboard = automl.leaderboard

    performance = best_model.model_performance(test_data=test)

    if print_metrics:
        print("Test Metrics:")
        print(f"mse: {performance.mse()}")
        print(f"R squared: {performance.r2()}")
        if 'AUC' in performance._metric_json:
            print(f"AUC: {performance.auc()}")
        if 'logloss' in performance._metric_json:
            print(f"Logloss: {performance.logloss()}")

        if print_confusion_matrix:
            print("Confusion matrix for test data):")
            print(performance.confusion_matrix())

    if show_scoring_history:
        try:
            scoring_df = best_model.scoring_history()
            if scoring_df is not None:
                if isinstance(scoring_df, h2o.H2OFrame):
                    scoring_df = scoring_df.as_data_frame()
                available_metrics = list(scoring_df.columns)

                possible_pairs = [
                    ("training_mse", "validation_mse"),
                    ("training_rmse", "validation_rmse"),
                    ("training_logloss", "validation_logloss")
                ]
                for train_col, valid_col in possible_pairs:
                    if train_col in available_metrics and valid_col in available_metrics and available_metrics[train_col].shape[0] > 1 and available_metrics[valid_col].shape[0] > 1:
                        plt.figure()
                        plt.plot(scoring_df[train_col], label="Train")
                        plt.plot(scoring_df[valid_col], label="Validation")
                        plt.title(f"{train_col} vs {valid_col} over iterations")
                        plt.xlabel("Iteration")
                        plt.ylabel(train_col.replace("_", " ").title())
                        plt.legend()
                        plt.show()
        except:
            pass

    if show_variable_importance and hasattr(best_model, "varimp"):
        try:
            varimp = best_model.varimp(use_pandas=True)
            if varimp['relative_importance'].sum()>0:
                plt.figure()
                plt.bar(varimp['variable'][:top_k_important_variables], varimp['relative_importance'][:top_k_important_variables])
                plt.title("Variable importance")
                plt.xlabel("Variables")
                plt.ylabel("Relative importance")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.show()
        except:
            pass

    if shutdown_h2o_after_train:
        h2o.shutdown(prompt=False)

    return performance, best_model, automl_leaderboard
