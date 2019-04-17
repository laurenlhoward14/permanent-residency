from matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score

def SplitData(X, y):
    '''Undersamples data and splits into training, validation and hold out data

    Arguments
    ---------
    X: feature columns from a dataframe
    y: predictor column from a datatframe

    Returns
    -------
    Data split into 60% Training data, 20% Validation data and 20% Hold Out data for final testing.
    Returns a dictionary of each dataset which can be accessed via key name.
    '''
    # undersample (over sampling is too computationally expensive)
    rus = RandomUnderSampler(random_state=27)
    X_undersample, y_undersample = rus.fit_sample(X,y)

    # Split out 80% for training
    X_training, X_hold_out, y_training, y_hold_out = train_test_split(X_undersample, y_undersample, test_size=.20, random_state=27)

    # Spliting training data into train and val
    X_train, X_val, y_train, y_val= train_test_split(X_training, y_training, test_size=.25, random_state=27)

    labels = ['X_training', 'X_hold_out', 'y_training', 'y_hold_out', 'X_train', 'X_val', 'y_train', 'y_val']
    split_data_list = (X_training, X_hold_out, y_training, y_hold_out, X_train, X_val, y_train, y_val)

    return dict(zip(labels, split_data_list))

################################################################################

def ErrorMetrics(model, model_name, x_train, y_train):
    """Prints all error metrics for model performance.

    Arguments
    ---------
    model: model to evaluate
    model_name: string
        Name of model to print
    x_train / y_train: values in which to train models on

    Returns
    -------
    A print statement for each error metric followed by score rounded to 3.d.p.
    """
    print(f"{model_name}")
    print(f"Accuracy: {(cross_val_score(model, x_train, y_train)).mean().round(3)}")
    print(f"Precision: {(cross_val_score(model, x_train, y_train, scoring='precision')).mean().round(3)}")
    print(f"Recall: {(cross_val_score(model, x_train, y_train, scoring='recall')).mean().round(3)}")
    print(f"F1: {(cross_val_score(model, x_train, y_train, scoring='f1')).mean().round(3)}")

################################################################################

def PrintConfusionMatrix(confusion_matrix, class_names=['Certified', 'Denied']):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label');
    return plt.figure(figsize=(10,7));

################################################################################

def ROC_AUC(model, name, y_val, X_val):
    '''Plots ROC Curve and AUC Score

    Arguments
    ---------
    model: model to plot for

    name: string
        name of model to use as title of plot
    y_val / X_val:
        test values for models

    Returns
    -------
    AUC Score
    ROC Curve as matplotlib.figure
    '''
    fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(X_val)[:,1], pos_label=1)
    plt.plot(fpr, tpr,lw=2)
    plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])


    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve for {name}');
    print("ROC AUC score = ", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))

################################################################################

def StratifiedSample(df, col, n_samples):
    '''Returns a stratified sample from a dataframe dependent on columns

    Arguments
    ---------
    df: pandas.dataframe
        dataframe of data to sample from
    col: column of dataframe
        column in which determines the stratified sampling
    n_samples: integer
        number of each sample size for each category

    Returns
    -------
    Sampled dataframe with equal number of rows for each sample
    '''
    n = min(n_samples, df[col].value_counts().min())
    df_sampled = df.groupby(col).apply(lambda x: x.sample(n))
    df_sampled.index = df_sampled.index.droplevel(0)
    return df_sampled
