from sklearn.metrics import f1_score


def f1(y_true, y_pred, average: str = "macro", positive_label: str = "Work"):
    if average == "binary":
        return f1_score(y_true, y_pred, average="binary", pos_label=positive_label)
    return f1_score(y_true, y_pred, average=average)