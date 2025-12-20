import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_submission(ticket_ids, preds, out_path: str, id_col: str = "TicketID", target_col: str = "TripReason"):
    sub = pd.DataFrame({id_col: ticket_ids, target_col: preds})
    sub.to_csv(out_path, index=False)
    return sub