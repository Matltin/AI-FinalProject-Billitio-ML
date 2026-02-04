from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    random_state: int = 42
    test_size: float = 0.2

    # ستون‌های شناسه/هش که معمولاً ارزش یادگیری ندارند
    drop_cols: tuple = (
        "TicketID",
        "UserID",
        "HashPassportNumber_p",
        "HashEmail",
        "BuyerMobile",
        "NationalCode",
        "VehicleType",
    )

    target_col: str = "TripReason"
    group_col: str = "BillID"