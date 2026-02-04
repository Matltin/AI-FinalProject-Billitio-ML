from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

from .config import Config

TIME_COLS = ["Created", "CancelTime", "DepartureTime"]


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    """Convert to datetime safely (bad values -> NaT)."""
    return pd.to_datetime(s, errors="coerce")


def add_group_features(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    ویژگی‌هایی که به ساختار چند-بلیطی هر سفارش مربوط است.
    - TicketPerOrder: تعداد بلیط در هر BillID
    - family: آیا در یک BillID هم Male=True هم Male=False داریم؟
    """
    out = df.copy()

    if group_col in out.columns and "TicketID" in out.columns:
        out["TicketPerOrder"] = out.groupby(group_col)["TicketID"].transform("count")
    else:
        out["TicketPerOrder"] = 1

    out["family"] = False
    if group_col in out.columns and "Male" in out.columns:
        male_bool = out["Male"].astype("boolean")
        grp = male_bool.groupby(out[group_col]).agg(lambda x: set(x.dropna().astype(bool).tolist()))
        family_billids = grp[grp.apply(lambda s: s == {True, False})].index
        out.loc[out[group_col].isin(family_billids), "family"] = True

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ویژگی‌های زمانی:
    - Departure_Created: فاصله زمان ثبت تا حرکت (روز)
    - DepartureMonth: ماه حرکت
    """
    out = df.copy()

    for c in TIME_COLS:
        if c in out.columns:
            out[c] = _safe_to_datetime(out[c])

    if "CancelTime" in out.columns and "Created" in out.columns:
        out["CancelTime"] = out["CancelTime"].fillna(out["Created"])

    if "DepartureTime" in out.columns and "Created" in out.columns:
        delta = (out["DepartureTime"] - out["Created"]).dt.total_seconds() / (3600 * 24)
        out["Departure_Created"] = delta.clip(lower=0)
        med = out["Departure_Created"].median(skipna=True)
        out["Departure_Created"] = out["Departure_Created"].fillna(med if pd.notna(med) else 0.0)
    else:
        out["Departure_Created"] = 0.0

    if "DepartureTime" in out.columns:
        out["DepartureMonth"] = out["DepartureTime"].dt.month.fillna(0).astype(int)
    else:
        out["DepartureMonth"] = 0

    return out


def add_discount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    پاکسازی CouponDiscount و ساخت Discount و محاسبه Price نهایی.
    """
    out = df.copy()

    if "CouponDiscount" in out.columns:
        out["CouponDiscount"] = pd.to_numeric(out["CouponDiscount"], errors="coerce").fillna(0.0)
        # منفی‌ها را صفر می‌کنیم (برای test). برای train بعداً می‌توانیم حذف کنیم.
        out["CouponDiscount"] = out["CouponDiscount"].clip(lower=0.0)
        out["Discount"] = out["CouponDiscount"] > 0
    else:
        out["CouponDiscount"] = 0.0
        out["Discount"] = False

    if "Price" in out.columns:
        out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
        med = out["Price"].median(skipna=True)
        out["Price"] = out["Price"].fillna(med if pd.notna(med) else 0.0)
        out["Price"] = out["Price"] - out["CouponDiscount"]
    else:
        out["Price"] = 0.0

    return out


def basic_cleaning(df: pd.DataFrame, vehicleclass_mode: Optional[object] = None) -> Tuple[pd.DataFrame, object]:
    """
    - پر کردن VehicleClass با mode
    - تبدیل چند ستون بولی رایج به 0/1
    """
    out = df.copy()

    if "VehicleClass" in out.columns:
        if vehicleclass_mode is None:
            mode = out["VehicleClass"].mode(dropna=True)
            vehicleclass_mode = mode.iloc[0] if len(mode) else 0
        out["VehicleClass"] = out["VehicleClass"].fillna(vehicleclass_mode)

    for b in ["Male", "Domestic", "Cancel"]:
        if b in out.columns:
            if out[b].dtype == object:
                out[b] = out[b].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
            out[b] = out[b].astype("boolean").fillna(False).astype(int)

    return out, vehicleclass_mode


def remove_price_outliers_iqr(train_df: pd.DataFrame, k: float = 10.0) -> Tuple[pd.DataFrame, float, float]:
    """
    حذف پرت‌های Price روی TRAIN فقط.
    bounds را برمی‌گرداند تا روی test کلیپ کنیم.
    """
    df = train_df.copy()
    q1 = df["Price"].quantile(0.25)
    q3 = df["Price"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]
    return df, float(lower), float(upper)


@dataclass
class Preprocessor:
    """
    این کلاس تمام پیش‌پردازش را یک‌جا نگه می‌دارد تا:
    - روی train فقط یک بار fit شود
    - روی test دقیقاً همان transform انجام شود
    """
    cfg: Config

    label_encoder: Optional[LabelEncoder] = None
    ohe_vehicle: Optional[OneHotEncoder] = None
    ord_encoder: Optional[OrdinalEncoder] = None

    vehicleclass_mode: Optional[object] = None
    price_lower: Optional[float] = None
    price_upper: Optional[float] = None

    vehicle_ohe_cols_: Optional[List[str]] = None
    ordinal_cols_: Optional[List[str]] = None
    feature_names_: Optional[List[str]] = None

    def fit_transform(self, df_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        خروجی:
        - X_train (features)
        - y_train (target)
        - groups (BillID) برای group split
        """
        df = df_train.copy()

        if self.cfg.group_col not in df.columns:
            raise ValueError(f"Missing group column: {self.cfg.group_col}")
        groups = df[self.cfg.group_col].copy()

        if self.cfg.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.cfg.target_col}")

        # ---- Feature Engineering
        df = add_group_features(df, self.cfg.group_col)
        df = add_time_features(df)
        df = add_discount_features(df)
        df, self.vehicleclass_mode = basic_cleaning(df, self.vehicleclass_mode)

        # حذف ستون‌های شناسه‌ای/هش (به جز BillID که برای split لازم است)
        drop_cols = [c for c in self.cfg.drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        # حذف ستون‌های datetime اصلی (فیچرهای مشتق‌شده قبلاً ساخته شده‌اند)
        time_drop = [c for c in TIME_COLS if c in df.columns]
        df = df.drop(columns=time_drop, errors="ignore")

        # train: قیمت‌های نامعتبر حذف می‌شوند
        df = df[df["Price"] > 0].copy()

        # train: حذف پرت‌ها
        df, self.price_lower, self.price_upper = remove_price_outliers_iqr(df, k=10.0)

        # ---- Target encoding
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df[self.cfg.target_col].astype(str))
        y = pd.Series(y, index=df.index, name=self.cfg.target_col)

        # ---- Features (بدون هدف)
        df_feat = df.drop(columns=[self.cfg.target_col], errors="ignore")

        # ---- OneHot برای Vehicle
        self.vehicle_ohe_cols_ = []
        if "Vehicle" in df_feat.columns:
            self.ohe_vehicle = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            veh_arr = self.ohe_vehicle.fit_transform(df_feat[["Vehicle"]].astype(str))
            self.vehicle_ohe_cols_ = [f"Vehicle__{c}" for c in self.ohe_vehicle.categories_[0].tolist()]
            df_feat = df_feat.drop(columns=["Vehicle"])
            df_veh = pd.DataFrame(veh_arr, columns=self.vehicle_ohe_cols_, index=df_feat.index)
            df_feat = pd.concat([df_feat, df_veh], axis=1)
        else:
            self.ohe_vehicle = None

        # ---- Ordinal برای سایر ستون‌های دسته‌ای
        cat_cols = df_feat.select_dtypes(include=["object"]).columns.tolist()
        self.ordinal_cols_ = cat_cols
        if len(cat_cols) > 0:
            self.ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df_feat[cat_cols] = self.ord_encoder.fit_transform(df_feat[cat_cols].astype(str))
        else:
            self.ord_encoder = None

        # ---- حذف BillID از features
        if self.cfg.group_col in df_feat.columns:
            df_feat = df_feat.drop(columns=[self.cfg.group_col])

        self.feature_names_ = df_feat.columns.tolist()
        groups = groups.loc[df_feat.index]

        return df_feat.astype(float), y, groups

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing to validation/test."""
        if self.label_encoder is None:
            raise RuntimeError("Preprocessor is not fitted. Run fit_transform() first.")

        df = df_any.copy()

        df = add_group_features(df, self.cfg.group_col)
        df = add_time_features(df)
        df = add_discount_features(df)
        df, _ = basic_cleaning(df, self.vehicleclass_mode)

        drop_cols = [c for c in self.cfg.drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        # حذف ستون‌های datetime اصلی
        time_drop = [c for c in TIME_COLS if c in df.columns]
        df = df.drop(columns=time_drop, errors="ignore")

        # price: clip به جای حذف
        if self.price_lower is not None and self.price_upper is not None:
            df["Price"] = df["Price"].clip(lower=self.price_lower, upper=self.price_upper)
        df["Price"] = df["Price"].clip(lower=0.0)

        if self.cfg.target_col in df.columns:
            df_feat = df.drop(columns=[self.cfg.target_col], errors="ignore")
        else:
            df_feat = df

        # Vehicle OHE
        if "Vehicle" in df_feat.columns and self.ohe_vehicle is not None:
            veh_arr = self.ohe_vehicle.transform(df_feat[["Vehicle"]].astype(str))
            df_feat = df_feat.drop(columns=["Vehicle"])
            df_veh = pd.DataFrame(veh_arr, columns=self.vehicle_ohe_cols_, index=df_feat.index)
            df_feat = pd.concat([df_feat, df_veh], axis=1)
        else:
            # اگر Vehicle نبود، ستون‌های OHE را صفر می‌گذاریم
            for c in (self.vehicle_ohe_cols_ or []):
                if c not in df_feat.columns:
                    df_feat[c] = 0.0

        # Ordinal for categorical columns
        if self.ordinal_cols_ and self.ord_encoder is not None:
            for c in self.ordinal_cols_:
                if c not in df_feat.columns:
                    df_feat[c] = "UNKNOWN"
            df_feat[self.ordinal_cols_] = self.ord_encoder.transform(df_feat[self.ordinal_cols_].astype(str))

        if self.cfg.group_col in df_feat.columns:
            df_feat = df_feat.drop(columns=[self.cfg.group_col])

        # align columns
        for c in (self.feature_names_ or []):
            if c not in df_feat.columns:
                df_feat[c] = 0.0
        df_feat = df_feat[self.feature_names_].copy()

        return df_feat.astype(float)

    def save(self, path: str) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str) -> "Preprocessor":
        return load(path)