"""
Loads all dataset/*.csv files into a single DataContext object.

Usage:
    from data_loader import load_data
    ctx = load_data("dataset")
    ctx.events            # DataFrame, all financial events
    ctx.profile(user_id)  # dict for one user
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

DATE_COLS = {
    "requests.csv": ["request_date", "desired_completion_date"],
    "sample_requests.csv": [
        "request_date",
        "desired_completion_date",
        "earliest_date_for_full_payment",
    ],
    "financial_events.csv": ["event_date", "settlement_date"],
    "exchange_rates.csv": ["rate_date"],
    "request_payment_options.csv": ["first_payment_date"],
    "messages.csv": [],  # sent_at is a full timestamp, parsed separately
}


def _read_csv(path: str, date_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@dataclass
class DataContext:
    requests: pd.DataFrame
    sample_requests: pd.DataFrame
    profiles: pd.DataFrame
    events: pd.DataFrame
    exchange_rates: pd.DataFrame
    payment_options: pd.DataFrame
    messages: pd.DataFrame
    images: pd.DataFrame
    output_template: pd.DataFrame

    _profiles_by_user: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._profiles_by_user = {
            row["user_id"]: row.to_dict() for _, row in self.profiles.iterrows()
        }

    def profile(self, user_id: str) -> dict:
        return self._profiles_by_user[user_id]

    def events_for_user(self, user_id: str) -> pd.DataFrame:
        return self.events[self.events["user_id"] == user_id].copy()

    def payment_options_for_request(self, request_id: str) -> pd.DataFrame:
        return self.payment_options[
            self.payment_options["request_id"] == request_id
        ].copy()

    def messages_for(self, user_id: str, request_id: str | None = None) -> pd.DataFrame:
        m = self.messages[self.messages["user_id"] == user_id]
        if request_id is not None:
            # messages tied to this request OR general (no request_id) for this user
            m = m[(m["request_id"] == request_id) | (m["request_id"].isna())]
        return m.copy()

    def images_for(self, user_id: str, request_id: str | None = None) -> pd.DataFrame:
        im = self.images[self.images["user_id"] == user_id]
        if request_id is not None:
            im = im[(im["request_id"] == request_id) | (im["request_id"].isna())]
        return im.copy()

    def image_for_event(self, event_id: str) -> pd.DataFrame:
        return self.images[self.images["related_event_id"] == event_id].copy()


def load_data(dataset_dir: str = "dataset") -> DataContext:
    def p(name: str) -> str:
        return os.path.join(dataset_dir, name)

    requests = _read_csv(p("requests.csv"), DATE_COLS["requests.csv"])
    sample_requests = _read_csv(
        p("sample_requests.csv"), DATE_COLS["sample_requests.csv"]
    )
    profiles = pd.read_csv(p("financial_profiles.csv"))
    events = _read_csv(p("financial_events.csv"), DATE_COLS["financial_events.csv"])
    exchange_rates = _read_csv(
        p("exchange_rates.csv"), DATE_COLS["exchange_rates.csv"]
    )
    payment_options = _read_csv(
        p("request_payment_options.csv"), DATE_COLS["request_payment_options.csv"]
    )
    messages = pd.read_csv(p("messages.csv"))
    if "sent_at" in messages.columns:
        messages["sent_at"] = pd.to_datetime(messages["sent_at"], errors="coerce")
    images = pd.read_csv(p("images.csv"))
    output_template = pd.read_csv(p("output.csv"))

    return DataContext(
        requests=requests,
        sample_requests=sample_requests,
        profiles=profiles,
        events=events,
        exchange_rates=exchange_rates,
        payment_options=payment_options,
        messages=messages,
        images=images,
        output_template=output_template,
    )


if __name__ == "__main__":
    ctx = load_data("dataset")
    print("requests:", len(ctx.requests))
    print("profiles:", len(ctx.profiles))
    print("events:", len(ctx.events))
    print("sample req 0:", ctx.requests.iloc[0].to_dict())
