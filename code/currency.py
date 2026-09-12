"""
Currency conversion using exchange_rates.csv.

Confirmed from the data (see notes below):
- Every foreign-currency financial_events row has a settlement_date that
  matches EXACTLY one row in exchange_rates.csv for its currency pair.
- Only a handful of direct pairs exist: EUR->ZAR, EUR->USD, USD->EUR,
  USD->IDR, USD->INR. There is no direct ZAR<->IDR, ZAR<->INR, IDR<->INR
  etc. If such a conversion is ever needed, we route through USD or EUR
  as a hub (two-hop), since USD/EUR connect to everything else.
- Per AGENTS.md 6.1: "use the row for its settlement date and the stated
  from_currency to to_currency direction." So we look up the EXACT date,
  not the nearest one. We also support looking up the reverse pair
  (1 / rate) in case the direction given doesn't match what's in the file
  for a given date.
"""
from __future__ import annotations

import pandas as pd

Rate = float


class CurrencyConverter:
    def __init__(self, exchange_rates: pd.DataFrame):
        # key: (date, from_ccy, to_ccy) -> rate
        self._rates: dict[tuple[pd.Timestamp, str, str], Rate] = {}
        for _, row in exchange_rates.iterrows():
            key = (row["rate_date"], row["from_currency"], row["to_currency"])
            self._rates[key] = float(row["rate"])

    def _direct_rate(self, date: pd.Timestamp, from_ccy: str, to_ccy: str) -> Rate | None:
        if from_ccy == to_ccy:
            return 1.0
        r = self._rates.get((date, from_ccy, to_ccy))
        if r is not None:
            return r
        r = self._rates.get((date, to_ccy, from_ccy))
        if r is not None:
            return 1.0 / r
        return None

    def rate(self, date: pd.Timestamp, from_ccy: str, to_ccy: str) -> Rate:
        """
        Returns the exact-date conversion rate from_ccy -> to_ccy.
        Tries a direct pair (either direction), then a two-hop route
        through USD and EUR as hubs. Raises if nothing works.
        """
        direct = self._direct_rate(date, from_ccy, to_ccy)
        if direct is not None:
            return direct

        for hub in ("USD", "EUR"):
            leg1 = self._direct_rate(date, from_ccy, hub)
            leg2 = self._direct_rate(date, hub, to_ccy)
            if leg1 is not None and leg2 is not None:
                return leg1 * leg2

        raise ValueError(
            f"No exchange rate path found for {from_ccy}->{to_ccy} on {date.date()}"
        )

    def convert(self, amount: float, from_ccy: str, to_ccy: str, date: pd.Timestamp) -> float:
        if pd.isna(amount):
            return amount
        return amount * self.rate(date, from_ccy, to_ccy)
