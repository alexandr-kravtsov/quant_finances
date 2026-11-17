from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd


class MarketDataClient(ABC):
    """
    Универсальный интерфейс источника рыночных данных.

    Реализации:
    - AlgoPack
    - MOEX FAST
    - Binance
    - Polygon
    - BCS
    """


    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m"
    ) -> pd.DataFrame:
        """
        Получить OHLCV данные.
        """

        pass


    @abstractmethod
    def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Получить поток сделок.
        """

        pass


    @abstractmethod
    def get_orderbook(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Получить исторический стакан.
        """

        pass


    @abstractmethod
    def get_tradestats(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        SuperCandles trade statistics.
        """

        pass


    @abstractmethod
    def get_obstats(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        SuperCandles order book statistics.
        """

        pass


    @abstractmethod
    def get_orderstats(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        SuperCandles order statistics.
        """

        pass


    @property
    @abstractmethod
    def name(self) -> str:
        """
        Название источника данных.
        """

        pass


    @abstractmethod
    def health_check(self) -> dict:
        """
        Проверка доступности источника.
        """

        pass