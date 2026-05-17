"""
SwingAdvisorBot — Module 3: Risk Management Engine
stubs/user_context_stub.py — DEPRECATED: Now delegates to M5

All methods now delegate to module5_memory.engine.memory_engine.
Kept for backward compatibility only.
"""

from __future__ import annotations

from decimal import Decimal

from module3_risk_engine.models import OpenPosition, RiskTolerance


class UserContextStub:
    """DEPRECATED — delegates to M5 MemoryEngine.

    All TODO-M5 stubs have been replaced.
    This class now proxies to memory_engine for backward compatibility.

    Prefer importing memory_engine directly:
        from module5_memory.engine import memory_engine
    """

    @staticmethod
    def get_user_context(user_id: str = "XCU700") -> dict:
        from module5_memory.engine import memory_engine
        return memory_engine.get_user_context(user_id)

    @staticmethod
    def get_open_positions(user_id: str = "XCU700") -> list[OpenPosition]:
        from module5_memory.engine import memory_engine
        return memory_engine.get_open_positions(user_id)

    @staticmethod
    def get_capital(user_id: str = "XCU700") -> Decimal:
        from module5_memory.engine import memory_engine
        return memory_engine.get_capital(user_id)

    @staticmethod
    def get_risk_tolerance(user_id: str = "XCU700") -> RiskTolerance:
        from module5_memory.engine import memory_engine
        return memory_engine.get_risk_tolerance(user_id)

    @staticmethod
    def get_sector_exposure(user_id: str = "XCU700") -> dict[str, Decimal]:
        from module5_memory.engine import memory_engine
        return memory_engine.get_sector_exposure(user_id)

    @staticmethod
    def get_display_name(user_id: str = "XCU700") -> str:
        from module5_memory.engine import memory_engine
        return memory_engine.get_display_name(user_id)
