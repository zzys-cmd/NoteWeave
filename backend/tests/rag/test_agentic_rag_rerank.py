"""检索后 rerank 的单测：重排回填、图证据隔离、失败降级。

主链路里 rerank 是增强项——任何异常都要退回原顺序，不能让回答失败。
"""

import pytest

from app.core.background_init import init_manager
from app.core.settings import settings
from app.rag.agentic_rag.schemas import Evidence
from app.rag.agentic_rag.service import AgenticRagService


def _evidence(idx: int, source: str = "knowledge_base", content: str | None = None) -> Evidence:
    return Evidence(
        id=f"e{idx}",
        source=source,
        title=f"t{idx}",
        content=content if content is not None else f"body {idx}",
    )


class FakeReranker:
    """分数取正文末尾的数字，内容越大分数越高，便于断言顺序。"""

    def __init__(self, *args, **kwargs):
        pass

    async def score(self, query, documents, timeout=10.0):
        FakeReranker.last_documents = list(documents)
        return [float(doc.rsplit(" ", 1)[-1]) for doc in documents]


class BrokenReranker:
    def __init__(self, *args, **kwargs):
        pass

    async def score(self, query, documents, timeout=10.0):
        raise RuntimeError("rerank api down")


class MismatchedReranker:
    """返回的分数数量与输入不符，必须视为失败。"""

    def __init__(self, *args, **kwargs):
        pass

    async def score(self, query, documents, timeout=10.0):
        return [0.5]


@pytest.fixture(autouse=True)
def _fast_settings(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_TOP_N", 20)


def _service() -> AgenticRagService:
    """_rerank_evidences 不依赖实例属性，跳过 __init__ 避免构造真实依赖。"""
    return AgenticRagService.__new__(AgenticRagService)


async def test_reorders_text_evidence_and_keeps_graph_slot(monkeypatch):
    monkeypatch.setattr(init_manager, "reorder_service", FakeReranker(), raising=False)
    evidences = [
        _evidence(0, content="body 0"),
        _evidence(1, content="body 9"),
        _evidence(2, source="graph", content="实体描述"),
        _evidence(3, content="body 3"),
    ]

    out = await _service()._rerank_evidences("q", evidences)

    # 非图证据按分数 9 > 3 > 0 回填原槽位，图证据仍停在第 3 位
    assert [e.id for e in out] == ["e1", "e3", "e2", "e0"]
    assert FakeReranker.last_documents == ["body 0", "body 9", "body 3"]


async def test_graph_only_evidences_skip_api(monkeypatch):
    monkeypatch.setattr(init_manager, "reorder_service", BrokenReranker(), raising=False)
    evidences = [
        _evidence(0, source="graph"),
        _evidence(1, source="graph"),
    ]

    out = await _service()._rerank_evidences("q", evidences)

    assert [e.id for e in out] == ["e0", "e1"]


async def test_failure_keeps_original_order(monkeypatch):
    monkeypatch.setattr(init_manager, "reorder_service", BrokenReranker(), raising=False)
    evidences = [_evidence(0), _evidence(1)]

    out = await _service()._rerank_evidences("q", evidences)

    assert [e.id for e in out] == ["e0", "e1"]


async def test_mismatched_score_count_keeps_original_order(monkeypatch):
    monkeypatch.setattr(init_manager, "reorder_service", MismatchedReranker(), raising=False)
    evidences = [_evidence(0), _evidence(1)]

    out = await _service()._rerank_evidences("q", evidences)

    assert [e.id for e in out] == ["e0", "e1"]


async def test_top_n_limits_reranked_head(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_TOP_N", 2)
    monkeypatch.setattr(init_manager, "reorder_service", FakeReranker(), raising=False)
    evidences = [
        _evidence(0, content="body 0"),
        _evidence(1, content="body 9"),
        _evidence(2, content="body 5"),
    ]

    out = await _service()._rerank_evidences("q", evidences)

    # 只有前 2 条进模型（9 > 0），第 3 条不在重排范围内、位置不变
    assert FakeReranker.last_documents == ["body 0", "body 9"]
    assert [e.id for e in out] == ["e1", "e0", "e2"]
