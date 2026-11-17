"""真实 rerank 调用的集成测试：本地配了 RERANKER_* 才跑，CI 上自动跳过。

单元测试用 FakeReranker 验排序/回填/降级逻辑；这里验的是「接口真的通、
模型真能区分相关与不相关」——换成别的 rerank 服务商时，这个测试会先报错。
"""

import pytest

from app.core.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (settings.RERANKER_API_BASE_URL and settings.RERANKER_API_KEY),
        reason="未配置 RERANKER_API_BASE_URL / RERANKER_API_KEY，跳过真实 rerank 调用",
    ),
]

DOCUMENTS = [
    "今天天气不错，适合出门散步。",
    "RAG 的做法是先把文档切片向量化，检索出相关片段再交给模型回答。",
    "向量检索按余弦相似度排序，取 top-k 片段作为上下文。",
]


async def test_real_rerank_scores_relevant_docs_higher():
    from app.rag.reorder_service import ReorderService

    service = ReorderService()
    scores = await service.score("检索结果怎么排序", DOCUMENTS, timeout=20.0)

    assert len(scores) == len(DOCUMENTS)
    assert all(isinstance(score, float) for score in scores)

    # 两条与检索相关的文档应当高于无关的那条（模型行为可能波动，只断言相对关系）
    unrelated = scores[0]
    related = max(scores[1], scores[2])
    assert related > unrelated, f"相关文档得分未高于无关文档: {scores}"


async def test_real_rerank_returns_empty_for_empty_input():
    from app.rag.reorder_service import ReorderService

    service = ReorderService()
    assert await service.score("任意查询", []) == []
