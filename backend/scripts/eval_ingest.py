"""向 Neo4j 注入 C-MTEB/DuRetrieval 评测语料，供 recall@k 检索评测。

数据源：ModelScope `mteb/duretrieval`（HF 网络不可达，走 modelscope 仓库文件直下）：
- corpus:  100001 条文档 (id, text, title)
- queries: 2000 条问题 (id, text)
- qrels(data): 9839 条 (query-id, corpus-id, score) —— 相关性 ground truth

流程：加载 → 抽样 n_query 条 query + n_doc 篇文档（相关文档全量 + 随机负例）
→ 切片 → 嵌入 → Neo4jGraphStore.upsert_chunks(kind="doc", source_id=doc_id)
→ 落 manifest（query↔doc_id 对照），供 eval_recall.py 复用。

隔离：固定 eval user_id；--clean 用 clear_all_docs 先清空该用户。
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.graph.storage.neo4j_graph_store import Neo4jGraphStore  # noqa: E402
from app.rag.text_spliter import AsyncTextSplitter  # noqa: E402
from app.utils.factory import EmbedModelFactory  # noqa: E402

DATASET_REPO = "mteb/duretrieval"
BASE = f"https://www.modelscope.cn/api/v1/datasets/{DATASET_REPO}/repo?Revision=master&FilePath="
FILES = {
    "corpus": "corpus/dev-00000-of-00001.parquet",
    "queries": "queries/dev-00000-of-00001.parquet",
    "qrels": "data/dev-00000-of-00001.parquet",
}
DEFAULT_USER_ID = "eval-duretrieval"


def ensure_files(dataset_dir: Path) -> dict[str, Path]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for kind, rel in FILES.items():
        p = dataset_dir / f"{kind}.parquet"
        if not p.exists():
            print(f"[data] downloading {kind}: {rel}")
            r = requests.get(BASE + rel, stream=True, timeout=300)
            r.raise_for_status()
            with open(p, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        paths[kind] = p
    return paths


def load_data(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = ensure_files(dataset_dir)
    queries = pd.read_parquet(paths["queries"])
    corpus = pd.read_parquet(paths["corpus"])
    qrels = pd.read_parquet(paths["qrels"])
    return queries, corpus, qrels


def sample(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    qrels: pd.DataFrame,
    n_query: int,
    n_doc: int,
    seed: int,
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    rng = random.Random(seed)
    qids = queries["_id"].tolist()
    sampled_qids = rng.sample(qids, min(n_query, len(qids)))
    qset = set(sampled_qids)

    sub = qrels[qrels["query-id"].isin(qset)]
    relevant: dict[str, list[str]] = {}
    for qid, cid in zip(sub["query-id"], sub["corpus-id"]):
        relevant.setdefault(str(qid), []).append(str(cid))
    relevant_doc_ids = set()
    for ids in relevant.values():
        relevant_doc_ids.update(ids)

    if len(relevant_doc_ids) >= n_doc:
        chosen: set[str] = set(relevant_doc_ids)
        print(f"[sample] relevant docs ({len(chosen)}) >= n_doc ({n_doc}); 不使用负例")
    else:
        neg_pool = [str(c) for c in corpus["_id"].tolist() if str(c) not in relevant_doc_ids]
        n_neg = n_doc - len(relevant_doc_ids)
        neg_ids = rng.sample(neg_pool, min(n_neg, len(neg_pool)))
        chosen = relevant_doc_ids | set(neg_ids)
    return sampled_qids, relevant, sorted(chosen)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", default=DEFAULT_USER_ID)
    ap.add_argument("--n-query", type=int, default=100)
    ap.add_argument("--n-doc", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-dir", type=Path, default=BACKEND / "data" / "eval" / "duretrieval")
    ap.add_argument("--results-dir", type=Path, default=BACKEND / "results" / "eval_retrieval")
    ap.add_argument("--clean", action="store_true", help="先 clear_all_docs 清空 eval 用户")
    ap.add_argument("--chunk-size", type=int, default=None, help="覆盖 document.yaml 的 chunk_size")
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="并发处理的文档数（embedding 与写入并行；1 = 串行，行为与旧版一致）")
    args = ap.parse_args()

    queries, corpus, qrels = load_data(args.dataset_dir)
    print(f"[data] corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    sampled_qids, relevant, doc_ids = sample(
        queries, corpus, qrels, args.n_query, args.n_doc, args.seed
    )
    qtext = dict(zip(queries["_id"].astype(str), queries["text"].astype(str)))
    ctext = dict(zip(corpus["_id"].astype(str), corpus["text"].astype(str)))
    ctitle = dict(zip(corpus["_id"].astype(str), corpus["title"].astype(str)))
    print(f"[sample] queries={len(sampled_qids)} relevant_pairs={sum(len(v) for v in relevant.values())} docs={len(doc_ids)}")

    store = Neo4jGraphStore()
    model = EmbedModelFactory().generator()  # 与生产同款（OpenAI 兼容 /embeddings）
    splitter = AsyncTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap) if args.chunk_size else AsyncTextSplitter()

    if args.clean:
        print(f"[clean] clear_all_docs({args.user_id})")
        await store.clear_all_docs(args.user_id)

    ingest_total = 0
    chunk_total = 0
    skip_empty = 0
    started = time.time()
    sem = asyncio.Semaphore(max(1, args.concurrency))
    progress_lock = asyncio.Lock()

    async def _embed_with_retry(pieces: list[str], retries: int = 3) -> list[list[float]]:
        """限流/网络抖动时退避重试；--concurrency>1 时靠 semaphore 控并发。"""
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                async with sem:
                    return await asyncio.to_thread(model.embed_documents, pieces)
            except Exception as e:  # noqa: BLE001 - 供应商限流形态各异
                last_exc = e
                await asyncio.sleep(1.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    async def _prepare_one(did: str) -> tuple[str, str, list[dict]] | None:
        """切分 + 嵌入，返回待写入的 (doc_id, title, chunks)；空文档返回 None。

        只做 CPU/网络部分，不碰 Neo4j —— 写入由 _flush 批量完成。
        """
        nonlocal skip_empty
        text = ctext.get(did, "")
        if not text or not text.strip():
            async with progress_lock:
                skip_empty += 1
            return None
        pieces = await splitter.split_text(text)
        if not pieces:
            async with progress_lock:
                skip_empty += 1
            return None
        vectors = await _embed_with_retry(pieces)
        chunks = [
            {"chunk_index": idx, "text": piece, "embedding": vectors[idx]}
            for idx, piece in enumerate(pieces)
        ]
        title = ctitle.get(did) or f"doc-{did}"
        return did, title, chunks

    async def _flush(batch: list[tuple[str, str, list[dict]]], done_after: int) -> None:
        """一批文档一次事务写入 Doc 节点 + Chunk 节点。

        逐 doc 调用 upsert_chunks 会产生 2×N 次事务，写 2 万文档时事务提交成为瓶颈
        （实测 ~0.9s/doc）；合并成一次 UNWIND 后降到几十毫秒一批。
        写入语义与 upsert_chunks 一致：复合键 (id, user_id) MERGE，可重复覆盖。
        """
        nonlocal ingest_total, chunk_total
        doc_rows = [
            {"sid": did, "uid": args.user_id, "title": title}
            for did, title, _ in batch
        ]
        chunk_rows: list[dict] = []
        for did, title, chunks in batch:
            for chunk in chunks:
                chunk_rows.append({
                    "id": f"doc:{did}:{chunk['chunk_index']}",
                    "user_id": args.user_id,
                    "kind": "doc",
                    "source_id": did,
                    "source_name": title,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "embedding": chunk["embedding"],
                    "page": None,
                    "image_paths": None,
                })
        await store._run(
            "UNWIND $rows AS row "
            "MERGE (d:Doc {id: row.sid, user_id: row.uid}) "
            "ON CREATE SET d.created_at = timestamp() "
            "SET d.filename = row.title",
            {"rows": doc_rows},
        )
        if chunk_rows:
            await store._run(
                "UNWIND $rows AS row "
                "MERGE (c:Chunk {id: row.id, user_id: row.user_id}) "
                "SET c.kind = row.kind, c.source_id = row.source_id, "
                "c.source_name = row.source_name, c.chunk_index = row.chunk_index, "
                "c.text = row.text, c.embedding = row.embedding, "
                "c.page = row.page, c.image_paths = row.image_paths",
                {"rows": chunk_rows},
            )
        ingest_total += len(batch)
        chunk_total += len(chunk_rows)
        print(
            f"[ingest] {done_after}/{len(doc_ids)} docs ({time.time() - started:.0f}s) "
            f"chunks={chunk_total}",
            flush=True,
        )

    # 分批：每批先并发切分+嵌入，再一次性写库（写入批大小 = concurrency*8）
    batch_size = max(1, args.concurrency * 8)
    for start in range(0, len(doc_ids), batch_size):
        part = doc_ids[start:start + batch_size]
        prepared = await asyncio.gather(*(_prepare_one(d) for d in part))
        batch = [item for item in prepared if item is not None]
        if batch:
            await _flush(batch, start + len(part))

    manifest = {
        "source_dataset": DATASET_REPO,
        "user_id": args.user_id,
        "sampled": {"n_query": len(sampled_qids), "n_doc": len(doc_ids), "seed": args.seed},
        # 仅保留选中的相关对照（qrels 裁剪到该批 query + doc）
        "query_ids": sampled_qids,
        "query_text": {q: qtext.get(q, "") for q in sampled_qids},
        "relevant_docs": {
            q: [d for d in relevant.get(q, []) if d in set(doc_ids)] for q in sampled_qids
        },
        "doc_ids": doc_ids,
        "ingested_doc_count": ingest_total,
        "total_chunks": chunk_total,
        "skipped_empty": skip_empty,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] ingested_docs={ingest_total} chunks={chunk_total} skipped_empty={skip_empty}")
    print(f"[done] manifest -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
