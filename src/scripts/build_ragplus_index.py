from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_output_dir(output_dir: str, index_name: str) -> Path:
    root = _project_root()
    base = Path(output_dir) if Path(output_dir).is_absolute() else root / output_dir
    return base / index_name


def build_index(
    file_path: str,
    index_name: str | None,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str,
    embedding_model: str,
    output_dir: str,
    title: str | None,
    entry_type: str | None,
) -> None:
    import utils.time.time as time_utils

    try:
        from ragplus.loaders import load_document
        from ragplus.chunker import chunk_text
        from ragplus.embedder import Embedder
        from ragplus.vectorstore import VectorStore
    except ImportError as e:
        print(f"FAILED: ragplus 导入失败，请确认已激活正确的 venv — {e}")
        sys.exit(1)

    file = Path(file_path)
    if not file.exists():
        print(f"FAILED: 文件不存在 — {file_path}")
        sys.exit(1)

    suffix = file.suffix.lower()
    if suffix not in {".md", ".markdown"}:
        print(f"WARNING: 文件扩展名为 {suffix}，非标准 Markdown 格式，继续处理")

    resolved_index_name = index_name or file.stem
    persist_dir = _resolve_output_dir(output_dir, resolved_index_name)

    try:
        text = load_document(str(file))
    except Exception as e:
        print(f"FAILED: 文件加载失败 — {e}")
        sys.exit(1)

    if not text or not text.strip():
        print("FAILED: 文件内容为空，无法建立索引")
        sys.exit(1)

    try:
        embedder = Embedder(model_name=embedding_model)
    except Exception as e:
        print(f"FAILED: 嵌入模型初始化失败 ({embedding_model}) — {e}")
        sys.exit(1)

    try:
        # semantic strategy requires an embedder instance
        chunks = chunk_text(
            text,
            size=chunk_size,
            overlap=chunk_overlap,
            strategy=chunk_strategy,
            embedder=embedder if chunk_strategy == "semantic" else None,
        )
    except Exception as e:
        print(f"FAILED: 文本分块失败 — {e}")
        sys.exit(1)

    if not chunks:
        print("FAILED: 分块结果为空，请检查文件内容或调整 chunk-size 参数")
        sys.exit(1)

    try:
        embeddings = embedder.encode(chunks)
    except Exception as e:
        print(f"FAILED: 向量编码失败 — {e}")
        sys.exit(1)

    doc_name = file.name
    doc_path = str(file.resolve())

    try:
        store = VectorStore(persist_dir=str(persist_dir))
        store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            doc_id=doc_name,
            metas=[
                {
                    "doc_id": doc_name,
                    "doc_name": doc_name,
                    "doc_path": doc_path,
                    "chunk_index": i,
                    "doc_title": title or file.stem,
                    "doc_type": entry_type or "",
                    "doc_create_time": time_utils.timestamp_full(),
                }
                for i in range(len(chunks))
            ],
        )
        store._save_to_dir()
    except Exception as e:
        print(f"FAILED: 索引写入失败，请检查目标目录权限或磁盘空间 — {e}")
        sys.exit(1)

    rel_path = persist_dir.relative_to(_project_root()) if persist_dir.is_relative_to(_project_root()) else persist_dir
    print(f"SUCCESS: Index built at {rel_path}")
    print(f"  - chunks: {len(chunks)}")
    print(f"  - embedding model: {embedding_model}")
    print(f"  - strategy: {chunk_strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Markdown 文件通过 ragplus 建立本地向量索引"
    )
    parser.add_argument("--file", help="Markdown 文件路径")
    parser.add_argument("--index-name", default=None, help="索引子目录名（默认为文件名去扩展）")
    parser.add_argument("--chunk-size", type=int, default=500, help="每个 chunk 的字符数（默认 500）")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="chunk 间重叠字符数（默认 50）")
    parser.add_argument(
        "--chunk-strategy",
        choices=["fixed", "sentence", "markdown", "heading", "semantic"],
        default="markdown",
        help="分块策略（默认 markdown）",
    )
    parser.add_argument(
        "--embedding-model",
        choices=["minilm", "bge-base", "bge-small", "bge-large", "e5-base", "e5-large"],
        default="bge-base",
        help="嵌入模型（默认 bge-base）",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/runtime",
        help="索引根目录，相对于项目根或绝对路径（默认 tests/runtime）",
    )
    parser.add_argument("--title", default=None, help="文档标题（默认为文件名去扩展）")
    parser.add_argument(
        "--entry-type",
        choices=["背景知识", "工作流程说明", "常用术语", "SOP", "最佳实践", "问题排查手册", "旅行指南"],
        default=None,
        help="知识条目类型（对应 KnowledgeEntryType）",
    )

    args = parser.parse_args()
    build_index(
        file_path=args.file,
        index_name=args.index_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_strategy=args.chunk_strategy,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
        title=args.title,
        entry_type=args.entry_type,
    )


if __name__ == "__main__":
    main()
