#!/usr/bin/env python3
"""Build a Mac-scale HotpotQA-mini retrieval QA dataset.

The output is intentionally small and transparent:

* train.jsonl / eval.jsonl contain QA examples with supporting paragraphs.
* corpus.jsonl is a deduplicated local retrieval corpus.
* summary.json records enough provenance to make paper notes reproducible.

The preferred source is Hugging Face `hotpot_qa` with the `distractor` config
because it exposes paragraphs and supporting facts. A FlashRAG fallback is
included for compatibility with the existing repo preprocessing code, although
FlashRAG examples may only expose `{id, question, golden_answers, metadata}`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def compact_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def ensure_question(text: str) -> str:
    question = compact_space(text)
    if question and question[-1] not in "?!.":  # keep declarative Hotpot questions intact
        question += "?"
    return question


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_answers(example: dict[str, Any]) -> list[str]:
    raw = first_present(
        example,
        ["golden_answers", "answers", "answer", "normalized_answer", "target"],
    )
    if isinstance(raw, dict):
        raw = first_present(raw, ["text", "answer", "answers"])
    answers = [compact_space(item) for item in as_list(raw)]
    return [answer for answer in answers if answer]


def normalize_supporting_facts(raw: Any) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    if raw is None:
        return facts

    if isinstance(raw, dict):
        titles = as_list(first_present(raw, ["title", "titles"]))
        sent_ids = as_list(first_present(raw, ["sent_id", "sent_ids", "sentence_id", "sentence_ids"]))
        for idx, title in enumerate(titles):
            sent_id = sent_ids[idx] if idx < len(sent_ids) else None
            facts.append({"title": compact_space(title), "sent_id": sent_id})
        return [fact for fact in facts if fact["title"]]

    for item in as_list(raw):
        if isinstance(item, dict):
            title = compact_space(first_present(item, ["title", "doc_title", "name"]))
            sent_id = first_present(item, ["sent_id", "sentence_id", "idx", "index"])
        elif isinstance(item, (list, tuple)) and item:
            title = compact_space(item[0])
            sent_id = item[1] if len(item) > 1 else None
        else:
            title = compact_space(item)
            sent_id = None
        if title:
            facts.append({"title": title, "sent_id": sent_id})
    return facts


def support_titles_from(example: dict[str, Any], metadata: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    raw = first_present(
        example,
        ["supporting_facts", "supporting_fact", "support_facts", "golden_facts"],
    )
    if raw is None:
        raw = first_present(
            metadata,
            ["supporting_facts", "supporting_fact", "support_facts", "golden_facts"],
        )
    facts = normalize_supporting_facts(raw)
    titles = sorted({fact["title"] for fact in facts if fact.get("title")})
    return titles, facts


def normalize_sentences(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [compact_space(raw)] if compact_space(raw) else []
    sentences: list[str] = []
    for item in as_list(raw):
        if isinstance(item, (list, tuple)):
            sentences.extend(normalize_sentences(item))
        else:
            sentence = compact_space(item)
            if sentence:
                sentences.append(sentence)
    return sentences


def paragraph_from_dict(item: dict[str, Any]) -> dict[str, Any] | None:
    title = compact_space(
        first_present(item, ["title", "doc_title", "name", "heading", "page_title"])
    )
    raw_text = first_present(item, ["text", "contents", "content", "passage", "paragraph"])
    raw_sentences = first_present(item, ["sentences", "sentence", "sents"])

    if raw_text is None and raw_sentences is not None:
        sentences = normalize_sentences(raw_sentences)
        text = " ".join(sentences)
    else:
        text = compact_space(raw_text)
        if "\n" in str(raw_text or "") and not title:
            first, _, rest = str(raw_text).partition("\n")
            title = compact_space(first)
            text = compact_space(rest)
        sentences = normalize_sentences(raw_sentences)

    if not text and sentences:
        text = " ".join(sentences)
    if not title and text:
        title = compact_space(text[:80])
    if not text:
        return None
    return {"title": title, "text": text, "sentences": sentences}


def paragraphs_from_hotpot_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    titles = as_list(first_present(context, ["title", "titles"]))
    sentence_groups = as_list(first_present(context, ["sentences", "sents"]))
    paragraphs: list[dict[str, Any]] = []
    for idx, title in enumerate(titles):
        sentences = normalize_sentences(sentence_groups[idx] if idx < len(sentence_groups) else [])
        text = " ".join(sentences)
        if text:
            paragraphs.append(
                {"title": compact_space(title), "text": compact_space(text), "sentences": sentences}
            )
    return paragraphs


def normalize_paragraphs(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []

    if isinstance(raw, dict):
        raw_titles = first_present(raw, ["title", "titles"])
        raw_sentence_groups = first_present(raw, ["sentences", "sents"])
        title_is_sequence = isinstance(raw_titles, (list, tuple))
        sentences_are_grouped = isinstance(raw_sentence_groups, (list, tuple)) and (
            not raw_sentence_groups
            or isinstance(raw_sentence_groups[0], (list, tuple))
        )
        if title_is_sequence and raw_sentence_groups is not None and sentences_are_grouped:
            return paragraphs_from_hotpot_context(raw)
        if "title" in raw and ("text" in raw or "contents" in raw or "sentences" in raw):
            maybe = paragraph_from_dict(raw)
            return [maybe] if maybe else []
        paragraphs = []
        for title, text in raw.items():
            if isinstance(text, (dict, list, tuple)):
                maybe = paragraph_from_dict({"title": title, "sentences": text})
            else:
                maybe = paragraph_from_dict({"title": title, "text": text})
            if maybe:
                paragraphs.append(maybe)
        return paragraphs

    paragraphs = []
    for item in as_list(raw):
        maybe = None
        if isinstance(item, dict):
            maybe = paragraph_from_dict(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            maybe = paragraph_from_dict({"title": item[0], "sentences": item[1]})
        elif isinstance(item, str):
            maybe = paragraph_from_dict({"text": item})
        if maybe:
            paragraphs.append(maybe)
    return paragraphs


def extract_paragraphs(example: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    paragraph_keys = [
        "context",
        "paragraphs",
        "docs",
        "documents",
        "passages",
        "evidence",
        "golden_docs",
        "golden_context",
        "supporting_context",
        "retrieval_result",
    ]
    for container in (example, metadata):
        raw = first_present(container, paragraph_keys)
        paragraphs = normalize_paragraphs(raw)
        if paragraphs:
            return paragraphs
    return []


def trim_text(text: str, max_chars: int) -> str:
    text = compact_space(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
    return trimmed or text[:max_chars].strip()


def stable_doc_key(title: str, text: str) -> str:
    return f"{compact_space(title).lower()}\n{compact_space(text).lower()}"


def normalize_record(
    example: dict[str, Any],
    split: str,
    index: int,
    source_name: str,
    max_paragraph_chars: int,
    max_distractors: int,
) -> dict[str, Any] | None:
    metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
    question = ensure_question(first_present(example, ["question", "query", "input"]) or "")
    answers = normalize_answers(example)
    paragraphs = extract_paragraphs(example, metadata)
    supporting_titles, supporting_facts = support_titles_from(example, metadata)
    support_title_set = {title.lower() for title in supporting_titles}

    if not question or not answers:
        return None

    normalized_paragraphs = []
    seen = set()
    for paragraph in paragraphs:
        title = compact_space(paragraph.get("title"))
        text = trim_text(paragraph.get("text", ""), max_paragraph_chars)
        if not text:
            continue
        key = stable_doc_key(title, text)
        if key in seen:
            continue
        seen.add(key)
        is_support = bool(title and title.lower() in support_title_set)
        normalized_paragraphs.append(
            {
                "title": title,
                "text": text,
                "is_support": is_support,
                "sentences": paragraph.get("sentences", []),
            }
        )

    if supporting_titles and normalized_paragraphs:
        support = [p for p in normalized_paragraphs if p["is_support"]]
        distractors = [p for p in normalized_paragraphs if not p["is_support"]]
        normalized_paragraphs = support + distractors[:max_distractors]
    elif normalized_paragraphs:
        normalized_paragraphs = normalized_paragraphs[: max(1, max_distractors)]

    source_id = compact_space(first_present(example, ["id", "_id", "qid"]) or f"{split}-{index}")
    return {
        "id": source_id,
        "source": source_name,
        "split": split,
        "question": question,
        "answer": answers[0],
        "golden_answers": answers,
        "supporting_titles": supporting_titles,
        "supporting_facts": supporting_facts,
        "paragraphs": normalized_paragraphs,
        "metadata": {
            "type": example.get("type") or metadata.get("type"),
            "level": example.get("level") or metadata.get("level"),
            "original_index": index,
        },
    }


def select_records(dataset: Any, size: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    n = len(dataset)
    if size < 0 or size >= n:
        indices = list(range(n))
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(n), size)
    return [(idx, dict(dataset[int(idx)])) for idx in indices]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_corpus(records_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    key_to_doc_id: dict[str, str] = {}
    corpus: list[dict[str, Any]] = []
    for records in records_by_split.values():
        for record in records:
            for paragraph in record.get("paragraphs", []):
                key = stable_doc_key(paragraph.get("title", ""), paragraph.get("text", ""))
                if key not in key_to_doc_id:
                    doc_id = f"hotpotmini_{len(corpus):06d}"
                    key_to_doc_id[key] = doc_id
                    corpus.append(
                        {
                            "id": doc_id,
                            "title": paragraph.get("title", ""),
                            "text": paragraph.get("text", ""),
                            "contents": f"{paragraph.get('title', '')}\n{paragraph.get('text', '')}".strip(),
                        }
                    )
                paragraph["doc_id"] = key_to_doc_id[key]
    return corpus


def split_name(dataset: Any, preferred: str, fallbacks: list[str]) -> str:
    if preferred in dataset:
        return preferred
    for candidate in fallbacks:
        if candidate in dataset:
            return candidate
    available = ", ".join(dataset.keys())
    raise KeyError(f"None of {preferred!r}, {fallbacks!r} found. Available splits: {available}")


def load_source(args: argparse.Namespace) -> tuple[Any, str, str, str]:
    import datasets

    errors = []
    source_order = ["hotpot_qa", "flashrag"] if args.source == "auto" else [args.source]
    for source in source_order:
        try:
            if source == "hotpot_qa":
                dataset = datasets.load_dataset(args.hotpot_dataset, args.hotpot_config)
                train_split = split_name(dataset, args.train_split, ["train"])
                eval_split = split_name(dataset, args.eval_split, ["validation", "dev", "test"])
                source_name = f"{args.hotpot_dataset}/{args.hotpot_config}"
                return dataset, train_split, eval_split, source_name
            if source == "flashrag":
                dataset = datasets.load_dataset(args.flashrag_dataset, args.flashrag_config)
                train_split = split_name(dataset, args.train_split, ["train"])
                eval_split = split_name(dataset, "dev" if args.eval_split == "validation" else args.eval_split, ["dev", "validation", "test"])
                source_name = f"{args.flashrag_dataset}/{args.flashrag_config}"
                return dataset, train_split, eval_split, source_name
            raise ValueError(f"Unknown source: {source}")
        except Exception as exc:  # keep auto fallback helpful
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
            if args.source != "auto":
                raise
    raise RuntimeError("Could not load any HotpotQA source:\n" + "\n".join(errors))


def make_summary(
    args: argparse.Namespace,
    source_name: str,
    train_split: str,
    eval_split: str,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
) -> dict[str, Any]:
    all_records = train_records + eval_records
    paragraph_counts = [len(record.get("paragraphs", [])) for record in all_records]
    support_counts = [
        sum(1 for paragraph in record.get("paragraphs", []) if paragraph.get("is_support"))
        for record in all_records
    ]
    level_counts = Counter(
        compact_space(record.get("metadata", {}).get("level")) or "unknown"
        for record in all_records
    )
    type_counts = Counter(
        compact_space(record.get("metadata", {}).get("type")) or "unknown"
        for record in all_records
    )
    return {
        "source": source_name,
        "train_split": train_split,
        "eval_split": eval_split,
        "train_size": len(train_records),
        "eval_size": len(eval_records),
        "corpus_size": len(corpus),
        "seed": args.seed,
        "max_paragraph_chars": args.max_paragraph_chars,
        "max_distractors": args.max_distractors,
        "avg_paragraphs_per_example": round(statistics.mean(paragraph_counts), 3)
        if paragraph_counts
        else 0.0,
        "examples_with_support_paragraphs": sum(1 for count in support_counts if count > 0),
        "examples_with_two_or_more_support_paragraphs": sum(1 for count in support_counts if count >= 2),
        "level_counts": dict(level_counts),
        "type_counts": dict(type_counts),
        "files": {
            "train": str(Path(args.output_dir) / "train.jsonl"),
            "eval": str(Path(args.output_dir) / "eval.jsonl"),
            "corpus": str(Path(args.output_dir) / "corpus.jsonl"),
            "summary": str(Path(args.output_dir) / "summary.json"),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/hotpotqa_mini")
    parser.add_argument("--source", choices=["auto", "hotpot_qa", "flashrag"], default="auto")
    parser.add_argument("--hotpot-dataset", default="hotpot_qa")
    parser.add_argument("--hotpot-config", default="distractor")
    parser.add_argument("--flashrag-dataset", default="RUC-NLPIR/FlashRAG_datasets")
    parser.add_argument("--flashrag-config", default="hotpotqa")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--eval-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-paragraph-chars", type=int, default=1200)
    parser.add_argument("--max-distractors", type=int, default=6)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    dataset, train_split, eval_split, source_name = load_source(args)

    train_examples = select_records(dataset[train_split], args.train_size, args.seed)
    eval_examples = select_records(dataset[eval_split], args.eval_size, args.seed + 1)

    train_records = [
        record
        for idx, example in train_examples
        if (
            record := normalize_record(
                example,
                split="train",
                index=idx,
                source_name=source_name,
                max_paragraph_chars=args.max_paragraph_chars,
                max_distractors=args.max_distractors,
            )
        )
        is not None
    ]
    eval_records = [
        record
        for idx, example in eval_examples
        if (
            record := normalize_record(
                example,
                split="eval",
                index=idx,
                source_name=source_name,
                max_paragraph_chars=args.max_paragraph_chars,
                max_distractors=args.max_distractors,
            )
        )
        is not None
    ]

    corpus = build_corpus({"train": train_records, "eval": eval_records})
    summary = make_summary(
        args,
        source_name=source_name,
        train_split=train_split,
        eval_split=eval_split,
        train_records=train_records,
        eval_records=eval_records,
        corpus=corpus,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "eval.jsonl", eval_records)
    write_jsonl(output_dir / "corpus.jsonl", corpus)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Built HotpotQA-mini")
    print("=" * 64)
    print(f"source: {source_name}")
    print(f"train:  {len(train_records)} examples from split {train_split!r}")
    print(f"eval:   {len(eval_records)} examples from split {eval_split!r}")
    print(f"corpus: {len(corpus)} unique paragraphs")
    print(
        "support coverage: "
        f"{summary['examples_with_support_paragraphs']}/{len(train_records) + len(eval_records)} "
        "examples have at least one support paragraph"
    )
    print(f"wrote:  {output_dir}")


if __name__ == "__main__":
    main()
