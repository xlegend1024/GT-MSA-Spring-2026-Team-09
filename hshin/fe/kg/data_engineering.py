from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI


PREFERRED_CATEGORIES = [
    "Ukraine & Russia",
    "Coronavirus",
    "Tech",
    "Pop-Culture",
    "Business",
    "US-Current-Affairs",
    "Global-Politics",
    "Politics",
]


@dataclass(slots=True)
class LabelConfig:
    batch_size: int = 50
    retry_seconds: int = 60
    max_retries_per_batch: int = 1000
    api_version: str = "2024-12-01-preview"
    request_timeout: float = 120.0
    temperature: float = 0.0


def build_azure_openai_client(
    *,
    api_version: str = "2024-12-01-preview",
    azure_endpoint: str | None = None,
    api_key: str | None = None,
) -> AzureOpenAI:
    load_dotenv()
    endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    key = api_key or os.getenv("AZURE_OPENAI_KEY")

    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set.")
    if not key:
        raise ValueError("AZURE_OPENAI_KEY is not set.")

    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=key)


def _build_prompt(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    few_shots= """ 
                # examples
                question: will-eth-reach-3pt7k-in-july
                category: Crypto
                crypto: ETH
                is_fed: false
                is_global_event: false

                question: bitcoin-above-60000-on-may-10
                category: Crypto
                crypto: BTC
                is_fed: false
                is_global_event: false

                question: will-the-fed-raise-rates-in-june
                category: US-Current-Affairs
                crypto: _
                is_fed: true
                is_global_event: true

                # Return example
                "slug": "bitcoin-above-60000", "category": "Crypto", "crypto": "BTC", "is_fed": false, "is_global_event": false
                """
    system_prompt = (
        "You are a strict labeling assistant for prediction-market events. "
        "Classify each item and return JSON only.\n"
        "Output format: {\"items\": [ ... ] }\n"
        "Each item must contain exactly these keys: "
        "slug, category, crypto, is_fed, is_global_event.\n"
        "Rules:\n"
        f"1) Prefer these categories when possible: {PREFERRED_CATEGORIES}.\n"
        "2) If none fit, choose the best category string based on the slug context.\n"
        "3) crypto: return ticker like BTC/ETH/ADA if crypto-related, otherwise return '_' (underscore).\n"
        "4) is_fed: true only when clearly related to Federal Reserve / FOMC / Fed rates / Fed officials.\n"
        "5) is_global_event: true when event impact/scope is global, otherwise false.\n"
        "6) Keep slug exactly as input.\n"
        "7) Return one output item per input item."
        f"{few_shots.strip()}"
    )

    user_prompt = {
        "items": batch,
        "note": "Return only valid JSON object with key 'items'.",
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]


def _extract_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    slug = str(item.get("slug", "")).strip()
    category = str(item.get("category", "")).strip() or "Unknown"
    crypto = str(item.get("crypto", "_")).strip() or "_"

    is_fed = bool(item.get("is_fed", False))
    is_global_event = bool(item.get("is_global_event", False))

    return {
        "slug": slug,
        "category": category,
        "crypto": crypto,
        "is_fed": is_fed,
        "is_global_event": is_global_event,
    }


def _read_checkpoint_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "slug",
                "category",
                "crypto",
                "is_fed",
                "is_global_event",
            ]
        )

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame(
            columns=[
                "slug",
                "category",
                "crypto",
                "is_fed",
                "is_global_event",
            ]
        )

    out = pd.DataFrame(rows)
    if "slug" in out.columns:
        out["slug"] = out["slug"].astype(str).str.strip()
        out = out.drop_duplicates(subset=["slug"], keep="last")
    return out


def _append_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        for item in items:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def _request_batch_labels(
    client: AzureOpenAI,
    deployment_name: str,
    batch: list[dict[str, Any]],
    config: LabelConfig,
) -> list[dict[str, Any]]:
    messages = _build_prompt(batch)

    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=config.temperature,
        timeout=config.request_timeout,
    )

    content = response.choices[0].message.content or ""
    payload = _extract_json_payload(content)
    items = payload.get("items", [])

    if not isinstance(items, list):
        raise ValueError("LLM response must contain 'items' as a list.")

    normalized = [_normalize_item(x) for x in items]

    input_slugs = [str(b["slug"]).strip() for b in batch]
    out_slugs = [str(x["slug"]).strip() for x in normalized]

    if set(input_slugs) != set(out_slugs):
        missing = sorted(set(input_slugs) - set(out_slugs))
        extras = sorted(set(out_slugs) - set(input_slugs))
        raise ValueError(
            f"Mismatch between input and output slugs. missing={missing[:5]} extras={extras[:5]}"
        )

    ordered = sorted(normalized, key=lambda x: input_slugs.index(str(x["slug"]).strip()))
    return ordered


def label_markets_dataframe(
    markets_df: pd.DataFrame,
    *,
    output_jsonl_path: str | Path,
    deployment_name: str | None = None,
    config: LabelConfig | None = None,
    azure_endpoint: str | None = None,
    azure_api_key: str | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Label markets using Azure OpenAI in batches and save progressive checkpoint to JSONL.

    Required input columns: market_id, slug
    Output columns: market_id, slug, category, crypto, is_fed, is_global_event
    """
    load_dotenv()

    if config is None:
        config = LabelConfig()

    deployment = (deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", "")).strip()
    if not deployment:
        raise ValueError(
            "deployment_name is not provided and AZURE_OPENAI_DEPLOYMENT is not set in .env/environment."
        )

    missing = [c for c in ["market_id", "slug"] if c not in markets_df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    output_path = Path(output_jsonl_path)
    checkpoint_df = _read_checkpoint_jsonl(output_path) if resume else pd.DataFrame()

    done_slugs: set[str] = set()
    if not checkpoint_df.empty and "slug" in checkpoint_df.columns:
        done_slugs = set(checkpoint_df["slug"].astype(str).str.strip().tolist())

    source = markets_df[["market_id", "slug"]].copy()
    source["slug"] = source["slug"].astype(str).str.strip()
    source = source[source["slug"] != ""]
    source = source.drop_duplicates(subset=["market_id"]).reset_index(drop=True)

    slug_source = source[["slug"]].drop_duplicates().reset_index(drop=True)
    pending = slug_source[~slug_source["slug"].isin(done_slugs)].reset_index(drop=True)

    if pending.empty:
        if checkpoint_df.empty:
            return source.assign(category="Unknown", crypto="_", is_fed=False, is_global_event=False)[
                ["market_id", "slug", "category", "crypto", "is_fed", "is_global_event"]
            ]
        merged_done = source.merge(
            checkpoint_df[["slug", "category", "crypto", "is_fed", "is_global_event"]],
            on="slug",
            how="left",
        )
        merged_done["category"] = merged_done["category"].fillna("Unknown")
        merged_done["crypto"] = merged_done["crypto"].fillna("_")
        merged_done["is_fed"] = merged_done["is_fed"].fillna(False)
        merged_done["is_global_event"] = merged_done["is_global_event"].fillna(False)
        return merged_done[["market_id", "slug", "category", "crypto", "is_fed", "is_global_event"]].copy()

    client = build_azure_openai_client(
        api_version=config.api_version,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
    )

    all_new_items: list[dict[str, Any]] = []
    total_pending = len(pending)
    processed = 0

    for start in range(0, len(pending), config.batch_size):
        batch_df = pending.iloc[start : start + config.batch_size].copy()
        batch = batch_df.to_dict(orient="records")

        attempt = 0
        while True:
            try:
                labeled_items = _request_batch_labels(client, deployment, batch, config)
                _append_jsonl(output_path, labeled_items)
                all_new_items.extend(labeled_items)
                processed += len(batch)
                progress_pct = (processed / total_pending) * 100 if total_pending > 0 else 100.0
                print(
                    f"Progress: {processed:,}/{total_pending:,} slugs ({progress_pct:.2f}%)"
                )
                break
            except Exception as exc:
                attempt += 1
                if attempt >= config.max_retries_per_batch:
                    raise RuntimeError(
                        f"Batch failed after {attempt} attempts at offset {start}."
                    ) from exc
                print(
                    f"Batch offset={start} failed (attempt {attempt}). Retrying in {config.retry_seconds}s. Error: {exc}"
                )
                time.sleep(config.retry_seconds)

    all_df = _read_checkpoint_jsonl(output_path)
    merged = source.merge(
        all_df[["slug", "category", "crypto", "is_fed", "is_global_event"]],
        on="slug",
        how="left",
    )
    merged["category"] = merged["category"].fillna("Unknown")
    merged["crypto"] = merged["crypto"].fillna("_")
    merged["is_fed"] = merged["is_fed"].fillna(False)
    merged["is_global_event"] = merged["is_global_event"].fillna(False)
    return merged[["market_id", "slug", "category", "crypto", "is_fed", "is_global_event"]].copy()


def label_markets_from_csv(
    input_csv_path: str | Path,
    *,
    output_jsonl_path: str | Path,
    deployment_name: str | None = None,
    config: LabelConfig | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    source_df = pd.read_csv(input_csv_path)
    return label_markets_dataframe(
        source_df,
        output_jsonl_path=output_jsonl_path,
        deployment_name=deployment_name,
        config=config,
        resume=resume,
    )
