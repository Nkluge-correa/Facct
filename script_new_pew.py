"""
Paper Classification Pipeline with Local vLLM
"""

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TRITON_CACHE_CLEANUP_AGE = 3600
VRAM_MB_TO_GB = 1024

TAXONOMY_DEFINITIONS: Dict[str, str] = {
    "Factuality": "The paper addresses the risk that AI systems produce or spread inaccurate, misleading, or fabricated information.",
    "Bias & Fairness": "The paper addresses the risk that AI systems make biased or discriminatory decisions.",
    "Authenticity": "The paper addresses the risk that AI is used to impersonate people or create deceptive synthetic content.",
    "Privacy": "The paper addresses the risk that people's personal information is collected, exposed, or misused by AI systems.",
    "Transparency": "The paper addresses the risk that people cannot understand what AI systems can do or how they work.",
    "Safety": "The paper addresses the risk that AI systems cause serious harm to people.",
    "Autonomy & Control": "The paper addresses the risk that AI systems act independently beyond human oversight.",
    "Trustworthiness": "The paper addresses the general question of whether and when AI systems can be trusted to make important personal decisions.",
    "Other": "The paper does not primarily address any of the above themes, or addresses an AI-related concern not captured by these categories."
}

TAXONOMY = list(TAXONOMY_DEFINITIONS.keys())

DEFAULT_SYSTEM_PROMPT = (
        "You are a strict research-paper classifier. "
        "Choose one or more categories from the allowed taxonomy and return JSON only. "
        "Use the category definitions carefully and pick all categories that meaningfully apply."
)

DEFAULT_PROMPT_PREFIX = """Classify this paper into one or more categories from this taxonomy.
Use the definitions below to select all categories that apply.

Taxonomy:
{taxonomy_with_definitions}

Instructions:
- Choose one or more categories.
- Use the definitions, not just the category names.
- Include all categories that meaningfully apply to the paper's main focus.
- Return valid JSON only.

Return valid JSON with exactly these keys:
{{
    \"categories\": [\"list of one or more categories from the taxonomy\"],
    \"reason\": \"short explanation referencing the definitions\"
}}

Paper:
"""

DEFAULT_PROMPT_SUFFIX = "\n\nNow return the JSON object only."


def setup_triton_cache() -> None:
        cache_dir = os.environ.get('TRITON_CACHE_DIR', './.cache/triton_cache')
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
        cuda_visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0').replace(',', '-')
        rank_cache_dir = f"{cache_dir}/{slurm_job_id}/rank_{cuda_visible_device}"

        os.makedirs(rank_cache_dir, exist_ok=True)
        os.environ['TRITON_CACHE_DIR'] = rank_cache_dir

        try:
                current_time = time.time()
                for root, _, files in os.walk(rank_cache_dir):
                        for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                        if os.path.getmtime(file_path) < current_time - TRITON_CACHE_CLEANUP_AGE:
                                                os.remove(file_path)
                                except (OSError, IOError):
                                        pass
        except Exception:
                pass


def get_nvidia_smi_vram() -> List[float]:
        try:
                result = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
                )
                vram_list = result.decode("utf-8").strip().split("\n")
                return [float(v) / VRAM_MB_TO_GB for v in vram_list]
        except Exception:
                return [0.0]


def load_model_and_tokenizer(model_name: str, cache_dir: str, tensor_parallel_size: int, gpu_memory_utilization: float) -> Tuple[AutoTokenizer, LLM]:
        tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                cache_dir=cache_dir,
        )

        model = LLM(
                model=model_name,
                dtype=torch.float16 if "AWQ" in model_name else torch.bfloat16,
                download_dir=cache_dir,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
        )

        return tokenizer, model


def generate_response(model: LLM, tokenizer: AutoTokenizer, prompt: str, system: str, sampling_params: SamplingParams) -> str:
        raw_text = tokenizer.apply_chat_template(
                [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_reasoning=False,
        )

        t0 = time.time()
        outputs = model.generate([raw_text], sampling_params, use_tqdm=False)
        elapsed_time = time.time() - t0

        response_text = outputs[0].outputs[0].text
        return response_text.strip()


def load_dataset(csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path)


def format_taxonomy_with_definitions(taxonomy_definitions: Dict[str, str]) -> str:
        return "\n".join(f"- {category}: {definition}" for category, definition in taxonomy_definitions.items())


def build_prompt(title: str, abstract: Any, taxonomy: List[str], taxonomy_definitions: Dict[str, str], prompt_prefix: str, prompt_suffix: str, max_abstract_chars: int) -> str:
        abstract_text = str(abstract) if pd.notna(abstract) else "No abstract available"
        if len(abstract_text) > max_abstract_chars:
                abstract_text = abstract_text[:max_abstract_chars] + "..."

        paper_block = f"Title: {title}\nAbstract: {abstract_text}"
        prefix = prompt_prefix.format(
                taxonomy=", ".join(taxonomy),
                taxonomy_with_definitions=format_taxonomy_with_definitions(taxonomy_definitions),
        )
        return f"{prefix}{paper_block}{prompt_suffix}"


def parse_model_response(raw_response: str, taxonomy: List[str]) -> Tuple[Optional[List[str]], Optional[str]]:
        raw_response = raw_response.strip()

        try:
                parsed = json.loads(raw_response)
                reason = parsed.get("reason")

                categories = parsed.get("categories")
                if isinstance(categories, list):
                        valid = []
                        for cat in categories:
                                if isinstance(cat, str):
                                        for allowed in taxonomy:
                                                if cat.strip().lower() == allowed.lower():
                                                        valid.append(allowed)
                                                        break
                        if valid:
                                return valid, reason

                category = parsed.get("category")
                if isinstance(category, str):
                        for allowed in taxonomy:
                                if category.strip().lower() == allowed.lower():
                                        return [allowed], reason
        except Exception:
                pass

        response_lower = raw_response.lower()
        found = [cat for cat in taxonomy if cat.lower() in response_lower]
        if found:
                return found, None

        return None, None


def save_jsonl_record(output_file: str, record: Dict[str, Any]) -> None:
        with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_processed_indices(output_file: str) -> set:
        processed = set()
        if not os.path.exists(output_file):
                return processed

        with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                        try:
                                obj = json.loads(line)
                                idx = obj.get("row_index")
                                if idx is not None:
                                        processed.add(int(idx))
                        except Exception:
                                continue
        return processed


def classify_sample(
        row_index: int,
        row: pd.Series,
        model: LLM,
        tokenizer: AutoTokenizer,
        sampling_params: SamplingParams,
        taxonomy: List[str],
        taxonomy_definitions: Dict[str, str],
        system: str,
        prompt_prefix: str,
        prompt_suffix: str,
        max_abstract_chars: int,
        title_column: str,
        abstract_column: str,
        log_prompt: bool = False
) -> Dict[str, Any]:
        title = row.get(title_column, "")
        abstract = row.get(abstract_column, "")
        prompt = build_prompt(
                title=title,
                abstract=abstract,
                taxonomy=taxonomy,
                taxonomy_definitions=taxonomy_definitions,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                max_abstract_chars=max_abstract_chars,
        )

        if log_prompt:
                print("\n" + "=" * 60)
                print(f"[DEBUG] SAMPLE FORMATTED PROMPT (Row {row_index})")
                print("=" * 60)
                print(prompt)
                print("=" * 60 + "\n")

        started = time.time()
        raw_response = generate_response(model, tokenizer, prompt, system, sampling_params)
        elapsed = time.time() - started

        categories, reason = parse_model_response(raw_response, taxonomy)
        status = "success" if categories else "unknown"

        return {
                "row_index": row_index,
                "title": title,
                "categories": categories,
                "reason": reason,
                "raw_response": raw_response,
                "status": status,
                "elapsed_seconds": round(elapsed, 3),
        }


def convert_jsonl_to_json(jsonl_path: str, json_path: str) -> None:
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                        try:
                                records.append(json.loads(line))
                        except Exception:
                                continue
        with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Converted {len(records)} records to JSON: {json_path}")


def process_dataset(args: argparse.Namespace) -> None:
        setup_triton_cache()

        tokenizer, model = load_model_and_tokenizer(
                args.model_name,
                args.cache_dir,
                args.tensor_parallel_size,
                args.gpu_memory_utilization,
        )

        sampling_params = SamplingParams(
                max_tokens=args.max_tokens,
                stop=[tokenizer.eos_token],
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                top_p=args.top_p,
        )

        df = load_dataset(args.dataset_path)
        os.makedirs(args.output_dir, exist_ok=True)
        jsonl_output = os.path.join(args.output_dir, args.output_file)
        json_base = args.output_file[:-6] if args.output_file.endswith(".jsonl") else args.output_file
        json_output = os.path.join(args.output_dir, json_base + ".json")

        processed_indices = get_processed_indices(jsonl_output)
        print(f"[INFO] Loaded dataset with {len(df)} rows.")
        print(f"[INFO] Resuming with {len(processed_indices)} rows already processed.")

        records_by_index: Dict[int, Dict[str, Any]] = {}
        if os.path.exists(jsonl_output):
                with open(jsonl_output, "r", encoding="utf-8") as f:
                        for line in f:
                                try:
                                        obj = json.loads(line)
                                        idx = int(obj["row_index"])
                                        records_by_index[idx] = obj
                                except Exception:
                                        continue

        errors = 0
        unknown = 0

        iterator = tqdm(df.iterrows(), total=len(df), desc="Classifying")
        logged_first_prompt = False

        for row_index, row in iterator:
                if row_index < args.row_start:
                        continue
                if row_index in processed_indices:
                        continue

                should_log = not logged_first_prompt
                if should_log:
                        logged_first_prompt = True

                try:
                        record = classify_sample(
                                row_index=row_index,
                                row=row,
                                model=model,
                                tokenizer=tokenizer,
                                sampling_params=sampling_params,
                                taxonomy=TAXONOMY,
                                taxonomy_definitions=TAXONOMY_DEFINITIONS,
                                system=args.system,
                                prompt_prefix=args.prompt_prefix,
                                prompt_suffix=args.prompt_suffix,
                                max_abstract_chars=args.max_abstract_chars,
                                title_column=args.title_column,
                                abstract_column=args.abstract_column,
                                log_prompt=should_log
                        )
                        records_by_index[row_index] = record
                        save_jsonl_record(jsonl_output, record)

                        if record["status"] == "unknown":
                                unknown += 1

                except Exception as exc:
                        errors += 1
                        error_record = {
                                "row_index": row_index,
                                "title": row.get(args.title_column, ""),
                                "categories": None,
                                "reason": None,
                                "raw_response": f"Error: {exc}",
                                "status": "error",
                                "elapsed_seconds": None,
                        }
                        records_by_index[row_index] = error_record
                        save_jsonl_record(jsonl_output, error_record)

                if (row_index + 1) % args.checkpoint_every == 0:
                        print(f"[INFO] Checkpoint at row {row_index + 1}")

        convert_jsonl_to_json(jsonl_output, json_output)

        all_categories = [
                cat
                for r in records_by_index.values()
                if r.get("categories")
                for cat in r["categories"]
        ]
        success = sum(1 for r in records_by_index.values() if r.get("status") == "success")

        print("\n" + "=" * 60)
        print("FINAL CLASSIFICATION SUMMARY")
        print("=" * 60)
        if all_categories:
                print(pd.Series(all_categories).value_counts())
        else:
                print("No successful classifications.")
        print(f"\n[INFO] Total rows: {len(df)}")
        print(f"[INFO] Successful classifications: {success}")
        print(f"[INFO] Unknown classifications: {unknown}")
        print(f"[INFO] Errors: {errors}")
        print(f"[INFO] JSONL output: {jsonl_output}")
        print(f"[INFO] JSON output: {json_output}")


def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Classify research papers locally using vLLM.")
        parser.add_argument("--dataset_path", type=str, default="facct_papers_final.csv", help="Input CSV path.")
        parser.add_argument("--title_column", type=str, default="Title", help="Title column name.")
        parser.add_argument("--abstract_column", type=str, default="Abstract", help="Abstract column name.")
        parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for outputs.")
        parser.add_argument("--output_file", type=str, default="facct_results.jsonl", help="JSONL output file.")
        parser.add_argument("--row_start", type=int, default=0, help="Row index to start from.")
        parser.add_argument("--checkpoint_every", type=int, default=25, help="Print checkpoint every N rows.")
        parser.add_argument("--max_abstract_chars", type=int, default=1500, help="Max abstract chars to include.")

        # CHANGED DEFAULT MODEL HERE
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B", help="Hugging Face model name.")
        parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for model loading.")
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization.")
        parser.add_argument("--cache_dir", type=str, default="./.cache", help="Directory to cache the model and tokenizer.")
        parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
        parser.add_argument("--max_tokens", type=int, default=120, help="Max generated tokens.")
        parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")

        parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System instruction.")
        parser.add_argument("--prompt_prefix", type=str, default=DEFAULT_PROMPT_PREFIX, help="Prompt prefix template.")
        parser.add_argument("--prompt_suffix", type=str, default=DEFAULT_PROMPT_SUFFIX, help="Prompt suffix.")
        return parser.parse_args()


def main() -> None:
        args = parse_args()
        print("[INFO] Starting paper classification...")
        print(f"[INFO] Target Model: {args.model_name}")
        print("[INFO] Using taxonomy definitions for disambiguation.")
        process_dataset(args)
        print("[INFO] Classification completed successfully.")

if __name__ == "__main__":
        main()

