"""
Paper Classification Pipeline with Local vLLM
"""

import argparse
import json
import os
import re
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
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.IGNORECASE | re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"(\{.*\})", flags=re.DOTALL)
JSON_SCHEMA_PATTERN = re.compile(
        r'"categories"\s*:\s*\[(?P<categories>.*?)\]\s*,\s*"reason"\s*:\s*"(?P<reason>(?:\\.|[^"\\])*)"',
        flags=re.DOTALL,
)
JSON_STRING_PATTERN = re.compile(r'"((?:\\.|[^"\\])*)"')

TAXONOMY_DEFINITIONS: Dict[str, str] = {
        "Authoritarianism": "Papers about surveillance, state or institutional control, censorship, predictive policing, social scoring, coercive governance, or uses of AI that concentrate power and restrict civil liberties.",
        "Bias & Inequality": "Papers about unfair discrimination, disparate impact, representational harm, inequitable access, marginalization, or how AI systems reproduce or worsen social inequality across groups.",
        "Disempowerment": "Papers about loss of human agency, labor displacement, deskilling, dependence on automated systems, reduced ability to contest decisions, or weakening of individual or community autonomy.",
        "Misinformation": "Papers about false, misleading, or manipulative content, including deepfakes, synthetic media, propaganda, deception, rumor amplification, and the spread or governance of misleading information.",
        "Robustness": "Papers about reliability, safety under failure, adversarial attacks, distribution shift, security, red teaming, evaluation under stress, or whether AI systems behave consistently and safely in real-world conditions.",
        "Extinction Risk": "Papers about catastrophic or existential risks from advanced AI, including loss of control, extreme power-seeking, runaway capabilities, or scenarios where AI could cause irreversible global-scale harm.",
        "Undefined": "Use this only when none of the other taxonomy categories meaningfully apply to the paper's main focus.",
}

TAXONOMY = list(TAXONOMY_DEFINITIONS.keys())

DEFAULT_SYSTEM_PROMPT = (
        "You are a strict research-paper classifier. "
        "Choose one or more categories from the allowed taxonomy and return only a JSON code block. "
        "Use the category definitions carefully and pick all categories that meaningfully apply. "
        "Use 'Undefined' only if none of the other categories apply."
)

DEFAULT_PROMPT_PREFIX = """Classify this paper into one or more categories from this taxonomy.
Use the definitions below to select all categories that apply.

Taxonomy:
{taxonomy_with_definitions}

Instructions:
- Choose one or more categories.
- Use the definitions, not just the category names.
- Include all categories that meaningfully apply to the paper's main focus.
- If no category applies, return ["Undefined"].
- Do not combine "Undefined" with any other category.
- Return exactly one fenced JSON code block and no other text.

Return exactly this structure:
```json
{
"categories": ["list of one or more categories from the taxonomy"],
"reason": "short explanation referencing the definitions for why you chose these categories"
}
```

Paper:
"""

DEFAULT_PROMPT_SUFFIX = "\n\nNow return the fenced JSON code block only."


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
                enable_thinking=False,
        )

        t0 = time.time()
        outputs = model.generate([raw_text], sampling_params, use_tqdm=False)
        elapsed_time = time.time() - t0

        response_text = outputs[0].outputs[0].text
        return response_text.strip()


def load_dataset(csv_path: str) -> pd.DataFrame:
        dataset_extension = os.path.splitext(csv_path)[1].lower()

        if dataset_extension == ".csv":
                return pd.read_csv(csv_path)

        if dataset_extension == ".json":
                with open(csv_path, "r", encoding="utf-8") as file_handle:
                        payload = json.load(file_handle)

                if isinstance(payload, list):
                        return pd.DataFrame(payload)

                if isinstance(payload, dict):
                        if isinstance(payload.get("records"), list):
                                return pd.DataFrame(payload["records"])
                        return pd.DataFrame([payload])

                raise ValueError(f"Unsupported JSON structure in dataset: {csv_path}")

        if dataset_extension == ".jsonl":
                return pd.read_json(csv_path, lines=True)

        raise ValueError(
                f"Unsupported dataset format '{dataset_extension}'. Use a CSV, JSON, or JSONL file."
        )


def format_taxonomy_with_definitions(taxonomy_definitions: Dict[str, str]) -> str:
        return "\n".join(f"- {category}: {definition}" for category, definition in taxonomy_definitions.items())


def render_prompt_prefix(prompt_prefix: str, taxonomy: List[str], taxonomy_definitions: Dict[str, str]) -> str:
        return (
                prompt_prefix
                .replace("{taxonomy}", ", ".join(taxonomy))
                .replace("{taxonomy_with_definitions}", format_taxonomy_with_definitions(taxonomy_definitions))
        )


def build_prompt(title: str, abstract: Any, taxonomy: List[str], taxonomy_definitions: Dict[str, str], prompt_prefix: str, prompt_suffix: str, max_abstract_chars: int) -> str:
        abstract_text = str(abstract) if pd.notna(abstract) else "No abstract available"
        if len(abstract_text) > max_abstract_chars:
                abstract_text = abstract_text[:max_abstract_chars] + "..."

        paper_block = f"Title: {title}\nAbstract: {abstract_text}"
        prefix = render_prompt_prefix(prompt_prefix, taxonomy, taxonomy_definitions)
        return f"{prefix}{paper_block}{prompt_suffix}"


def extract_json_payload(raw_response: str) -> Optional[str]:
        fenced_match = JSON_BLOCK_PATTERN.search(raw_response)
        if fenced_match:
                return fenced_match.group(1).strip()

        brace_match = JSON_OBJECT_PATTERN.search(raw_response)
        if brace_match:
                return brace_match.group(1).strip()

        return None


def extract_structured_fields(raw_response: str) -> Optional[Tuple[List[str], Optional[str]]]:
        json_payload = extract_json_payload(raw_response) or raw_response
        schema_match = JSON_SCHEMA_PATTERN.search(json_payload)
        if not schema_match:
                return None

        categories = [
                json.loads(f'"{match.group(1)}"')
                for match in JSON_STRING_PATTERN.finditer(schema_match.group("categories"))
        ]
        reason = json.loads(f'"{schema_match.group("reason")}"')
        return categories, reason


def parse_model_response(raw_response: str, taxonomy: List[str]) -> Tuple[Optional[List[str]], Optional[str]]:
        raw_response = raw_response.strip()
        json_payload = extract_json_payload(raw_response) or raw_response

        try:
                parsed = json.loads(json_payload)
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
                                valid = list(dict.fromkeys(valid))
                                if "Undefined" in valid and len(valid) > 1:
                                        valid = [cat for cat in valid if cat != "Undefined"]
                                return valid, reason

                category = parsed.get("category")
                if isinstance(category, str):
                        for allowed in taxonomy:
                                if category.strip().lower() == allowed.lower():
                                        return [allowed], reason
        except Exception:
                structured_fields = extract_structured_fields(raw_response)
                if structured_fields:
                        categories, reason = structured_fields
                        valid = []
                        for cat in categories:
                                if isinstance(cat, str):
                                        for allowed in taxonomy:
                                                if cat.strip().lower() == allowed.lower():
                                                        valid.append(allowed)
                                                        break
                        if valid:
                                valid = list(dict.fromkeys(valid))
                                if "Undefined" in valid and len(valid) > 1:
                                        valid = [cat for cat in valid if cat != "Undefined"]
                                return valid, reason

        response_lower = raw_response.lower()
        found = [cat for cat in taxonomy if cat.lower() in response_lower]
        if found:
                found = list(dict.fromkeys(found))
                if "Undefined" in found and len(found) > 1:
                        found = [cat for cat in found if cat != "Undefined"]
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
                                if idx is not None and obj.get("status") != "error":
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
        title_key: str,
        abstract_key: str,
) -> Dict[str, Any]:
        title = row.get(title_key, "")
        abstract = row.get(abstract_key, "")
        prompt = build_prompt(
                title=title,
                abstract=abstract,
                taxonomy=taxonomy,
                taxonomy_definitions=taxonomy_definitions,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                max_abstract_chars=max_abstract_chars,
        )

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
        records_by_index: Dict[int, Dict[str, Any]] = {}
        ordered_records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                        try:
                                record = json.loads(line)
                        except Exception:
                                continue

                        row_index = record.get("row_index")
                        if row_index is None:
                                ordered_records.append(record)
                                continue

                        records_by_index[int(row_index)] = record

        if records_by_index:
                ordered_records.extend(records_by_index[row_index] for row_index in sorted(records_by_index))

        with open(json_path, "w", encoding="utf-8") as f:
                json.dump(ordered_records, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Converted {len(ordered_records)} records to JSON: {json_path}")


def print_prompt_example(
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        processed_indices: set,
        args: argparse.Namespace,
) -> None:
        sample_index: Optional[int] = None
        sample_row: Optional[pd.Series] = None

        for row_index, row in df.iterrows():
                if row_index < args.row_start:
                        continue
                if row_index in processed_indices:
                        continue
                sample_index = row_index
                sample_row = row
                break

        if sample_row is None:
                print("[DEBUG] No unprocessed rows available for prompt preview.")
                return

        prompt = build_prompt(
                title=sample_row.get(args.title_key, ""),
                abstract=sample_row.get(args.abstract_key, ""),
                taxonomy=TAXONOMY,
                taxonomy_definitions=TAXONOMY_DEFINITIONS,
                prompt_prefix=args.prompt_prefix,
                prompt_suffix=args.prompt_suffix,
                max_abstract_chars=args.max_abstract_chars,
        )
        raw_text = tokenizer.apply_chat_template(
                [
                        {"role": "system", "content": args.system},
                        {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
        )

        print("\n" + "=" * 60)
        print(f"[DEBUG] SAMPLE USER PROMPT (Row {sample_index})")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        print(f"[DEBUG] SAMPLE CHAT-FORMATTED PROMPT (Row {sample_index})")
        print("=" * 60)
        print(raw_text)
        print("=" * 60 + "\n")


def process_dataset(args: argparse.Namespace) -> None:
        setup_triton_cache()

        df = load_dataset(args.dataset_path)
        missing_columns = [
                column_name
                for column_name in [args.title_key, args.abstract_key]
                if column_name not in df.columns
        ]
        if missing_columns:
                raise ValueError(
                        "Dataset is missing required columns: "
                        + ", ".join(missing_columns)
                        + f". Available columns: {', '.join(df.columns.astype(str))}"
                )

        os.makedirs(args.output_dir, exist_ok=True)
        jsonl_output = os.path.join(args.output_dir, args.output_file)
        json_base = args.output_file[:-6] if args.output_file.endswith(".jsonl") else args.output_file
        json_output = os.path.join(args.output_dir, json_base + ".json")

        processed_indices = get_processed_indices(jsonl_output)
        print(f"[INFO] Loaded dataset with {len(df)} rows.")
        print(f"[INFO] Resuming with {len(processed_indices)} rows already processed.")

        tokenizer, model = load_model_and_tokenizer(
                args.model_name,
                args.cache_dir,
                args.tensor_parallel_size,
                args.gpu_memory_utilization,
        )

        print_prompt_example(df, tokenizer, processed_indices, args)

        sampling_params = SamplingParams(
                max_tokens=args.max_tokens,
                stop=[tokenizer.eos_token],
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                top_p=args.top_p,
        )

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

        for row_index, row in iterator:
                if row_index < args.row_start:
                        continue
                if row_index in processed_indices:
                        continue

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
                                title_key=args.title_key,
                                abstract_key=args.abstract_key,
                        )
                        records_by_index[row_index] = record
                        save_jsonl_record(jsonl_output, record)

                        if record["status"] == "unknown":
                                unknown += 1

                except Exception as exc:
                        errors += 1
                        error_record = {
                                "row_index": row_index,
                                "title": row.get(args.title_key, ""),
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
        parser.add_argument("--dataset_path", type=str, default="abstracts.csv", help="Input dataset path (.csv, .json, or .jsonl).")
        parser.add_argument("--title_key", type=str, default="title", help="The key/column name for the paper title in the dataset.")
        parser.add_argument("--abstract_key", type=str, default="abstract", help="The key/column name for the paper abstract in the dataset.")
        parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for outputs.")
        parser.add_argument("--output_file", type=str, default="output.jsonl", help="JSONL output file.")
        parser.add_argument("--row_start", type=int, default=0, help="Row index to start from.")
        parser.add_argument("--checkpoint_every", type=int, default=100, help="Print checkpoint every N rows.")
        parser.add_argument("--max_abstract_chars", type=int, default=10000, help="Max abstract chars to include.")

        parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Hugging Face model name.")
        parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for model loading.")
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization.")
        parser.add_argument("--cache_dir", type=str, default="./.cache", help="Directory to cache the model and tokenizer.")
        parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
        parser.add_argument("--max_tokens", type=int, default=1000, help="Max generated tokens.")
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

