"""Black-Box Model Stealing Attack Simulator (AI Heist).
Simulates an adversary querying the deployed ML API and collecting query-response pairs
to reconstruct a stolen training dataset.
"""
import time
import argparse
import logging
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib
import requests

import config
from utils.preprocessing import clean_and_prepare, prepare_dataset_splits
from security.rate_limiter import rate_limiter
from security.suspicious_detection import suspicious_detector
from security.perturbation import apply_output_perturbation
from security.security_logger import security_logger
from attack.state import attack_state_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Attacker")

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _next_run_id(mode: str) -> tuple[str, int]:
    prefix = f"stolen_dataset_{mode}_run"
    existing = list(config.DATA_PROCESSED_DIR.glob(f"{prefix}*.csv"))
    numbers = []
    for path in existing:
        suffix = path.stem.replace(prefix, "")
        if suffix.isdigit():
            numbers.append(int(suffix))
    number = max(numbers, default=0) + 1
    return f"run{number:03d}", number

def _append_experiment_record(record: Dict[str, Any]) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with config.EXPERIMENTS_JSONL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    csv_path = config.EXPERIMENTS_CSV_PATH
    flat = {
        key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
        for key, value in record.items()
    }
    frame = pd.DataFrame([flat])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(csv_path, index=False)

def get_query_pool(num_queries: int) -> pd.DataFrame:
    """Retrieve unlabeled feature vectors for the attacker to send to the API.
    Ensures attacker NEVER sees true labels.
    """
    if not config.ATTACK_QUERY_POOL_PATH.exists():
        logger.info("Attack query pool not found. Generating fresh splits...")
        prepare_dataset_splits(save_splits=True)
        
    df_pool = pd.read_csv(config.ATTACK_QUERY_POOL_PATH)
    if config.TARGET_COLUMN in df_pool.columns:
        df_pool = df_pool.drop(columns=[config.TARGET_COLUMN])
        
    replace = num_queries > len(df_pool)
    sample_df = df_pool.sample(n=num_queries, replace=replace, random_state=config.RANDOM_STATE).reset_index(drop=True)
    return sample_df

def run_attack_simulation(
    query_count: int = config.DEFAULT_ATTACK_QUERIES,
    defended: bool = False,
    progress_callback: Optional[Any] = None,
    base_url: str = "http://127.0.0.1:5000"
) -> Dict[str, Any]:
    """Execute the black-box model extraction attack against the local API.
    
    Args:
        query_count: Number of queries the attacker attempts to send.
        defended: If True, simulates the attack against active API defenses (rate limiting,
                  suspicious detection, and adaptive output perturbation).
        progress_callback: Optional callback(current, total, successful, blocked, perturbed, log_msg)
        
    Returns:
        Dict with attack metrics, output paths, and harvested query counts.
    """
    original_sec_config = json.loads(json.dumps(config.SECURITY_CONFIG))
    effective_config = json.loads(json.dumps(config.SECURITY_CONFIG))
    if not defended:
        effective_config["rate_limiting_enabled"] = False
        effective_config["suspicious_detection_enabled"] = False
        effective_config["perturbation_enabled"] = False
        config.SECURITY_CONFIG.update({
            "rate_limiting_enabled": False,
            "suspicious_detection_enabled": False,
            "perturbation_enabled": False
        })

    query_df = get_query_pool(query_count)
    stolen_records: List[Dict[str, Any]] = []
    
    total = len(query_df)
    successful = 0
    blocked_429 = 0
    blocked_401 = 0
    other_errors = 0
    perturbed_count = 0
    total_server_ms = 0.0
    http_statuses: Dict[str, int] = {}
    client_id = "attacker_key_heist_2026"
    session = requests.Session()
    
    logger.info(f"Initiating {'defended' if defended else 'undefended'} attack simulation ({total} queries)...")
    start_time = time.time()
    
    # Chunking for smooth, responsive UI animation
    # Update progress every chunk
    chunk_size = max(5, total // 25)
    
    rate_limiter.reset()
    suspicious_detector.reset()

    run_id, _ = _next_run_id("defended" if defended else "undefended")
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        for idx in range(total):
            row = query_df.iloc[idx]
            payload = row.to_dict()
            request_start = time.perf_counter()
            try:
                response = session.post(
                    f"{base_url.rstrip('/')}/api/predict",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": "attacker_key_heist_2026"
                    },
                    timeout=15
                )
                status = response.status_code
                total_server_ms += response.elapsed.total_seconds() * 1000.0
                http_statuses[str(status)] = http_statuses.get(str(status), 0) + 1
                response_data = response.json()
            except requests.RequestException as exc:
                status = 0
                other_errors += 1
                response_data = {"status": "error", "message": str(exc)}
                http_statuses["0"] = http_statuses.get("0", 0) + 1

            if status == 429:
                blocked_429 += 1
            elif status == 401:
                blocked_401 += 1
            elif status != 200:
                other_errors += 1

            if status == 200 and response_data.get("status") == "success":
                final_label = int(response_data["prediction"])
                final_prob = float(response_data["probability"])
                defense_applied = bool(response_data.get("security", {}).get("defense_applied", False))
                if defense_applied:
                    perturbed_count += 1
                successful += 1
                record = dict(payload)
                record[config.TARGET_COLUMN] = final_label
                record["_stolen_prob"] = final_prob
                record["_perturbed"] = defense_applied
                record["_http_status"] = status
                stolen_records.append(record)
            
            # Periodic logging and progress emission
            if idx % chunk_size == 0 or idx == total - 1:
                if status == 429:
                    log_msg = f"⛔ Query #{idx+1}: HTTP 429 BLOCKED by rate limiter"
                elif status == 401:
                    log_msg = f"⛔ Query #{idx+1}: HTTP 401 API authentication rejected"
                elif status == 200:
                    log_msg = f"📥 Query #{idx+1}: HTTP 200 harvested label={response_data['prediction']} p={response_data['probability']:.2f}"
                    if response_data.get("security", {}).get("defense_applied"):
                        log_msg += " [PERTURBED]"
                else:
                    log_msg = f"⚠️ Query #{idx+1}: HTTP {status} {response_data.get('message', 'request failed')}"
                    
                if progress_callback:
                    progress_callback(idx + 1, total, successful, blocked_429, perturbed_count, log_msg)
                time.sleep(0.005)
                
    finally:
        config.SECURITY_CONFIG.clear()
        config.SECURITY_CONFIG.update(original_sec_config)

    elapsed_time = round(time.time() - start_time, 2)
    
    # Save stolen dataset to disk
    stolen_df = pd.DataFrame(stolen_records)
    mode = "defended" if defended else "undefended"
    output_path = config.DATA_PROCESSED_DIR / f"stolen_dataset_{mode}_{run_id}.csv"
    
    if not stolen_df.empty:
        cols_to_save = config.ALL_FEATURE_COLUMNS + [config.TARGET_COLUMN]
        stolen_df[cols_to_save].to_csv(output_path, index=False)
        logger.info(f"Saved {len(stolen_df)} stolen records to {output_path}")
        # Keep a clearly labeled compatibility copy for existing copycat training workflows.
        latest_path = config.STOLEN_DATASET_DEFENDED_PATH if defended else config.STOLEN_DATASET_UNDEFENDED_PATH
        stolen_df[cols_to_save].to_csv(latest_path, index=False)

    dataset_hash = _sha256_file(output_path) if output_path.exists() else None
    all_defenses_disabled = (
        defended
        and not effective_config["rate_limiting_enabled"]
        and not effective_config["suspicious_detection_enabled"]
        and not effective_config["perturbation_enabled"]
    )

    result = {
        "run_id": run_id,
        "experiment_timestamp": timestamp,
        "mode": mode,
        "total_queries_sent": total,
        "harvested_samples": len(stolen_df),
        "rate_limited_count": blocked_429,
        "unauthorized_count": blocked_401,
        "other_error_count": max(0, other_errors - blocked_429 - blocked_401),
        "perturbed_count": perturbed_count,
        "elapsed_seconds": elapsed_time,
        "average_http_latency_ms": round(total_server_ms / max(1, total), 2),
        "http_status_counts": http_statuses,
        "output_file": str(output_path),
        "dataset_sha256": dataset_hash,
        "defense_config_snapshot": effective_config,
        "defenses_disabled_warning": bool(all_defenses_disabled),
        "query_seed": config.RANDOM_STATE,
        "copycat_parameters": {
            "architecture": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": config.RANDOM_STATE
        },
        "status": "success" if len(stolen_df) > 0 else "failed"
    }
    _append_experiment_record(result)
    return result

def run_attack_simulation_task(
    query_count: int,
    defended: bool,
    base_url: str = "http://127.0.0.1:5000"
) -> None:
    """Background task executed in a worker thread. Updates attack_state_manager."""
    mode = "defended" if defended else "undefended"
    attack_state_manager.start_attack(query_count, mode)
    
    def on_progress(current, total, successful, blocked, perturbed, log_msg):
        attack_state_manager.update_progress(
            completed=current,
            total=total,
            successful=successful,
            blocked=blocked,
            perturbed=perturbed,
            log_line=log_msg
        )
        
    try:
        result = run_attack_simulation(
            query_count=query_count,
            defended=defended,
            progress_callback=on_progress,
            base_url=base_url
        )
        attack_state_manager.finish_attack(result=result)
    except Exception as e:
        logger.error(f"Attack simulation error: {str(e)}", exc_info=True)
        attack_state_manager.finish_attack(error=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Black-Box Model Stealing Attack")
    parser.add_argument("--queries", type=int, default=1000, help="Number of API queries to send")
    parser.add_argument("--defended", action="store_true", help="Simulate attack against defended API")
    args = parser.parse_args()
    
    res = run_attack_simulation(query_count=args.queries, defended=args.defended)
    print("Attack complete:", res)
