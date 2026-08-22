"""
Dataset Downloader & Sampler for MSMARCO-XI (Hindi & English).
Downloads validation split from HuggingFace (ai4bharat/MSMARCO-XI),
extracts 10,000 Hindi passages and 10,000 English passages,
and creates a held-out calibration query set (in-domain + off-topic queries).
"""

import os
import json
import random
import time
import requests
from pathlib import Path
import pandas as pd

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Off-topic queries for calibration
OFF_TOPIC_HINDI_QUERIES = [
    "आज मुंबई में मौसम कैसा है?",
    "पाई का मान क्या होता है?",
    "मुझे एक अच्छी चाय बनाने की रेसिपी बताओ।",
    "दुनिया का सबसे ऊंचा पर्वत कौन सा है?",
    "पायथन में फॉर लूप कैसे लिखते हैं?",
    "शाहरुख खान की नई फिल्म कौन सी है?",
    "विटामिन सी के क्या फायदे हैं?",
    "फुटबॉल मैच में कितने खिलाड़ी होते हैं?",
    "मंगल ग्रह पर पानी है या नहीं?",
    "कंप्यूटर रिबूट कैसे करते हैं?"
]

OFF_TOPIC_ENGLISH_QUERIES = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "What is the speed of light in vacuum?",
    "Who won the 2022 FIFA World Cup?",
    "How to install PyTorch on Mac?",
    "What is the stock price of Apple today?",
    "How many planets are in the solar system?",
    "What is the formula for pythagorean theorem?",
    "How to tie a necktie step by step?",
    "What are the symptoms of common cold?"
]

def ensure_directories():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    target_path = RAW_DIR / "hinval.parquet"
    if target_path.exists() and target_path.stat().st_size > 100 * 1024 * 1024:
        print(f"✅ Found existing dataset at {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        return str(target_path)
        
    url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
    print(f"📥 Downloading MSMARCO-XI validation dataset from {url}...")
    t0 = time.time()
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB chunks
    downloaded = 0
    
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    if downloaded % (10 * 1024 * 1024) < chunk_size:
                        print(f"  Downloaded {mb_downloaded:.1f}/{mb_total:.1f} MB ({downloaded/total_size*100:.1f}%) in {time.time()-t0:.1f}s")
                        
    print(f"✅ Download completed in {time.time() - t0:.2f}s -> Saved to {target_path}")
    return str(target_path)

def process_and_sample_dataset(parquet_path, target_passages_per_lang=10000, num_calibration_queries=15):
    print("⚙️ Processing dataset and extracting samples...")
    df = pd.read_parquet(parquet_path)
    print(f"📊 Loaded {len(df)} query-passage records from parquet.")
    
    hindi_passages = []
    english_passages = []
    
    in_domain_calibration = []
    processed_queries = 0
    
    for idx, row in df.iterrows():
        query_id = str(row.get("query_id", idx))
        
        # Exact field mapping for MSMARCO-XI
        query_hi = str(row.get("query", "") or "").strip()
        query_en = str(row.get("Eng_Query", "") or "").strip().lstrip(". ")
        
        passages_data = row.get("passages")
        if not isinstance(passages_data, dict):
            continue
            
        p_texts_hi = list(passages_data.get("Translated_passages", []))
        p_texts_en = list(passages_data.get("English_passages", []))
        p_selected = list(passages_data.get("is_selected", [0] * len(p_texts_hi)))
        
        # Extract held-out calibration queries
        if processed_queries < num_calibration_queries and query_hi and query_en:
            selected_passages_hi = [p_texts_hi[i] for i, sel in enumerate(p_selected) if sel == 1]
            selected_passages_en = [p_texts_en[i] for i, sel in enumerate(p_selected) if sel == 1]
            
            in_domain_calibration.append({
                "query_id": f"hi_calib_{query_id}",
                "query_text": query_hi,
                "language": "hi",
                "is_off_topic": False,
                "relevant_contexts": selected_passages_hi if selected_passages_hi else p_texts_hi[:1]
            })
            
            in_domain_calibration.append({
                "query_id": f"en_calib_{query_id}",
                "query_text": query_en,
                "language": "en",
                "is_off_topic": False,
                "relevant_contexts": selected_passages_en if selected_passages_en else p_texts_en[:1]
            })
            processed_queries += 1
        
        # Extract passages
        for p_idx, (p_hi, p_en, sel) in enumerate(zip(p_texts_hi, p_texts_en, p_selected)):
            if p_hi and len(hindi_passages) < target_passages_per_lang:
                hindi_passages.append({
                    "id": f"hi_p_{len(hindi_passages)}",
                    "doc_id": f"{query_id}_{p_idx}",
                    "text": str(p_hi).strip(),
                    "language": "hi",
                    "is_selected": int(sel),
                    "source_query_id": query_id
                })
                
            if p_en and len(english_passages) < target_passages_per_lang:
                english_passages.append({
                    "id": f"en_p_{len(english_passages)}",
                    "doc_id": f"{query_id}_{p_idx}",
                    "text": str(p_en).strip(),
                    "language": "en",
                    "is_selected": int(sel),
                    "source_query_id": query_id
                })
        
        if len(hindi_passages) >= target_passages_per_lang and len(english_passages) >= target_passages_per_lang:
            break

    print(f"✅ Extracted {len(hindi_passages)} Hindi passages.")
    print(f"✅ Extracted {len(english_passages)} English passages.")
    
    # Save passage JSON files
    hi_path = PROCESSED_DIR / "hindi_passages.json"
    en_path = PROCESSED_DIR / "english_passages.json"
    
    with open(hi_path, "w", encoding="utf-8") as f:
        json.dump(hindi_passages, f, ensure_ascii=False, indent=2)
        
    with open(en_path, "w", encoding="utf-8") as f:
        json.dump(english_passages, f, ensure_ascii=False, indent=2)

    # Build calibration dataset
    calibration_queries = in_domain_calibration.copy()
    
    for idx, q in enumerate(OFF_TOPIC_HINDI_QUERIES):
        calibration_queries.append({
            "query_id": f"hi_offtopic_{idx}",
            "query_text": q,
            "language": "hi",
            "is_off_topic": True,
            "relevant_contexts": []
        })
        
    for idx, q in enumerate(OFF_TOPIC_ENGLISH_QUERIES):
        calibration_queries.append({
            "query_id": f"en_offtopic_{idx}",
            "query_text": q,
            "language": "en",
            "is_off_topic": True,
            "relevant_contexts": []
        })
        
    calib_path = PROCESSED_DIR / "calibration_queries.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calibration_queries, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Saved calibration queries ({len(calibration_queries)} total) to {calib_path}")

def main():
    ensure_directories()
    parquet_path = download_dataset()
    process_and_sample_dataset(parquet_path)
    print("\n🎉 Day 0 Dataset Download and Sampling Complete!")

if __name__ == "__main__":
    main()
