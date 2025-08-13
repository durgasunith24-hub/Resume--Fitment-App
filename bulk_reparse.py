import glob
import os
import re
import time
import sqlite3
import json
from analytics_resume_parser import AnalyticsResumeParser

DB_PATH = os.path.join(os.getcwd(), "resumes_analytics.db")
JSON_DIR = os.path.join(os.getcwd(), "analytics_json")

def extract_index(path):
    m = re.search(r'\((\d+)\)', os.path.basename(path))
    return int(m.group(1)) if m else float('inf')

def load_existing_resume_data(db_path):
    """Returns a dict: {resume_path: full_parsed_json}"""
    if not os.path.exists(db_path):
        return {}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT resume_path, full_parsed_json FROM resumes;")
        rows = cur.fetchall()
        return {r[0]: r[1] for r in rows if r and r[0]}
    except Exception as e:
        print(f"WARNING: Could not load existing paths: {e}")
        return {}
    finally:
        conn.close()

def is_json_empty_or_invalid(json_str):
    """Check if the parsed JSON is empty or has all default fields (invalid)."""
    if not json_str:
        return True
    try:
        data = json.loads(json_str)
        # Consider it invalid if very few fields are populated
        essential_keys = ["person_name", "email", "skills_list", "experience_details"]
        if isinstance(data, dict):
            return all(not data.get(k) for k in essential_keys)
        return True
    except Exception:
        return True

def bulk_parse(folder_path):
    parser = AnalyticsResumeParser()
    print(f"DEBUG: Starting bulk parse in {folder_path}")

    # Gather all PDF/DOCX files
    patterns = [
        os.path.join(folder_path, '*.[pP][dD][fF]'),
        os.path.join(folder_path, '*.[dD][oO][cC][xX]')
    ]
    files = []
    for pat in patterns:
        matched = glob.glob(pat)
        print(f"DEBUG: Pattern '{pat}' → {len(matched)} files")
        files.extend(matched)

    if not files:
        print(f"ERROR: No resumes found in {folder_path}")
        return

    files = sorted(files, key=extract_index)
    existing_data = load_existing_resume_data(DB_PATH)
    print(f"DEBUG: {len(existing_data)} resumes already in DB")

    success, skipped, reparsed, failed = 0, 0, 0, []
    error_details = []

    for path in files:
        json_in_db = existing_data.get(path)

        if json_in_db and not is_json_empty_or_invalid(json_in_db):
            print(f"SKIP: Already valid {os.path.basename(path)}")
            skipped += 1
            continue
        elif json_in_db:
            print(f"REPARSE: In DB but invalid → {os.path.basename(path)}")
        else:
            print(f"NEW: Not in DB → {os.path.basename(path)}")

        start = time.time()
        try:
            rid, parsed, jfile = parser.process_resume(path)
            if rid is None:
                failed.append(os.path.basename(path))
                error_details.append(f"{os.path.basename(path)}: Returned None")
                print(f"FAILED: {os.path.basename(path)} - returned None")
                continue

            elapsed = time.time() - start
            print(f"✅ Parsed: {os.path.basename(path)} → ID {rid} ({elapsed:.1f}s)")

            if json_in_db:
                reparsed += 1
            else:
                success += 1

        except Exception as e:
            failed.append(os.path.basename(path))
            error_details.append(f"{os.path.basename(path)}: {str(e)}")
            print(f"FAILED: {os.path.basename(path)} - {str(e)}")
            continue

    # Final summary
    print(f"\n{'='*60}")
    print(f"BULK PARSE COMPLETE:")
    print(f"  ✅ {success} newly parsed")
    print(f"  🔁 {reparsed} reparsed (due to invalid JSON)")
    print(f"  ⏭️  {skipped} skipped (already valid)")
    print(f"  ❌ {len(failed)} failed")
    print(f"  📁 Total files processed: {len(files)}")

    if failed:
        print(f"\nFAILED FILES DETAILS:")
        for detail in error_details:
            print(f"  • {detail}")

if __name__ == "__main__":
    folder = r"C:\Users\Admin\Desktop\Capstone2\All_Resumes"
    os.makedirs(JSON_DIR, exist_ok=True)
    bulk_parse(folder)
