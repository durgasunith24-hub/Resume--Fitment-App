# analytics_resume_parser.py

import sqlite3
import json
import os
import re
import time
import PyPDF2
import docx
from langchain_ollama import OllamaLLM
from updated_database_schema import ResumeDatabase

# Directory to store JSON exports
JSON_OUTPUT_DIR = os.path.join(os.getcwd(), "analytics_json")
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

class AnalyticsResumeParser:
    def __init__(self):
        # Use the pulled quantized model
        self.llm = OllamaLLM(model="llama3:8b-instruct-q4_0")
        self.db = ResumeDatabase()

    def extract_text(self, path):
        """Dispatch to PDF or DOCX extractor."""
        if path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(path)
        return self.extract_text_from_docx(path)

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    text += p.extract_text() or ""
        except Exception as e:
            print(f"ERROR: PDF read failed for {pdf_path}: {e}")
        return text

    def extract_text_from_docx(self, docx_path):
        text = ""
        try:
            doc = docx.Document(docx_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"ERROR: DOCX read failed for {docx_path}: {e}")
        return text

    def parse_analytics_with_llama3(self, text):
        """Parse resume via Llama3 into a complete field dict."""
        prompt = (
            "Extract these fields in JSON format:\n"
            "- person_name (string)\n"
            "- current_role (string)\n"
            "- email (string)\n"
            "- total_experience (integer)\n"
            "- current_tenure (integer)\n"
            "- certification_count (integer)\n"
            "- publication_count (integer)\n"
            "- skill_density (integer)\n"
            "- skills_list (array of strings)\n"
            "- experience_details (array of objects)\n"
            "- certifications (array of strings)\n"
            "- publications (array of strings)\n"
            "- education (string)\n\n"
            f"Resume text:\n{text}\n\nReturn only valid JSON."
        )
        print(f"DEBUG: Sending prompt ({len(prompt)} chars) to Llama3")
        resp = self.llm.invoke(prompt)
        print(f"DEBUG: Llama3 returned ({len(resp)} chars)")
        try:
            jstart = resp.index('{')
            jend = resp.rindex('}') + 1
            parsed = json.loads(resp[jstart:jend])
        except Exception as e:
            print(f"ERROR: JSON parsing failed: {e}")
            parsed = {}

        # Build a result dict with defaults for every key
        result = {
            "person_name":        parsed.get("person_name", ""),
            "current_role":       parsed.get("current_role", ""),
            "email":              parsed.get("email", ""),
            "total_experience":   int(parsed.get("total_experience", 0) or 0),
            "current_tenure":     int(parsed.get("current_tenure", 0) or 0),
            "certification_count":int(parsed.get("certification_count", 0) or 0),
            "publication_count":  int(parsed.get("publication_count", 0) or 0),
            "skill_density":      int(parsed.get("skill_density", 0) or 0),
            "skills_list":        parsed.get("skills_list", []),
            "experience_details": parsed.get("experience_details", []),
            "certifications":     parsed.get("certifications", []),
            "publications":       parsed.get("publications", []),
            "education":          parsed.get("education", "")
        }
        # Ensure skill_density at least equals number of skills
        if isinstance(result["skills_list"], list):
            result["skill_density"] = max(result["skill_density"], len(result["skills_list"]))
        return result

    def _export_analytics_json(self, data, src_path):
        base = os.path.splitext(os.path.basename(src_path))[0]
        fname = f"{base}_analytics.json"
        out = os.path.join(JSON_OUTPUT_DIR, fname)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return out

    def process_resume(self, file_path):
        """Full pipeline: extract → parse → save → export."""
        print(f"DEBUG: process_resume start for {file_path}")
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}")
            return None, None, None

        raw = self.extract_text(file_path)
        print(f"DEBUG: Extracted {len(raw)} chars")
        if not raw.strip():
            # Create minimal record if no text extracted
            print(f"WARNING: No text extracted from {file_path}")
            parsed = {
                "person_name": "",
                "current_role": "",
                "email": "",
                "total_experience": 0,
                "current_tenure": 0,
                "certification_count": 0,
                "publication_count": 0,
                "skill_density": 0,
                "skills_list": [],
                "experience_details": [],
                "certifications": [],
                "publications": [],
                "education": "",
                "parsing_error": "No text extracted"
            }
        else:
            parsed = self.parse_analytics_with_llama3(raw)

        print("DEBUG: Parsed JSON ready")

        resume_id = self.db.save_resume_data(
            person_name        = parsed["person_name"],
            resume_path        = file_path,
            current_role       = parsed["current_role"],
            total_experience   = parsed["total_experience"],
            current_tenure     = parsed["current_tenure"],
            certification_count= parsed["certification_count"],
            publication_count  = parsed["publication_count"],
            skill_density      = parsed["skill_density"],
            full_parsed_json   = json.dumps(parsed, indent=2),
            email              = parsed["email"]
        )
        print(f"DEBUG: Saved to DB with ID {resume_id}")

        json_path = self._export_analytics_json(parsed, file_path)
        print(f"DEBUG: Exported JSON to {json_path}")
        return resume_id, parsed, json_path
