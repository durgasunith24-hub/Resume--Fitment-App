# Required installs:
# pip install streamlit plotly nltk rouge-score sentence-transformers torch pandas python-docx langchain-ollama scikit-learn

import os
import io
import re
import json
import sqlite3
import pandas as pd
import streamlit as st
from functools import reduce
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from docx import Document
from langchain_ollama import OllamaLLM
from analytics_resume_parser import AnalyticsResumeParser
from updated_database_schema import ResumeDatabase

# ----------- Setup -----------

st.set_page_config(page_title="Zero-shot Feature Engineering and Resume Ranking using LLMs", layout="wide")
st.title("Zero-shot Feature Engineering and Resume Ranking using LLMs")

# Database paths (update accordingly)
DB_PATH = r"C:\Users\Admin\Desktop\Capstone2\ALL_PARSER_FILES\LLM_TRY_Reparsed\CompleteLLamaParsedDB.db"
GT_DB = r"C:\Users\Admin\Desktop\Capstone2\ALL_PARSER_FILES\LLM_TRY_Reparsed\Groundtruth.db"
BLEU_ROUG_DB = r"C:\Users\Admin\Desktop\Capstone2\ALL_PARSER_FILES\Bleu_Roug\Blooms.db"
FITMENT_DB_NO_AHP = r"C:\Users\Admin\Desktop\Capstone2\ALL_PARSER_FILES\Bleu_Roug\LLM_Fitment_without_AHP.db"
FITMENT_DB_WITH_AHP = r"C:\Users\Admin\Desktop\Capstone2\ALL_PARSER_FILES\Bleu_Roug\LLM_fitment_with_AHP.db"

# Initialize once
llm = OllamaLLM(model="llama3:8b-instruct-q4_0")
parser = AnalyticsResumeParser()
db = ResumeDatabase(DB_PATH)

# Clear cache button
if st.button("Clear Streamlit Cache"):
    st.cache_data.clear()
    st.experimental_rerun()

# ----------- Utility Functions -----------

@st.cache_data
def load_bleu_rouge_data():
    tables = ["GTCSV", "GPTCSV", "LLamaCSV", "FlanCSV", "NLPCSV"]
    conn = sqlite3.connect(BLEU_ROUG_DB)
    dfs = {t: pd.read_sql_query(f"SELECT * FROM {t}", conn) for t in tables}
    conn.close()
    return dfs, tables

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def fitment_level_color(val):
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "fit":
            return 'background-color: lightgreen'
        elif v == "medium fit":
            return 'background-color: khaki'
        elif v == "low fit":
            return 'background-color: lightcoral'
    return ''

def style_fitment_table(df):
    level_cols = [col for col in df.columns if col.endswith("_Level") or col.endswith("Level") or col == "Fitment Level"]
    return df.style.applymap(fitment_level_color, subset=level_cols)

def extract_number(resume_name):
    match = re.search(r'\((\d+)\)', resume_name)
    if match:
        return int(match.group(1))
    else:
        return float('inf')

def normalize_weights(weights):
    total = sum(weights.values())
    if total == 0:
        return {k: 0 for k in weights}
    return {k: v / total for k, v in weights.items()}

def classify_fit(score):
    if score < 31:
        return "Low Fit"
    elif score < 71:
        return "Medium Fit"
    else:
        return "Fit"

def compute_metrics_vs_gt(y_true, y_pred, labels=["Low Fit", "Medium Fit", "Fit"]):
    label_map = {l: i for i, l in enumerate(labels)}
    y_true_num = [label_map.get(y, -1) for y in y_true]
    y_pred_num = [label_map.get(y, -1) for y in y_pred]

    filtered = [(t, p) for t, p in zip(y_true_num, y_pred_num) if t != -1 and p != -1]
    if not filtered:
        return 0.0, 0.0, 0.0

    y_true_num, y_pred_num = zip(*filtered)
    accuracy = accuracy_score(y_true_num, y_pred_num)
    precision = precision_score(y_true_num, y_pred_num, average='weighted', zero_division=0)
    recall = recall_score(y_true_num, y_pred_num, average='weighted', zero_division=0)
    return accuracy, precision, recall


@st.cache_data(ttl=3600) 
def load_fitment_data(db_path, fitment_tables):
    conn = sqlite3.connect(db_path)
    dfs = []
    for tbl in fitment_tables:
        try:
            df = pd.read_sql_query(
                f'SELECT resume_path, person_name, fitment_level, fitment_score FROM "{tbl}"',
                conn
            )
        except Exception as e:
            st.warning(f"⚠ Skipping missing table: {tbl}")
            continue
        model_prefix = tbl.split("_")[0].replace("CSV", "")
        df.rename(columns={
            "person_name": f"{model_prefix}_Name",
            "fitment_level": f"{model_prefix}_Level",
            "fitment_score": f"{model_prefix}Score"
        }, inplace=True)
        dfs.append(df)
    conn.close()
    if not dfs:
        return pd.DataFrame()
    combined_df = reduce(lambda left, right: pd.merge(left, right, on="resume_path", how="outer"), dfs)
    score_cols = [c for c in combined_df.columns if c.endswith("Score")]
    combined_df[score_cols] = combined_df[score_cols].apply(pd.to_numeric, errors='coerce').round(2)
    return combined_df

def get_gt_name_map():
    conn = sqlite3.connect(GT_DB)
    cur = conn.cursor()
    cur.execute("SELECT resume_path, person_name FROM groundtruth")
    rows = cur.fetchall()
    conn.close()
    return {
        os.path.basename(path.strip()).lower(): name.strip() if name else "Unknown"
        for path, name in rows if path
    }

def get_groundtruth_data():
    conn = sqlite3.connect(GT_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT person_name, resume_path, total_experience, current_tenure,
               certification_count, publication_count, skill_density
        FROM groundtruth
    """)
    rows = cur.fetchall()
    conn.close()

    result = {}
    for row in rows:
        name, resume_path, exp, tenure, cert, pub, skill = row
        resume_key = os.path.basename(resume_path.strip()).lower()
        result[resume_key] = {
            "person_name": name.strip() if name else "Unknown",
            "features": {
                "experience": exp or 0,
                "tenure": tenure or 0,
                "certification": cert or 0,
                "publication": pub or 0,
                "skill": skill or 0
            }
        }
    return result

def get_all_resumes():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT rowid, full_parsed_json, person_name, resume_path FROM resumes")
    rows = cur.fetchall()
    conn.close()
    parsed = []
    for rid, js, name, path in rows:
        try:
            data = json.loads(js)
            data["rowid"] = rid
            data["person_name"] = name
            data["resume_path"] = path
            parsed.append(data)
        except:
            continue
    return parsed

def get_all_jds():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, jd_title, jd_parsed_json FROM job_descriptions ORDER BY added_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

def compute_ahp_fit_score(resume, jd_weights, base_weights):
    norm_base = normalize_weights(base_weights)
    features = {
        "skill": resume.get("skill_density", 0),
        "experience": resume.get("total_experience", 0),
        "tenure": resume.get("current_tenure", 0),
        "publication": resume.get("publication_count", 0),
        "certification": resume.get("certification_count", 0)
    }
    score = 0.0
    for feature, value in features.items():
        jd_weight = jd_weights.get(feature, 1)
        weighted = value * jd_weight * norm_base.get(feature, 0)
        score += weighted
    return round(score, 2)

def extract_docx_text(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    except:
        return ""

def parse_jd_features_with_llm(text):
    prompt = (
        "Extract exactly these 6 fields from the job description below and return as valid JSON:\n"
        "1. skills (list of strings)\n"
        "2. experience (int)\n"
        "3. tenure (int)\n"
        "4. certification (int)\n"
        "5. publication (int)\n"
        "6. skill_density (int = number of skills)\n\n"
        "Return ONLY valid JSON. No explanation. No extra text.\n\n"
        f"JD:\n{text}"
    )
    try:
        response = llm.invoke(prompt).strip()
        return json.loads(response)
    except:
        return None

def upsert_jd_to_db(title, text, parsed):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    skills = parsed.get("skills", [])
    cur.execute("""
        INSERT INTO job_descriptions (
            jd_title, jd_text, jd_parsed_json, skills, experience,
            tenure, certification, publication, skill_density
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(jd_title) DO UPDATE SET
            jd_text=excluded.jd_text,
            jd_parsed_json=excluded.jd_parsed_json,
            skills=excluded.skills,
            experience=excluded.experience,
            tenure=excluded.tenure,
            certification=excluded.certification,
            publication=excluded.publication,
            skill_density=excluded.skill_density
    """, (
        title, text, json.dumps(parsed), ", ".join(skills),
        parsed.get("experience", 0),
        parsed.get("tenure", 0),
        parsed.get("certification", 0),
        parsed.get("publication", 0),
        parsed.get("skill_density", len(skills))
    ))
    conn.commit()
    conn.close()

def build_gt_name_map(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT resume_path, person_name FROM {table_name}")
    rows = cur.fetchall()
    conn.close()
    name_map = {}
    for resume_path, person_name in rows:
        key = os.path.basename(resume_path.strip()).lower()
        name_map[key] = person_name.strip() if person_name else "Unknown"
    return name_map

def load_resumes_with_scores(db_path, jd_weights, base_weights, table_name):
    norm_base = normalize_weights(base_weights)
    norm_jd = normalize_weights(jd_weights)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    query = f"""
        SELECT person_name, resume_path, total_experience, current_tenure,
               certification_count, publication_count, skill_density
        FROM {table_name}
    """
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    max_vals = {
        "experience": max((r[2] or 0 for r in rows), default=1),
        "tenure": max((r[3] or 0 for r in rows), default=1),
        "certification": max((r[4] or 0 for r in rows), default=1),
        "publication": max((r[5] or 0 for r in rows), default=1),
        "skill": max((r[6] or 0 for r in rows), default=1),
    }
    max_score = sum(max_vals[f] * norm_jd[f] * norm_base[f] for f in max_vals)
    max_score = max(max_score, 1)

    result = {}
    for row in rows:
        name, resume_path, exp, tenure, cert, pub, skill = row
        features = {
            "experience": exp or 0,
            "tenure": tenure or 0,
            "certification": cert or 0,
            "publication": pub or 0,
            "skill": skill or 0
        }
        raw_score = sum(features[f] * norm_jd[f] * norm_base[f] for f in features)
        norm_score = round((raw_score / max_score) * 100, 2)
        resume_key = os.path.basename(resume_path.strip()).lower()
        result[resume_key] = {
            "person_name": name,
            "score": norm_score,
            "fitment": classify_fit(norm_score)
        }
    return result

# ----------- Load Bleu and Rouge once -----------
dfs_bleu_rouge, tables_bleu_rouge = load_bleu_rouge_data()
sentence_model = load_sentence_model()

# ----------- Tabs -----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Tab 1: Feature Comparison",
    "Tab 2: Similarity Scores",
    "Tab 3: LLMs Fitment Scoring",
    "Tab 4: LLM with AHP",
    "Tab 5: AHP Fitment Scoring(Live)",
    "Tab 6: Chatbot"
])

# ----------- Tab 1 --------------
with tab1:
    st.header("Feature Extraction and Groundtruth Comparison")

    other_table = st.selectbox("Select Table to Compare with Groundtruth", [t for t in tables_bleu_rouge if t != "GTCSV"], key="tab1_select")

    feature_cols = [c for c in dfs_bleu_rouge["GTCSV"].columns if c not in ["resume_path", "person_name"]]

    df_gt_full = dfs_bleu_rouge["GTCSV"][["resume_path", "person_name"] + feature_cols].copy()
    df_other_full = dfs_bleu_rouge[other_table][["resume_path", "person_name"] + feature_cols].copy()

    df_other_full = df_other_full.set_index("resume_path").reindex(df_gt_full["resume_path"]).reset_index()

    df_gt_display = df_gt_full.drop(columns=["resume_path"])
    df_other_display = df_other_full.drop(columns=["resume_path"])

    def highlight_matches_other_table(row):
        gt_row = df_gt_full.loc[row.name]
        styles = []
        styles.append('background-color: lightgreen' if str(row['person_name']).strip().lower() == str(gt_row['person_name']).strip().lower() else '')
        for feat in feature_cols:
            styles.append('background-color: lightgreen' if str(row[feat]).strip().lower() == str(gt_row[feat]).strip().lower() else '')
        return styles

    col1, col2 = st.columns([4, 4])
    with col1:
        st.subheader("Groundtruth Table")
        st.dataframe(df_gt_display, use_container_width=True)
    with col2:
        st.subheader(f"{other_table} Table")
        st.dataframe(df_other_display.style.apply(highlight_matches_other_table, axis=1), use_container_width=True)

    # Feature-level similarity metrics below tables
    st.subheader("Feature-level Similarity Metrics vs Groundtruth")

    similarity_metrics = []

    for feat in feature_cols:
        gt_vals = df_gt_full[feat]
        other_vals = df_other_full[feat]

        # Try numeric similarity
        try:
            gt_num = pd.to_numeric(gt_vals, errors='coerce')
            other_num = pd.to_numeric(other_vals, errors='coerce')
            if gt_num.notna().all() and other_num.notna().all():
                # Numeric similarity: Pearson correlation or MAE
                corr = gt_num.corr(other_num)
                mae = (gt_num - other_num).abs().mean()
                similarity_metrics.append({
                    "Feature": feat,
                    "Type": "Numeric",
                    "Pearson Corr": corr if corr is not None else 0,
                    "Mean Absolute Error": mae
                })
                continue
        except:
            pass

        # Else treat as categorical/string, compute exact match %
        matches = (gt_vals.astype(str).str.lower() == other_vals.astype(str).str.lower())
        match_pct = matches.mean() * 100
        similarity_metrics.append({
            "Feature": feat,
            "Type": "Categorical/Text",
            "Exact Match %": match_pct
        })

    metrics_df = pd.DataFrame(similarity_metrics)
    st.dataframe(metrics_df)

    # Overall similarity score
    num_corrs = metrics_df.loc[metrics_df['Type'] == 'Numeric', 'Pearson Corr'].dropna()
    cat_match_pct = metrics_df.loc[metrics_df['Type'] == 'Categorical/Text', 'Exact Match %'].dropna()

    overall_score = 0
    if not num_corrs.empty and not cat_match_pct.empty:
        overall_score = (num_corrs.mean() + (cat_match_pct.mean() / 100)) / 2 * 100
    elif not num_corrs.empty:
        overall_score = num_corrs.mean() * 100
    elif not cat_match_pct.empty:
        overall_score = cat_match_pct.mean()

    st.markdown(f"**Overall Similarity Score:** {overall_score:.2f}%")

# ----------- Tab 2 --------------
with tab2:
    st.header("Similarity Scores per Resume (Aggregated)")

    other_table_tab2 = st.selectbox("Select Table to Compare with GTCSV", [t for t in tables_bleu_rouge if t != "GTCSV"], key="tab2_select")

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    df_gt = dfs_bleu_rouge["GTCSV"][["resume_path"] + feature_cols].copy()

    def compute_scores(other_table):
        df_other = dfs_bleu_rouge[other_table][["resume_path"] + feature_cols].copy()
        df_other = df_other.set_index("resume_path").reindex(df_gt["resume_path"]).reset_index()

        results = []
        for idx, row in df_gt.iterrows():
            resume_path = row['resume_path']
            bleu_scores, rouge_scores, semantic_scores = [], [], []

            for feat in feature_cols:
                gt_text = str(row[feat]) if pd.notna(row[feat]) else ""
                other_text = str(df_other.loc[idx, feat]) if pd.notna(df_other.loc[idx, feat]) else ""

                if not gt_text.strip() or not other_text.strip():
                    continue

                try:
                    bleu = sentence_bleu([gt_text.split()], other_text.split(), smoothing_function=smoothie)
                except:
                    bleu = 0.0

                rouge_l_f1 = scorer.score(gt_text, other_text)['rougeL'].fmeasure

                emb1 = sentence_model.encode(gt_text, convert_to_tensor=True)
                emb2 = sentence_model.encode(other_text, convert_to_tensor=True)
                semantic_sim = util.cos_sim(emb1, emb2).item()

                bleu_scores.append(bleu)
                rouge_scores.append(rouge_l_f1)
                semantic_scores.append(semantic_sim)

            avg_bleu = sum(bleu_scores)/len(bleu_scores) if bleu_scores else 0
            avg_rouge = sum(rouge_scores)/len(rouge_scores) if rouge_scores else 0
            avg_semantic = sum(semantic_scores)/len(semantic_scores) if semantic_scores else 0
            combined_score = (avg_bleu + avg_rouge + avg_semantic) / 3

            results.append({
                "resume_path": resume_path,
                "Avg_BLEU": avg_bleu,
                "Avg_ROUGE_L_F1": avg_rouge,
                "Avg_Semantic_Similarity": avg_semantic,
                "Combined_Score": combined_score
            })

        agg_df = pd.DataFrame(results)
        agg_df['resume_num'] = agg_df['resume_path'].apply(extract_number)
        agg_df = agg_df.sort_values(by='resume_num', ascending=True).drop(columns=['resume_num'])
        return agg_df

    agg_df = compute_scores(other_table_tab2)
    st.subheader(f"Aggregated Similarity Scores for {other_table_tab2}")
    st.dataframe(agg_df, use_container_width=True)

    buffer = io.BytesIO()
    agg_df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(f"Download Aggregated Similarity Scores ({other_table_tab2})", data=buffer,
                       file_name=f"aggregated_similarity_scores_{other_table_tab2}.csv", mime="text/csv")

    st.subheader("Average Combined Scores Across All Models")
    avg_scores = {}
    for tbl in [t for t in tables_bleu_rouge if t != "GTCSV"]:
        df_scores = compute_scores(tbl)
        avg_scores[tbl] = df_scores["Combined_Score"].mean().round(2)

    avg_scores_df = pd.DataFrame.from_dict(avg_scores, orient='index', columns=["Average Combined Score"])
    avg_scores_df = avg_scores_df.sort_values(by="Average Combined Score", ascending=False)

    st.table(avg_scores_df.style.format("{:.4f}"))

    st.write("**Interpretation:** Higher average combined score indicates the model's output is closer to the ground truth features (GTCSV). Thus, the model with the highest score is performing best overall.")

# ----------- Tab 3 --------------
with tab3:
    st.header("LLM JD Matching and Fitment Scoring")

    fitment_tables_no_ahp = [
        'GTCSV_fitment', 'FlanCSV_fitment', 'GPTCSV_fitment',
        'LLamaCSV_fitment', 'NLPCSV_fitment'
    ]
    combined_fitment_df = load_fitment_data(FITMENT_DB_NO_AHP, fitment_tables_no_ahp)  # Use no AHP DB here

    if combined_fitment_df.empty:
        st.error("No fitment data found.")
    else:
        display_df = combined_fitment_df.drop(columns=["resume_path"])
        st.subheader("Fitment Table")
        st.dataframe(style_fitment_table(display_df), use_container_width=True)

        st.subheader("Average Scores Gauges")
        score_cols = [c for c in display_df.columns if c.endswith("Score")]
        avg_scores = display_df[score_cols].mean().round(2)
        cols = st.columns(len(avg_scores))

        for i, (model, score) in enumerate(avg_scores.items()):
            color = "red" if score < 50 else "yellow" if score < 80 else "green"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': model.replace("Score", "")},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightcoral'},
                        {'range': [50, 80], 'color': 'khaki'},
                        {'range': [80, 100], 'color': 'lightgreen'}
                    ]
                }
            ))
            cols[i].plotly_chart(fig, use_container_width=True, key=f"tab3_gauge_{model}_{i}")

        if all(score < 50 for score in avg_scores):
            interpretation = (
                "All models and the Ground Truth are in the Low Fit range (<50). "
                "This suggests most resumes in the dataset are poor matches for the job description.\n\n"
                "Current average scores by model: " +
                ", ".join([f"{model}: {score:.2f}" for model, score in avg_scores.items()])
            )
        else:
            interpretation = (
                "Mixed fitment levels across models.\n\n"
                "Current average scores by model: " +
                ", ".join([f"{model}: {score:.2f}" for model, score in avg_scores.items()])
            )
        st.markdown(f"**Interpretation of Gauges**\n\n{interpretation}")

        # Confusion matrices and metrics (2 per row)
        st.subheader("Confusion Matrices & Metrics (Ground Truth vs Models)")
        labels = ["Low Fit", "Medium Fit", "Fit"]

        level_cols = [col for col in display_df.columns if col.endswith("_Level") and not col.startswith("GT")]
        for i in range(0, len(level_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(level_cols[i:i+2]):
                model_name = col_name.replace("_Level", "")
                y_true = display_df["GT_Level"].str.strip()
                y_pred = display_df[col_name].str.strip()

                cm = confusion_matrix(y_true, y_pred, labels=labels)
                fig_cm = ff.create_annotated_heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    showscale=True
                )
                fig_cm.update_layout(title=f"Confusion Matrix: GT vs {model_name}", width=350, height=350)

                with cols[j]:
                    st.plotly_chart(fig_cm, use_container_width=False, key=f"tab3_cm_{model_name}")
                    accuracy, precision, recall = compute_metrics_vs_gt(y_true, y_pred, labels=labels)
                    st.markdown(f"**Metrics for {model_name}:**  Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%}")

        # Calculate percentage match of best (green) model to GT
        import math

        cleaned_avg_scores = {
            k: (v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else 0)
            for k, v in avg_scores.items()
        }

        best_model = max(cleaned_avg_scores, key=cleaned_avg_scores.get)
        best_model_level_col = best_model.replace("Score", "Level")
        if best_model_level_col in display_df.columns:
            y_true = display_df["GT_Level"].str.strip()
            y_pred = display_df[best_model_level_col].str.strip()
            accuracy, precision, recall = compute_metrics_vs_gt(y_true, y_pred, labels=labels)
            st.success(f"🎯 The best model **{best_model.replace('Score','')}** matches Groundtruth with Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")


# ----------- Tab 4 --------------
with tab4:
    st.header("JD Matching and Fitment Scoring With AHP")

    fitment_tables_with_ahp = [
        'GTCSV_AHP_fitment', 'FlanCSV_AHP_fitment', 'GPTCSV_AHP_fitment',
        'LLamaCSV_AHP_fitment', 'NLPCSV_AHP_fitment'
    ]

    combined_fitment_df = load_fitment_data(FITMENT_DB_WITH_AHP, fitment_tables_with_ahp)  # Now it's defined!

    if combined_fitment_df.empty:
        st.error("No fitment data found in pre-calculated tables.")    
    else:
        display_df = combined_fitment_df.drop(columns=["resume_path"])
        st.subheader("Fitment Table with Color Coding")
        st.dataframe(style_fitment_table(display_df), use_container_width=True)

        st.subheader("Average Fitment Scores Gauges")
        score_cols = [c for c in display_df.columns if c.endswith("Score")]
        avg_scores = display_df[score_cols].mean().round(2)
        cols = st.columns(len(avg_scores))

        for i, (model, score) in enumerate(avg_scores.items()):
            color = "red" if score < 50 else "yellow" if score < 80 else "green"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': model.replace("Score", "")},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightcoral'},
                        {'range': [50, 80], 'color': 'khaki'},
                        {'range': [80, 100], 'color': 'lightgreen'}
                    ]
                }
            ))
            cols[i].plotly_chart(fig, use_container_width=True, key=f"tab4_gauge_{model}_{i}")

        # After displaying gauges, add this interpretation
        if all(score < 50 for score in avg_scores):
            interpretation = (
                "All models and the Ground Truth are in the Low Fit range (<50). "
                "This suggests most resumes in the dataset are poor matches for the job description.\n\n"
                "Current average scores by model: " +
                ", ".join([f"{model}: {score:.2f}" for model, score in avg_scores.items()])
            )
        else:
            interpretation = (
                "Mixed fitment levels across models.\n\n"
                "Current average scores by model: " +
                ", ".join([f"{model}: {score:.2f}" for model, score in avg_scores.items()])
            )

        st.markdown(f"**Interpretation of Scores:**\n\n{interpretation}")




        # Confusion matrix and metrics (2 per row)
        st.subheader("Confusion Matrices & Metrics (Ground Truth vs Models)")
        labels = ["Low Fit", "Medium Fit", "Fit"]

        level_cols = [col for col in display_df.columns if col.endswith("_Level") and not col.startswith("GT")]
        for i in range(0, len(level_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(level_cols[i:i+2]):
                model_name = col_name.replace("_Level", "")
                y_true = display_df["GT_Level"].str.strip()
                y_pred = display_df[col_name].str.strip()

                cm = confusion_matrix(y_true, y_pred, labels=labels)
                fig_cm = ff.create_annotated_heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    showscale=True
                )
                fig_cm.update_layout(title=f"Confusion Matrix: GT vs {model_name}", width=350, height=350)

                with cols[j]:
                    st.plotly_chart(fig_cm, use_container_width=False, key=f"tab4_cm_{model_name}")
                    accuracy, precision, recall = compute_metrics_vs_gt(y_true, y_pred, labels=labels)
                    st.markdown(f"**Metrics for {model_name}:**  Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%}")

# ----------- Tab 5 --------------
with tab5:
    st.header("Fitment Scoring With AHP (Live Calculation) - Using GPT Model")

    # Load GPT model features from Bleus_Roug DB
    conn = sqlite3.connect(BLEU_ROUG_DB)
    df_features = pd.read_sql_query("SELECT person_name, total_experience, current_tenure, certification_count, publication_count, skill_density, resume_path FROM GPTCSV", conn)
    conn.close()

    if df_features.empty:
        st.error("No GPT resume features found in Bleus_Roug GPTCSV table.")
        st.stop()

    # Load JDs from main DB
    conn = sqlite3.connect(DB_PATH)
    df_jds = pd.read_sql_query("SELECT id, jd_title, jd_parsed_json FROM job_descriptions ORDER BY added_at DESC", conn)
    conn.close()

    if df_jds.empty:
        st.error("No Job Descriptions found in JD DB.")
        st.stop()

    # Select Job Description
    jd_titles = df_jds['jd_title'].tolist()
    selected_jd_title = st.selectbox("Select Job Description (JD)", jd_titles)

    jd_row = df_jds[df_jds['jd_title'] == selected_jd_title].iloc[0]
    try:
        jd_parsed = json.loads(jd_row['jd_parsed_json'])
    except Exception:
        jd_parsed = {}

    jd_weights = {
        "skill": jd_parsed.get("skill_density", 1),
        "experience": jd_parsed.get("experience", 1),
        "tenure": jd_parsed.get("tenure", 1),
        "publication": jd_parsed.get("publication", 1),
        "certification": jd_parsed.get("certification", 1),
    }

    # Sidebar sliders for AHP weights
    st.sidebar.header("Adjust AHP Weights (1 to 10)")

    base_weights = {
        "skill": st.sidebar.slider("Skill Density", 1, 10, 5),
        "experience": st.sidebar.slider("Experience", 1, 10, 4),
        "tenure": st.sidebar.slider("Tenure", 1, 10, 3),
        "publication": st.sidebar.slider("Publication", 1, 10, 2),
        "certification": st.sidebar.slider("Certification", 1, 10, 1),
    }

    norm_base = normalize_weights(base_weights)
    norm_jd = normalize_weights(jd_weights)

    # Map DB columns to feature keys
    feature_map = {
        "skill_density": "skill",
        "total_experience": "experience",
        "current_tenure": "tenure",
        "publication_count": "publication",
        "certification_count": "certification"
    }

    # Max values per feature for normalization
    max_vals = {}
    for db_col, key in feature_map.items():
        if db_col in df_features.columns and not df_features[db_col].isnull().all():
            max_vals[key] = df_features[db_col].max()
            if max_vals[key] == 0 or pd.isna(max_vals[key]):
                max_vals[key] = 1
        else:
            max_vals[key] = 1

    max_score = sum(max_vals[k] * norm_jd[k] * norm_base[k] for k in max_vals)
    if max_score == 0:
        max_score = 1

    # Calculate fitment score per row dynamically
    def calc_fitment(row):
        total = 0
        for db_col, key in feature_map.items():
            val = row[db_col] if pd.notna(row[db_col]) else 0
            total += val * norm_jd[key] * norm_base[key]
        return round((total / max_score) * 100, 2)

    df_features['Fitment Score'] = df_features.apply(calc_fitment, axis=1)
    df_features['Fitment Level'] = df_features['Fitment Score'].apply(classify_fit)

    # Show only needed columns
    display_df = df_features[['person_name', 'Fitment Score', 'Fitment Level']].copy()
    display_df.rename(columns={
        'person_name': 'Person Name',
        'Fitment Score': 'Fitment Score',
        'Fitment Level': 'Fitment Level'
    }, inplace=True)

    st.subheader("Calculated Fitment Table")
    st.dataframe(display_df.style.applymap(fitment_level_color, subset=['Fitment Level']), use_container_width=True)

    # Display gauge and confusion matrix side by side
    cols = st.columns(2)

    # Average fitment gauge
    avg_score = df_features['Fitment Score'].mean().round(2)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_score,
        title={'text': "Average Fitment Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'lightgreen' if avg_score > 70 else 'khaki' if avg_score > 30 else 'lightcoral'},
            'steps': [
                {'range': [0, 30], 'color': 'lightcoral'},
                {'range': [30, 70], 'color': 'khaki'},
                {'range': [70, 100], 'color': 'lightgreen'},
            ],
        }
    )).update_layout(width=280, height=280)

    cols[0].plotly_chart(fig_gauge, use_container_width=True, key="tab5_avg_gauge")

    # Load GPT fitment level from fitment DB for confusion matrix (instead of GT here)
    conn = sqlite3.connect(FITMENT_DB_NO_AHP)
    gt_df = pd.read_sql_query("SELECT resume_path, fitment_level FROM GPTCSV_fitment", conn)
    conn.close()

    # Map resume_path to fitment_level
    gt_map = dict(zip(gt_df['resume_path'], gt_df['fitment_level']))
    df_features['GPT_Level'] = df_features['resume_path'].map(gt_map).fillna("Unknown")

    valid_df = df_features[df_features['GPT_Level'] != "Unknown"]

    # Confusion matrix and metrics (GPT live score vs GPT fitment levels from DB)
    labels = ["Low Fit", "Medium Fit", "Fit"]
    y_true = valid_df['GPT_Level'].str.strip()
    y_pred = valid_df['Fitment Level'].str.strip()

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    )
    fig_cm.update_layout(title="Confusion Matrix (GPT DB vs Live Fitment)", width=350, height=350)
    cols[1].plotly_chart(fig_cm, use_container_width=False, key="tab5_confusion_matrix")

    st.markdown(f"**Metrics:** Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%}")

# ----------- Tab 6 --------------
with tab6:
    st.subheader("Upload New Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file and st.button("Parse and Save Resume"):
        upload_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.read())
        try:
            rid, data, _ = parser.process_resume(upload_path)
            st.success(f"✅ Resume parsed and saved (RowID: {rid})")
        except Exception as e:
            st.error(f"❌ Failed to parse resume: {e}")

    st.subheader("Upload New Job Description (JD)")
    jd_file = st.file_uploader("Upload JD File (DOCX)", type=["docx"], key="jd_upload")
    if jd_file and st.button("Parse and Save JD"):
        jd_path = os.path.join("uploads", jd_file.name)
        with open(jd_path, "wb") as f:
            f.write(jd_file.read())
        jd_text = extract_docx_text(jd_path)
        parsed = parse_jd_features_with_llm(jd_text)
        if parsed:
            upsert_jd_to_db(os.path.splitext(jd_file.name)[0], jd_text, parsed)
            st.success("✅ JD Parsed and Saved.")
        else:
            st.error("❌ Failed to parse JD using LLaMA.")

    st.subheader("Chatbot Assistant")
    chat_mode = st.radio("Context:", ["Resume", "Job Description", "Resume + JD"], horizontal=True)

    resumes = get_all_resumes()
    resume_map = {f"[{r['rowid']}] {r['person_name']}": r for r in resumes}
    jds = get_all_jds()
    jd_map = {f"[{j[0]}] {j[1]}": j for j in jds}

    selected_resume = None
    selected_jd = None

    if chat_mode == "Resume":
        selected_resume = st.selectbox("Select Resume", list(resume_map.keys()), key="resume_select_chat")
    elif chat_mode == "Job Description":
        selected_jd = st.selectbox("Select JD", list(jd_map.keys()), key="jd_select_chat")
    elif chat_mode == "Resume + JD":
        selected_resume = st.selectbox("Select Resume", list(resume_map.keys()), key="resume_select_chat")
        selected_jd = st.selectbox("Select JD", list(jd_map.keys()), key="jd_select_chat")

    user_question = st.text_area("💬 Your Question")

    if st.button("Ask"):
        blocks = []
        if "Resume" in chat_mode and selected_resume:
            resume_data = resume_map[selected_resume]
            res_json = json.dumps(resume_data, indent=2)
            blocks.append(f"Resume:\n{res_json}")

        if "Job Description" in chat_mode and selected_jd:
            try:
                jd_json = json.loads(jd_map[selected_jd][2])
                blocks.append(f"Job Description:\n{json.dumps(jd_json, indent=2)}")
            except Exception as e:
                st.error(f"❌ Failed to load JD JSON: {e}")
                st.stop()

        if not blocks:
            st.warning("⚠️ Please select appropriate input(s) based on the context.")
            st.stop()

        context_text = "\n\n".join(blocks)
        prompt = f"You are a hiring expert.\n\n{context_text}\n\nQuestion: {user_question}\nAnswer:"

        try:
            reply = llm.invoke(prompt)
            st.success("🧠 LLaMA3 says:")
            st.write(reply)
        except Exception as e:
            st.error(f"❌ Error: {e}")
