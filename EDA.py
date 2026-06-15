import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
from collections import Counter

# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = "New_Papers.xlsx"   # <-- change this
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

CLEAN_OUTPUT_FILE = f"cleaned_dataset_{timestamp}.xlsx"
DUP_OUTPUT_FILE = f"duplicates_found_{timestamp}.xlsx"
REPORT_FILE = f"eda_report_{timestamp}.txt"

DOMAIN_ORDER = ["AI", "Security", "Hardware", "Medicine"]

PREFIX_MAP = {
    "AI": "AI",
    "Security": "SEC",
    "Hardware": "HW",
    "Medicine": "MED"
}

# ==========================================================
# HELPERS
# ==========================================================

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def extract_id_number(paper_id):
    try:
        return int(str(paper_id).split("_")[-1])
    except:
        return None

def get_year(date_text):
    try:
        return str(date_text)[:4]
    except:
        return "Unknown"

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel(INPUT_FILE)

print("==================================================")
print("DATASET LOADED")
print("==================================================")
print("Total rows:", len(df))
print("Columns:", list(df.columns))

# Preserve original row order
df["original_order"] = range(len(df))

# ==========================================================
# CHECK DOMAIN ORDER
# ==========================================================

print("\n==================================================")
print("CHECKING DOMAIN ORDER")
print("==================================================")

df["Domain_tag"] = pd.Categorical(df["Domain_tag"], categories=DOMAIN_ORDER, ordered=True)

domain_sequence = df["Domain_tag"].tolist()
sorted_domain_sequence = sorted(domain_sequence, key=lambda x: DOMAIN_ORDER.index(x))

if domain_sequence == sorted_domain_sequence:
    domain_order_ok = True
    print("Domain order is CORRECT.")
else:
    domain_order_ok = False
    print("Domain order is NOT correct.")

# ==========================================================
# CHECK PAPER ID ORDER
# ==========================================================

print("\n==================================================")
print("CHECKING PAPER ID ORDER")
print("==================================================")

id_issues = []

for domain in DOMAIN_ORDER:
    domain_df = df[df["Domain_tag"] == domain].copy()

    if domain_df.empty:
        continue

    expected_prefix = PREFIX_MAP[domain]
    ids = domain_df["Paper_id"].astype(str).tolist()

    numeric_ids = []
    for pid in ids:
        if not pid.startswith(expected_prefix + "_"):
            id_issues.append(f"{domain}: Wrong prefix found -> {pid}")
        num = extract_id_number(pid)
        if num is not None:
            numeric_ids.append(num)

    # Check if IDs are increasing
    if numeric_ids != sorted(numeric_ids):
        id_issues.append(f"{domain}: IDs are NOT in increasing order")

    # Check missing numbers
    if numeric_ids:
        expected_range = list(range(min(numeric_ids), max(numeric_ids) + 1))
        missing = sorted(set(expected_range) - set(numeric_ids))
        if missing:
            id_issues.append(f"{domain}: Missing ID numbers -> {missing[:20]}{'...' if len(missing) > 20 else ''}")

if not id_issues:
    print("Paper IDs are in proper order.")
else:
    print("Paper ID issues found:")
    for issue in id_issues:
        print("-", issue)

# ==========================================================
# DUPLICATE CHECK
# ==========================================================

print("\n==================================================")
print("CHECKING DUPLICATES")
print("==================================================")

df["normalized_title"] = df["Title"].apply(normalize_text)

dup_arxiv = df.duplicated(subset=["arXiv_id"], keep="first")
dup_url = df.duplicated(subset=["URL"], keep="first")
dup_title = df.duplicated(subset=["normalized_title"], keep="first")

duplicate_mask = dup_arxiv | dup_url | dup_title

duplicates_df = df[duplicate_mask].copy()
clean_df = df[~duplicate_mask].copy()

print("Duplicate papers found:", len(duplicates_df))
print("Unique papers remaining:", len(clean_df))

# ==========================================================
# SORT CLEAN DATASET IN REQUIRED ORDER
# ==========================================================

print("\n==================================================")
print("SORTING DATASET")
print("==================================================")

clean_df["Domain_tag"] = pd.Categorical(clean_df["Domain_tag"], categories=DOMAIN_ORDER, ordered=True)
clean_df["Paper_id_num"] = clean_df["Paper_id"].apply(extract_id_number)

clean_df = clean_df.sort_values(by=["Domain_tag", "Paper_id_num"]).reset_index(drop=True)

# Remove helper cols before saving
save_clean_df = clean_df.drop(columns=["normalized_title", "original_order", "Paper_id_num"], errors="ignore")
save_dup_df = duplicates_df.drop(columns=["normalized_title", "original_order"], errors="ignore")

save_clean_df.to_excel(CLEAN_OUTPUT_FILE, index=False)
save_dup_df.to_excel(DUP_OUTPUT_FILE, index=False)

print("Saved cleaned file:", CLEAN_OUTPUT_FILE)
print("Saved duplicates file:", DUP_OUTPUT_FILE)

# ==========================================================
# BASIC EDA
# ==========================================================

print("\n==================================================")
print("RUNNING EDA")
print("==================================================")

# Domain counts
domain_counts = clean_df["Domain_tag"].value_counts().reindex(DOMAIN_ORDER, fill_value=0)

# Year counts
# clean_df["Year"] = clean_df["Published"].apply(get_year)
# year_counts = clean_df["Year"].value_counts().sort_index()

# Abstract length
clean_df["Abstract_Length"] = clean_df["Abstract"].fillna("").apply(lambda x: len(str(x).split()))

# Title word frequencies
stopwords = {
    "for", "and", "the", "of", "in", "to", "a", "with", "on", "using",
    "via", "from", "by", "an", "is", "are", "towards", "based", "learning",
    "deep", "machine", "study", "analysis", "system", "systems", "model",
    "models", "approach", "toward", "new", "efficient"
}

all_title_words = []
for title in clean_df["Title"].fillna(""):
    words = normalize_text(title).split()
    all_title_words.extend([w for w in words if w not in stopwords and len(w) > 2])

top_words = Counter(all_title_words).most_common(20)

# ==========================================================
# SAVE TEXT REPORT
# ==========================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write("EDA REPORT\n")
    f.write("==================================================\n\n")

    f.write(f"Original rows: {len(df)}\n")
    f.write(f"Unique rows after duplicate removal: {len(clean_df)}\n")
    f.write(f"Duplicates removed: {len(duplicates_df)}\n\n")

    f.write("--------------------------------------------------\n")
    f.write("DOMAIN ORDER CHECK\n")
    f.write("--------------------------------------------------\n")
    f.write(f"Domain order correct: {domain_order_ok}\n\n")

    f.write("--------------------------------------------------\n")
    f.write("PAPER ID CHECK\n")
    f.write("--------------------------------------------------\n")
    if not id_issues:
        f.write("Paper IDs are in correct order.\n\n")
    else:
        for issue in id_issues:
            f.write(issue + "\n")
        f.write("\n")

    f.write("--------------------------------------------------\n")
    f.write("DOMAIN COUNTS\n")
    f.write("--------------------------------------------------\n")
    f.write(domain_counts.to_string())
    f.write("\n\n")

    # f.write("--------------------------------------------------\n")
    # f.write("YEAR COUNTS\n")
    # f.write("--------------------------------------------------\n")
    # f.write(year_counts.to_string())
    # f.write("\n\n")

    f.write("--------------------------------------------------\n")
    f.write("ABSTRACT LENGTH\n")
    f.write("--------------------------------------------------\n")
    f.write(f"Mean abstract length: {clean_df['Abstract_Length'].mean():.2f} words\n")
    f.write(f"Median abstract length: {clean_df['Abstract_Length'].median():.2f} words\n")
    f.write(f"Min abstract length: {clean_df['Abstract_Length'].min()} words\n")
    f.write(f"Max abstract length: {clean_df['Abstract_Length'].max()} words\n\n")

    f.write("--------------------------------------------------\n")
    f.write("TOP TITLE WORDS\n")
    f.write("--------------------------------------------------\n")
    for word, count in top_words:
        f.write(f"{word}: {count}\n")

print("Saved EDA report:", REPORT_FILE)

# ==========================================================
# PLOTS
# ==========================================================

# 1. Domain distribution
plt.figure(figsize=(8, 5))
domain_counts.plot(kind="bar")
plt.title("Number of Papers per Domain")
plt.xlabel("Domain")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"domain_distribution_{timestamp}.png")
plt.show()

# 2. Year distribution
plt.figure(figsize=(8, 5))
# year_counts.plot(kind="bar")
plt.title("Number of Papers per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"year_distribution_{timestamp}.png")
plt.show()

# 3. Abstract length distribution
plt.figure(figsize=(8, 5))
plt.hist(clean_df["Abstract_Length"], bins=30)
plt.title("Abstract Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"abstract_length_distribution_{timestamp}.png")
plt.show()

# 4. Top title words
top_words_df = pd.DataFrame(top_words, columns=["Word", "Count"])

plt.figure(figsize=(10, 6))
plt.bar(top_words_df["Word"], top_words_df["Count"])
plt.title("Top 20 Most Frequent Words in Titles")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"title_word_frequency_{timestamp}.png")
plt.show()

print("\n==================================================")
print("ALL DONE")
print("==================================================")
print("Cleaned dataset:", CLEAN_OUTPUT_FILE)
print("Duplicates file:", DUP_OUTPUT_FILE)
print("EDA report:", REPORT_FILE)