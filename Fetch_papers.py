import requests
import feedparser
import pandas as pd
import time
import urllib.parse
import re
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================================
# CONFIG
# ==========================================================

EXISTING_FILE = "arxiv_unique_papers_20260326_223108.xlsx"
PAPERS_PER_DOMAIN = 1500
BATCH_SIZE = 100
SLEEP_TIME = 3

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"arxiv_new_6000_unique_{timestamp}.xlsx"

# Use a real email here
CONTACT_EMAIL = "dharu.manne@gmail.com"

DATE_FILTER = "submittedDate:[201901010000 TO 202612312359]"

queries = {
    "AI": f"(cat:cs.AI OR cat:cs.LG) AND {DATE_FILTER}",
    "Security": f"(cat:cs.CR) AND {DATE_FILTER}",
    "Hardware": f"(cat:cs.AR OR cat:cs.ET) AND {DATE_FILTER}",
    "Medicine": f"(cat:q-bio.BM OR cat:q-bio.GN OR cat:q-bio.QM OR cat:q-bio.SC OR cat:q-bio.NC) AND {DATE_FILTER}"
}

domain_prefix = {
    "AI": "AI",
    "Security": "SEC",
    "Hardware": "HW",
    "Medicine": "MED"
}

domain_order = ["AI", "Security", "Hardware", "Medicine"]

# ==========================================================
# SESSION
# ==========================================================

def create_session():
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": f"arxiv-paper-scraper/1.0 ({CONTACT_EMAIL})",
        "Accept": "application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.1"
    })

    return session

session = create_session()

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

def load_existing_data(existing_file):
    try:
        df = pd.read_excel(existing_file)
        print(f"Loaded existing file: {existing_file}")
        print(f"Existing rows: {len(df)}")
    except FileNotFoundError:
        print(f"Existing file not found: {existing_file}")
        return set(), set(), set(), {
            "AI": 101,
            "Security": 101,
            "Hardware": 101,
            "Medicine": 101
        }

    existing_arxiv_ids = set()
    existing_urls = set()
    existing_titles = set()

    next_ids = {
        "AI": 101,
        "Security": 101,
        "Hardware": 101,
        "Medicine": 101
    }

    for _, row in df.iterrows():
        arxiv_id = str(row.get("arXiv_id", "")).strip()
        url = str(row.get("URL", "")).strip()
        title = normalize_text(row.get("Title", ""))

        if arxiv_id and arxiv_id.lower() != "nan":
            existing_arxiv_ids.add(arxiv_id)
        if url and url.lower() != "nan":
            existing_urls.add(url)
        if title:
            existing_titles.add(title)

        domain = str(row.get("Domain_tag", "")).strip()
        paper_id = str(row.get("Paper_id", "")).strip()
        if domain in next_ids and "_" in paper_id:
            try:
                num = int(paper_id.split("_")[-1])
                if num >= next_ids[domain]:
                    next_ids[domain] = num + 1
            except:
                pass

    return existing_arxiv_ids, existing_urls, existing_titles, next_ids

# ==========================================================
# FETCH FROM ARXIV
# ==========================================================

def fetch_arxiv(query, start=0, max_results=100):
    base_url = "https://export.arxiv.org/api/query"

    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    try:
        response = session.get(base_url, params=params, timeout=(20, 60))
        print("Request URL:", response.url)
        print("HTTP Status:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type", ""))

        response.raise_for_status()

        # Debug: if arXiv sends HTML instead of XML
        content_type = response.headers.get("Content-Type", "").lower()
        if "xml" not in content_type and "atom" not in content_type:
            print("Unexpected response body (first 300 chars):")
            print(response.text[:300])
            return []

        feed = feedparser.parse(response.text)

        if getattr(feed, "bozo", 0):
            print("Feed parse warning:", feed.bozo_exception)
            print("Response preview:", response.text[:300])

        return feed.entries

    except requests.RequestException as e:
        print("Request failed:", e)
        return []

# ==========================================================
# MAIN SCRAPER
# ==========================================================

def build_new_unique_dataset():
    existing_arxiv_ids, existing_urls, existing_titles, next_ids = load_existing_data(EXISTING_FILE)

    new_arxiv_ids = set()
    new_urls = set()
    new_titles = set()

    all_new_rows = []

    for domain in domain_order:
        query = queries[domain]
        prefix = domain_prefix[domain]

        collected = 0
        start = 0
        skipped_existing = 0
        skipped_new_dup = 0
        empty_rounds = 0

        print(f"\nFetching NEW {domain} papers...")

        while collected < PAPERS_PER_DOMAIN:
            entries = fetch_arxiv(query, start=start, max_results=BATCH_SIZE)

            if not entries:
                empty_rounds += 1
                print(f"No valid entries for {domain} on start={start}")
                if empty_rounds >= 3:
                    print(f"Stopping {domain}: repeated empty/invalid responses.")
                    break
                time.sleep(5)
                continue

            added_this_round = 0

            for entry in entries:
                try:
                    arxiv_id = entry.id.split("/abs/")[-1].strip()
                    title = entry.title.strip().replace("\n", " ")
                    normalized_title = normalize_text(title)
                    abstract = entry.summary.strip().replace("\n", " ")
                    authors = ", ".join(a.name for a in entry.authors) if hasattr(entry, "authors") else ""
                    url = entry.link.strip()
                    published = entry.published

                    if (
                        arxiv_id in existing_arxiv_ids or
                        url in existing_urls or
                        normalized_title in existing_titles
                    ):
                        skipped_existing += 1
                        continue

                    if (
                        arxiv_id in new_arxiv_ids or
                        url in new_urls or
                        normalized_title in new_titles
                    ):
                        skipped_new_dup += 1
                        continue

                    custom_id = f"{prefix}_{next_ids[domain]}"

                    all_new_rows.append({
                        "Paper_id": custom_id,
                        "Title": title,
                        "Authors": authors,
                        "Abstract": abstract,
                        "Published": published,
                        "Source": "arXiv",
                        "URL": url,
                        "Domain_tag": domain,
                        "arXiv_id": arxiv_id
                    })

                    new_arxiv_ids.add(arxiv_id)
                    new_urls.add(url)
                    new_titles.add(normalized_title)

                    next_ids[domain] += 1
                    collected += 1
                    added_this_round += 1

                    if collected >= PAPERS_PER_DOMAIN:
                        break

                except Exception as inner_e:
                    print(f"Skipping one bad entry in {domain}: {inner_e}")

            print(
                f"{domain}: collected {collected}/{PAPERS_PER_DOMAIN} | "
                f"skipped existing={skipped_existing} | skipped new dup={skipped_new_dup}"
            )

            start += BATCH_SIZE
            empty_rounds = 0 if added_this_round > 0 else empty_rounds + 1

            if empty_rounds >= 5:
                print(f"Stopping {domain}: too many pages with no new unique papers.")
                break

            time.sleep(SLEEP_TIME)

        print(f"Finished {domain}: {collected} new unique papers")

    if not all_new_rows:
        print("\nNo new unique papers found.")
        return

    df_new = pd.DataFrame(all_new_rows)
    df_new["Domain_tag"] = pd.Categorical(df_new["Domain_tag"], categories=domain_order, ordered=True)
    df_new = df_new.sort_values(by=["Domain_tag", "Paper_id"]).reset_index(drop=True)

    try:
        df_new.to_excel(OUTPUT_FILE, index=False)
        print("\nNew unique dataset created successfully")
        print("Total new papers:", len(df_new))
        print("Saved to:", OUTPUT_FILE)
        print("\nCount by domain:")
        print(df_new["Domain_tag"].value_counts().reindex(domain_order, fill_value=0))
    except PermissionError:
        fallback_csv = f"arxiv_new_6000_unique_{timestamp}.csv"
        df_new.to_csv(fallback_csv, index=False, encoding="utf-8-sig")
        print("\nExcel file is locked. Saved CSV instead:", fallback_csv)

if __name__ == "__main__":
    build_new_unique_dataset()



# check for duplicates.


# import pandas as pd
# import re
# from datetime import datetime

# # ===========================================================
# # CONFIG
# # ===========================================================

# INPUT_FILE = "arxiv_unique_papers_20260326_223108.xlsx"   # change to your file name
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# OUTPUT_UNIQUE_FILE = f"arxiv_unique_papers_{timestamp}.xlsx"
# OUTPUT_DUPLICATES_FILE = f"arxiv_removed_duplicates_{timestamp}.xlsx"

# # Keep this exact order
# DOMAIN_ORDER = ["AI", "Security", "Hardware", "Medicine"]

# # ===========================================================
# # HELPER: CLEAN TITLE FOR DUPLICATE CHECK
# # ===========================================================

# def normalize_title(title):
#     if pd.isna(title):
#         return ""
    
#     title = str(title).strip().lower()
    
#     # remove extra spaces
#     title = re.sub(r"\s+", " ", title)
    
#     # remove punctuation
#     title = re.sub(r"[^\w\s]", "", title)
    
#     return title

# # ===========================================================
# # MAIN
# # ===========================================================

# def remove_duplicates():
#     # Load file
#     df = pd.read_excel(INPUT_FILE)

#     print("Original total rows:", len(df))

#     # Preserve domain order
#     df["Domain_tag"] = pd.Categorical(df["Domain_tag"], categories=DOMAIN_ORDER, ordered=True)
#     df = df.sort_values(by=["Domain_tag"]).reset_index(drop=True)

#     # Add original row number for stable ordering within each domain
#     df["original_order"] = range(len(df))

#     # Normalize title
#     df["normalized_title"] = df["Title"].apply(normalize_title)

#     # -------------------------------------------------------
#     # Mark duplicates
#     # A paper is duplicate if:
#     #   same arXiv_id OR same URL OR same normalized title
#     # -------------------------------------------------------

#     dup_arxiv = df.duplicated(subset=["arXiv_id"], keep="first")
#     dup_url = df.duplicated(subset=["URL"], keep="first")
#     dup_title = df.duplicated(subset=["normalized_title"], keep="first")

#     duplicate_mask = dup_arxiv | dup_url | dup_title

#     duplicates_df = df[duplicate_mask].copy()
#     unique_df = df[~duplicate_mask].copy()

#     # -------------------------------------------------------
#     # Sort again exactly in wanted order
#     # AI -> Security -> Hardware -> Medicine
#     # Preserve original order inside each domain
#     # -------------------------------------------------------

#     unique_df = unique_df.sort_values(by=["Domain_tag", "original_order"]).reset_index(drop=True)
#     duplicates_df = duplicates_df.sort_values(by=["Domain_tag", "original_order"]).reset_index(drop=True)

#     # Remove helper columns before saving
#     unique_df = unique_df.drop(columns=["normalized_title", "original_order"], errors="ignore")
#     duplicates_df = duplicates_df.drop(columns=["normalized_title", "original_order"], errors="ignore")

#     # Save files
#     unique_df.to_excel(OUTPUT_UNIQUE_FILE, index=False)
#     duplicates_df.to_excel(OUTPUT_DUPLICATES_FILE, index=False)

#     # Print summary
#     print("Unique papers:", len(unique_df))
#     print("Removed duplicates:", len(duplicates_df))
#     print("Saved unique file:", OUTPUT_UNIQUE_FILE)
#     print("Saved duplicates file:", OUTPUT_DUPLICATES_FILE)

#     # Domain-wise count
#     print("\nUnique papers by domain:")
#     print(unique_df["Domain_tag"].value_counts().reindex(DOMAIN_ORDER, fill_value=0))

# if __name__ == "__main__":
#     remove_duplicates()



# ====================
# fetch papers
# -----------------------------------------------------------
# # ARXIV PAPER SCRAPER (CLEAN + SAFE + LARGE SCALE)



# import feedparser
# import pandas as pd
# import time
# import urllib.parse

# # -----------------------------------------------------------
# # CONFIG
# # -----------------------------------------------------------

# TARGET_PAPERS = 6000
# BATCH_SIZE = 100   # arXiv max per request
# SLEEP_TIME = 3     # avoid rate limit

# OUTPUT_FILE = "papers_large_dataset.xlsx"

# # -----------------------------------------------------------
# # DOMAIN QUERIES (PURE SEPARATION)
# # -----------------------------------------------------------

# queries = {
#     "AI": "cat:cs.AI OR cat:cs.LG",
#     "Security": "cat:cs.CR",
#     "Hardware": "cat:cs.AR OR cat:cs.ET",
#     "Medicine": "cat:q-bio OR cat:stat.ML AND medical"
# }

# # -----------------------------------------------------------
# # FUNCTION: FETCH FROM ARXIV
# # -----------------------------------------------------------

# def fetch_arxiv(query, start=0, max_results=100):

#     base_url = "http://export.arxiv.org/api/query?"

#     query_encoded = urllib.parse.quote(query)

#     url = (
#         f"{base_url}"
#         f"search_query={query_encoded}"
#         f"&start={start}"
#         f"&max_results={max_results}"
#         f"&sortBy=submittedDate"
#         f"&sortOrder=descending"
#     )

#     feed = feedparser.parse(url)

#     return feed.entries


# # -----------------------------------------------------------
# # MAIN SCRAPER
# # -----------------------------------------------------------

# def build_dataset():

#     all_papers = []
#     paper_id = 1

#     per_domain_target = TARGET_PAPERS // len(queries)

#     for domain, query in queries.items():

#         print(f"\n🔍 Fetching {domain} papers...")

#         collected = 0
#         start = 0

#         while collected < per_domain_target:

#             try:
#                 entries = fetch_arxiv(query, start=start, max_results=BATCH_SIZE)

#                 if len(entries) == 0:
#                     print("No more results for this domain.")
#                     break

#                 for entry in entries:

#                     # YEAR FILTER (IMPORTANT)
#                     published_year = int(entry.published[:4])

#                     if published_year < 2023:
#                         continue

#                     title = entry.title.strip().replace("\n", " ")
#                     abstract = entry.summary.strip().replace("\n", " ")
#                     authors = ", ".join([a.name for a in entry.authors])
#                     url = entry.link
#                     arxiv_id = entry.id.split("/abs/")[-1]

#                     all_papers.append({
#                         "Paper_id": f"P{paper_id}",
#                         "Title": title,
#                         "Authors": authors,
#                         "Abstract": abstract,
#                         "Source": "arXiv",
#                         "URL": url,
#                         "Domain_tag": domain,
#                         "arXiv_id": arxiv_id
#                     })

#                     paper_id += 1
#                     collected += 1

#                     if collected >= per_domain_target:
#                         break

#                 start += BATCH_SIZE

#                 print(f"{domain}: {collected} collected")

#                 time.sleep(SLEEP_TIME)

#             except Exception as e:
#                 print("Error:", e)
#                 print("Retrying...")
#                 time.sleep(5)

#         print(f"✅ Finished {domain}: {collected} papers")

#     # -------------------------------------------------------
#     # SAVE DATASET
#     # -------------------------------------------------------

#     df = pd.DataFrame(all_papers)
#     df.to_excel(OUTPUT_FILE, index=False)

#     print("\n🎯 DATASET CREATED SUCCESSFULLY")
#     print("Total papers:", len(df))


# # -----------------------------------------------------------
# # RUN
# # -----------------------------------------------------------

# if __name__ == "__main__":
#     build_dataset()