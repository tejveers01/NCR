import io
import streamlit as st
import requests
import json
import urllib.parse
import certifi
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re
import logging
import os
from dotenv import load_dotenv
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Tuple, Dict, Any, List

# ---------------------------
# Basic configuration
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

WATSONX_API_URL = os.getenv("WATSONX_API_URL")
MODEL_ID = os.getenv("MODEL_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")

# Endpoints
LOGIN_URL = "https://dms.asite.com/apilogin/"
SEARCH_URL = "https://adoddleak.asite.com/commonapi/formsearchapi/search"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# ---------------------------
# Requests session with retries
# ---------------------------
@st.cache_resource
def create_requests_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

session = create_requests_session()

# ---------------------------
# Helper network functions
# ---------------------------
def get_access_token(api_key: str, session: requests.Session) -> str:
    """Generate IAM access token (don't log secrets)."""
    if not api_key:
        logger.error("No API key provided for IAM token generation")
        return None

    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    try:
        resp = session.post(IAM_TOKEN_URL, headers=headers, data=data, verify=certifi.where(), timeout=30)
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if token:
            logger.info("IAM access token obtained")
        return token
    except requests.HTTPError:
        logger.exception("IAM token request failed: %s", getattr(resp, 'text', ''))
    except Exception:
        logger.exception("Unexpected error while obtaining IAM token")
    return None


def login_to_asite(email: str, password: str, session: requests.Session) -> str:
    """Login to Asite and retrieve session id."""
    if not email or not password:
        logger.error("Email or password missing for Asite login")
        return None

    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"emailId": email, "password": password}
    try:
        resp = session.post(LOGIN_URL, headers=headers, data=payload, verify=certifi.where(), timeout=30)
        resp.raise_for_status()
        session_id = resp.json().get("UserProfile", {}).get("Sessionid")
        logger.info("Logged into Asite")
        return session_id
    except requests.HTTPError:
        logger.exception("Asite login failed: %s", getattr(resp, 'text', ''))
    except Exception:
        logger.exception("Unexpected error during Asite login")
    return None


# ---------------------------
# JSON extraction utility
# ---------------------------

def clean_and_parse_json(text: str) -> Dict:
    """Attempt to extract and parse the first JSON object found in a text blob.

    This is tolerant: we try a couple of heuristics and return None if parsing fails.
    """
    if not text:
        return None

    # Quick direct load if it's already JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find outermost braces
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        candidate = text[start_idx:end_idx+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Fall through: try to find smaller balanced JSON by scanning
            pass

    # Find balanced JSON object by counting braces
    brace_count = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0 and start is not None:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    brace_count = 0
                    continue
    return None


# ---------------------------
# Fetching & pagination
# ---------------------------

def fetch_project_data(session_id: str, project_name: str, form_name: str, record_limit: int = 1000,
                       session: requests.Session = session) -> Tuple[Dict[str, Any], List[Dict]]:
    """Fetch project data from Asite with pagination. Returns header info and all FormList entries."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "Cookie": f"ASessionID={session_id}"
    }

    all_data = []
    start_record = 1
    total_records = None

    start_time = datetime.now()
    st.info(f"Fetching data started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Fetching data for %s / %s", project_name, form_name)

    while True:
        search_criteria = {
            "criteria": [
                {"field": "ProjectName", "operator": 1, "values": [project_name]},
                {"field": "FormName", "operator": 1, "values": [form_name]}
            ],
            "recordStart": start_record,
            "recordLimit": record_limit
        }
        search_criteria_str = json.dumps(search_criteria)
        encoded_payload = f"searchCriteria={urllib.parse.quote(search_criteria_str)}"

        try:
            resp = session.post(SEARCH_URL, headers=headers, data=encoded_payload, verify=certifi.where(), timeout=60)
            resp.raise_for_status()
        except requests.HTTPError:
            logger.exception("Search request failed: %s", getattr(resp, 'text', ''))
            st.error("❌ Error fetching data - check logs")
            break
        except Exception:
            logger.exception("Unexpected error fetching project data")
            st.error("❌ Error fetching data - check logs")
            break

        # Try to parse robustly
        try:
            response_json = resp.json()
        except Exception:
            response_json = clean_and_parse_json(resp.text)
            if response_json is None:
                logger.error("Failed to parse search response as JSON")
                st.error("❌ Failed to parse search response")
                break

        if total_records is None:
            total_records = int(response_json.get("responseHeader", {}).get("results-total", 0) or 0)

        form_list = response_json.get("FormList", {}).get("Form", []) or []
        all_data.extend(form_list)
        st.info(f"Fetched {len(all_data)} / {total_records} records")

        if start_record + record_limit - 1 >= total_records:
            break
        start_record += record_limit

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    st.info(f"Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration:.1f}s)")
    logger.info("Fetching complete. Total records: %d", len(all_data))

    return {"responseHeader": {"results": len(all_data), "total_results": total_records}}, all_data


# ---------------------------
# Processing JSON into DataFrame
# ---------------------------

def process_json_data(json_data: List[Dict]) -> pd.DataFrame:
    """Convert Asite JSON entries into a normalized DataFrame. Normalizes Discipline to UPPER-case."""
    data = []

    for item in json_data:
        form_details = item.get('FormDetails', {})
        created_date = form_details.get('FormCreationDate', None)
        expected_close_date = form_details.get('UpdateDate', None)
        form_status = form_details.get('FormStatus', None)

        discipline = None
        description = None
        custom_fields = form_details.get('CustomFields', {}).get('CustomField', []) or []

        for field in custom_fields:
            fname = field.get('FieldName')
            if fname == 'CFID_DD_DISC':
                discipline = field.get('FieldValue')
            elif fname == 'CFID_RTA_DES':
                description = BeautifulSoup(field.get('FieldValue', None) or '', "html.parser").get_text()

        # Normalize discipline
        discipline_norm = None
        if discipline:
            discipline_norm = str(discipline).strip().upper()

        # Calculate days if possible
        days_diff = None
        if created_date and expected_close_date:
            try:
                created_date_obj = datetime.strptime(created_date.split('#')[0].strip(), "%d-%b-%Y")
                expected_close_date_obj = datetime.strptime(expected_close_date.split('#')[0].strip(), "%d-%b-%Y")
                days_diff = (expected_close_date_obj - created_date_obj).days
            except Exception:
                days_diff = None

        # Use provided FormID or fallback to a generated uuid-like index
        record_id = None
        if 'FormDetails' in item and isinstance(item.get('FormDetails'), dict):
            record_id = form_details.get('FormID') or item.get('ID') or None

        data.append({
            'Record_ID': record_id,
            'Days': days_diff,
            'Created Date (WET)': created_date,
            'Expected Close Date (WET)': expected_close_date,
            'Description': description,
            'Status': form_status,
            'Discipline': discipline_norm
        })

    df = pd.DataFrame(data)
    # Parse dates properly
    df['Created Date (WET)'] = pd.to_datetime(df['Created Date (WET)'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')
    df['Expected Close Date (WET)'] = pd.to_datetime(df['Expected Close Date (WET)'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')

    if df.empty:
        logger.warning("DataFrame is empty after processing")
        st.warning("⚠️ No data processed")

    return df


# ---------------------------
# Module & Tower extractors
# ---------------------------

def extract_modules_from_description(description: str) -> List[str]:
    """Extract module numbers from description text. Returns ['Common'] when none found."""
    if not description or not isinstance(description, str):
        return ["Common"]

    text = description.lower()
    modules = set()

    # Ranges: module 1 to 3, mod 1-3
    for s, e in re.findall(r"(?:module|mod|m)[\s\-]*(\d{1,3})\s*(?:to|-|–)\s*(\d{1,3})", text, flags=re.IGNORECASE):
        try:
            s_i, e_i = int(s), int(e)
            if 0 < s_i <= e_i <= 500:
                modules.update(f"Module {i}" for i in range(s_i, e_i + 1))
        except Exception:
            continue

    # Individual mentions: Module 5, Mod-5, M5
    for num in re.findall(r"(?:module|mod|m)[\s\-]*(\d{1,3})\b", text, flags=re.IGNORECASE):
        try:
            n = int(num)
            if 0 < n <= 500:
                modules.add(f"Module {n}")
        except Exception:
            continue

    # M1 style without word boundary
    for num in re.findall(r"\bM\s*[-]?\s*(\d{1,3})\b", text, flags=re.IGNORECASE):
        try:
            n = int(num)
            modules.add(f"Module {n}")
        except Exception:
            continue

    return sorted(modules) if modules else ["Common"]


def determine_tower_assignment(description: str) -> str:
    """Return an assigned tower string based on description keywords."""
    if not description or not isinstance(description, str):
        return "Common_Area"

    d = description.lower()
    if any(phrase in d for phrase in ["eligo clubhouse", "eligo-clubhouse", "eligo club"]):
        return "Eligo-Club"

    # Match patterns like: Tower 3, Tower-F, T5, T-3, Tower A
    m = re.search(r"\b(?:tower|t)[\s\-:]*([A-Za-z]|\d{1,3})\b", description, re.IGNORECASE)
    if m:
        val = m.group(1).upper()
        return f"Eligo-Tower-{val}"

    return "Common_Area"


# ---------------------------
# Report generation: NCR
# ---------------------------

@st.cache_data
def generate_ncr_report_for_eligo(df: pd.DataFrame, report_type: str,
                                  start_date=None, end_date=None, Until_Date=None) -> Tuple[Dict[str, Any], str]:
    """Generate NCR report preserving ALL duplicates."""
    try:
        if df is None or df.empty:
            return {"error": "Empty DataFrame"}, ""

        df_local = df.copy()
        df_local = df_local[df_local['Created Date (WET)'].notna()]

        # Standardize date bounds
        if report_type == "Closed":
            start_date = pd.to_datetime(start_date) if start_date else df_local['Created Date (WET)'].min()
            end_date = pd.to_datetime(end_date) if end_date else df_local['Expected Close Date (WET)'].max()

            df_local = df_local[df_local['Expected Close Date (WET)'].notna()]
            if 'Days' not in df_local.columns:
                df_local['Days'] = (pd.to_datetime(df_local['Expected Close Date (WET)']) -
                                     pd.to_datetime(df_local['Created Date (WET)'])).dt.days

            filtered_df = df_local[
                (df_local['Status'] == 'Closed') &
                (pd.to_datetime(df_local['Created Date (WET)']) >= start_date) &
                (pd.to_datetime(df_local['Created Date (WET)']) <= end_date) &
                (pd.to_numeric(df_local['Days'], errors='coerce') > 21)
            ].copy()
        else:
            if Until_Date is None:
                return {"error": "Open Until Date is required"}, ""
            today = pd.to_datetime(Until_Date)
            filtered_df = df_local[df_local['Status'] == 'Open'].copy()
            filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
            filtered_df = filtered_df[filtered_df['Days_From_Today'] > 21].copy()

        if filtered_df.empty:
            return {"error": f"No {report_type} records"}, ""

        processed_data = filtered_df.to_dict(orient='records')
        all_results = {report_type: {"Sites": {}, "Grand_Total": 0}}

        for rec in processed_data:
            try:
                description = str(rec.get("Description", "") or "").strip()
                discipline = (str(rec.get("Discipline", "") or "")).strip().upper()

                # Skip invalid
                if not discipline or discipline == "NONE" or "HSE" in discipline and discipline != 'HSE':
                    # keep non-HSE unless explicitly HSE and handled later
                    pass

                # Categorize
                if any(k in discipline for k in ["SW", "STRUCTURE"]):
                    disc_category = "SW"
                elif any(k in discipline for k in ["FW", "FINISHING", "CIVIL"]):
                    disc_category = "FW"
                elif discipline == "HSE":
                    disc_category = "HSE"
                else:
                    disc_category = "MEP"

                modules = extract_modules_from_description(description)
                tower = determine_tower_assignment(description)

                if tower not in all_results[report_type]["Sites"]:
                    all_results[report_type]["Sites"][tower] = {
                        "Descriptions": [],
                        "Created Date (WET)": [],
                        "Expected Close Date (WET)": [],
                        "Status": [],
                        "Discipline": [],
                        "Modules": [],
                        "SW": 0,
                        "FW": 0,
                        "MEP": 0,
                        "HSE": 0,
                        "Total": 0,
                        "ModulesCount": {},
                        "Record_IDs": []
                    }

                site_data = all_results[report_type]["Sites"][tower]
                site_data["Descriptions"].append(description)
                site_data["Created Date (WET)"].append(str(rec.get("Created Date (WET)", "")))
                site_data["Expected Close Date (WET)"].append(str(rec.get("Expected Close Date (WET)", "")))
                site_data["Status"].append(str(rec.get("Status", "")))
                site_data["Discipline"].append(str(rec.get("Discipline", "")))
                site_data["Modules"].append(modules)

                # Use real Record_ID if present
                record_identifier = rec.get('Record_ID') if rec.get('Record_ID') is not None else rec.get('Record_ID')
                site_data["Record_IDs"].append(record_identifier)

                if disc_category in ["SW", "FW", "MEP", "HSE"]:
                    site_data[disc_category] += 1
                site_data["Total"] += 1

                for mod in modules:
                    site_data["ModulesCount"][mod] = site_data["ModulesCount"].get(mod, 0) + 1

                all_results[report_type]["Grand_Total"] += 1

            except Exception:
                logger.exception("Error while processing a record")
                continue

        # Build summary
        table_data = []
        for site, data in all_results[report_type]["Sites"].items():
            unique_desc = len(set(data["Descriptions"]))
            dup_count = data["Total"] - unique_desc
            table_data.append({
                "Site": site,
                "SW": data["SW"],
                "FW": data["FW"],
                "MEP": data["MEP"],
                "HSE": data.get("HSE", 0),
                "Total": data["Total"],
                "Unique": unique_desc,
                "Duplicates": dup_count
            })

        if table_data:
            df_table = pd.DataFrame(table_data)
            st.write(f"### {report_type} NCR Summary (Duplicates Preserved)")
            st.dataframe(df_table, use_container_width=True)

        return all_results, json.dumps(all_results, default=str)

    except Exception:
        logger.exception("Unhandled error in generate_ncr_report_for_eligo")
        return {"error": "Unhandled exception"}, ""


# ---------------------------
# Housekeeping & Safety reports
# ---------------------------

@st.cache_data
def generate_ncr_Housekeeping_report_for_eligo(df: pd.DataFrame, report_type: str, start_date=None, end_date=None, until_date=None):
    try:
        if df is None or df.empty:
            return {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}, ""

        today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
        housekeeping_keywords = [
            'housekeeping', 'cleaning', 'cleanliness', 'waste disposal', 'garbage', 'trash', 'rubbish', 'debris',
            'litter', 'dust', 'untidy', 'cluttered', 'construction waste', 'pile of garbage', 'poor housekeeping',
            'material storage', 'construction debris', 'cleaning schedule', 'garbage collection', 'waste bins', 'dirty'
        ]

        def is_housekeeping_record(description):
            if not description or not isinstance(description, str):
                return False
            return any(keyword in description.lower() for keyword in housekeeping_keywords)

        if report_type == "Closed":
            filtered_df = df[(df['Discipline'] == 'HSE') & (df['Status'] == 'Closed') & (df['Days'].notnull()) & (df['Days'] > 7) & (df['Description'].apply(is_housekeeping_record))].copy()
        else:
            open_records = df[(df['Status'] == 'Open') & (pd.to_datetime(df['Created Date (WET)']).notna())].copy()
            open_records.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(open_records['Created Date (WET)'])).dt.days
            filtered_df = open_records[(open_records['Days_From_Today'] > 7) & (open_records['Discipline'] == 'HSE') & (open_records['Description'].apply(is_housekeeping_record))].copy()

        if filtered_df.empty:
            return {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}, ""

        result = {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}
        for idx, row in filtered_df.iterrows():
            description = str(row.get("Description", "") or "").strip()
            site = "Common_Area"
            tower_match = re.search(r"(tower|t)\s*-?\s*([A-Za-z0-9]+)", description, re.IGNORECASE)
            if tower_match:
                site = f"Eligo-Tower-{tower_match.group(2).upper()}"

            if site not in result["Housekeeping"]["Sites"]:
                result["Housekeeping"]["Sites"][site] = {"Count": 0, "Descriptions": [], "Created Date (WET)": [], "Expected Close Date (WET)": [], "Status": [], "Record_IDs": []}

            result["Housekeeping"]["Sites"][site]["Descriptions"].append(description)
            result["Housekeeping"]["Sites"][site]["Created Date (WET)"].append(str(row.get("Created Date (WET)", "")))
            result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"].append(str(row.get("Expected Close Date (WET)", "")))
            result["Housekeeping"]["Sites"][site]["Status"].append(str(row.get("Status", "")))
            result["Housekeeping"]["Sites"][site]["Record_IDs"].append(row.get("Record_ID", idx))
            result["Housekeeping"]["Sites"][site]["Count"] += 1
            result["Housekeeping"]["Grand_Total"] += 1

        st.write(f"Total {report_type} records processed: {result['Housekeeping']['Grand_Total']} (All duplicates preserved)")
        return result, json.dumps(result, default=str)

    except Exception:
        logger.exception("Error in Housekeeping report")
        return {"error": "exception"}, ""


@st.cache_data
def generate_ncr_Safety_report_for_eligo(df: pd.DataFrame, report_type: str, start_date=None, end_date=None, until_date=None):
    try:
        if df is None or df.empty:
            return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

        today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
        safety_keywords = [
            'safety precautions', 'temporary electricity', 'safety norms', 'safety belt', 'helmet', 'lifeline', 'guard rails',
            'fall protection', 'ppe', 'electrical hazard', 'unsafe platform', 'catch net', 'edge protection', 'scaffold',
            'lifting equipment', 'dust suppression', 'debris chute', 'spill control', 'crane operator', 'fire hazard', 'working at height'
        ]

        def is_safety_record(description):
            if not description or not isinstance(description, str):
                return False
            return any(keyword in description.lower() for keyword in safety_keywords)

        if report_type == "Closed":
            filtered_df = df[(df['Discipline'] == 'HSE') & (df['Status'] == 'Closed') & (df['Days'].notnull()) & (df['Days'] > 7) & (df['Description'].apply(is_safety_record))].copy()
        else:
            open_records = df[(df['Status'] == 'Open') & (pd.to_datetime(df['Created Date (WET)']).notna())].copy()
            open_records.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(open_records['Created Date (WET)'])).dt.days
            filtered_df = open_records[(open_records['Days_From_Today'] > 7) & (open_records['Discipline'] == 'HSE') & (open_records['Description'].apply(is_safety_record))].copy()

        if filtered_df.empty:
            return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

        result = {"Safety": {"Sites": {}, "Grand_Total": 0}}
        for idx, row in filtered_df.iterrows():
            description = str(row.get("Description", "") or "").strip()
            site = "Common_Area"
            tower_match = re.search(r"(tower|t)\s*-?\s*([A-Za-z0-9]+)", description, re.IGNORECASE)
            if tower_match:
                site = f"Eligo-Tower-{tower_match.group(2).upper()}"

            if site not in result["Safety"]["Sites"]:
                result["Safety"]["Sites"][site] = {"Count": 0, "Descriptions": [], "Created Date (WET)": [], "Expected Close Date (WET)": [], "Status": [], "Record_IDs": []}

            result["Safety"]["Sites"][site]["Descriptions"].append(description)
            result["Safety"]["Sites"][site]["Created Date (WET)"].append(str(row.get("Created Date (WET)", "")))
            result["Safety"]["Sites"][site]["Expected Close Date (WET)"].append(str(row.get("Expected Close Date (WET)", "")))
            result["Safety"]["Sites"][site]["Status"].append(str(row.get("Status", "")))
            result["Safety"]["Sites"][site]["Record_IDs"].append(row.get("Record_ID", idx))
            result["Safety"]["Sites"][site]["Count"] += 1
            result["Safety"]["Grand_Total"] += 1

        st.write(f"Total {report_type} records processed: {result['Safety']['Grand_Total']} (All duplicates preserved)")
        return result, json.dumps(result, default=str)

    except Exception:
        logger.exception("Error in Safety report")
        return {"error": "exception"}, ""


# ---------------------------
# Export functions
# ---------------------------

def export_results_to_excel(result_data: Dict, report_name: str) -> BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        title_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': 'yellow', 'border': 1, 'font_size': 12})
        header_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True})
        data_format = workbook.add_format({'align': 'left', 'valign': 'vcenter', 'border': 1, 'text_wrap': True})

        ws_summary = workbook.add_worksheet('Summary')
        ws_summary.set_column('A:F', 15)
        now = datetime.now().strftime('%d_%B_%Y')
        ws_summary.merge_range('A1:F1', f'{report_name} Summary {now}', title_format)

        headers = ['Site', 'Total Records', 'Unique Records', 'Duplicates', 'Duplication %']
        for col, header in enumerate(headers):
            ws_summary.write(1, col, header, header_format)

        row = 2
        for report_type, sites in result_data.items():
            for site, site_data in sites.get("Sites", {}).items():
                total = site_data.get("Count", site_data.get("Total", 0))
                unique = len(set(site_data.get("Descriptions", [])))
                dupes = total - unique
                dup_pct = f"{(dupes/total*100):.1f}%" if total > 0 else "0%"

                ws_summary.write(row, 0, site, data_format)
                ws_summary.write(row, 1, total, data_format)
                ws_summary.write(row, 2, unique, data_format)
                ws_summary.write(row, 3, dupes, data_format)
                ws_summary.write(row, 4, dup_pct, data_format)
                row += 1

        ws_detail = workbook.add_worksheet('All Records')
        ws_detail.set_column('A:A', 5)
        ws_detail.set_column('B:B', 25)
        ws_detail.set_column('C:C', 50)
        ws_detail.set_column('D:G', 18)

        ws_detail.merge_range('A1:G1', f'{report_name} - All Records (with Duplicates)', title_format)
        headers = ['#', 'Site', 'Description', 'Created Date', 'Expected Close Date', 'Status', 'Record ID']
        for col, header in enumerate(headers):
            ws_detail.write(1, col, header, header_format)

        row = 2
        record_num = 1
        for report_type, sites_data in result_data.items():
            for site, site_data in sites_data.get("Sites", {}).items():
                descriptions = site_data.get("Descriptions", [])
                created = site_data.get("Created Date (WET)", [])
                close_date = site_data.get("Expected Close Date (WET)", [])
                status = site_data.get("Status", [])
                record_ids = site_data.get("Record_IDs", [])

                max_len = max(len(descriptions), len(created), len(close_date), len(status), len(record_ids))
                for i in range(max_len):
                    ws_detail.write(row, 0, record_num, data_format)
                    ws_detail.write(row, 1, site, data_format)
                    ws_detail.write(row, 2, descriptions[i] if i < len(descriptions) else "", data_format)
                    ws_detail.write(row, 3, created[i] if i < len(created) else "", data_format)
                    ws_detail.write(row, 4, close_date[i] if i < len(close_date) else "", data_format)
                    ws_detail.write(row, 5, status[i] if i < len(status) else "", data_format)
                    ws_detail.write(row, 6, record_ids[i] if i < len(record_ids) else "", data_format)
                    row += 1
                    record_num += 1

    output.seek(0)
    return output


def generate_combined_excel_report_for_eligo(all_reports: Dict[str, Any], report_name: str) -> BytesIO:
    """Create a combined Excel workbook from multiple report objects."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        title_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#FFFF00', 'border': 1, 'font_size': 12})
        header_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True})
        data_format = workbook.add_format({'align': 'left', 'valign': 'vcenter', 'border': 1, 'text_wrap': True})
        percent_format = workbook.add_format({'num_format': '0.0%', 'align': 'center', 'border': 1})

        ws_summary = workbook.add_worksheet('Combined_Summary')
        ws_summary.set_column('A:F', 18)
        now = datetime.now().strftime('%d_%B_%Y')
        ws_summary.merge_range('A1:F1', f'{report_name} - Combined Summary {now}', title_format)

        summary_rows = []
        for report_type, report_obj in all_reports.items():
            sites = report_obj.get("Sites", {})
            for site_name, site_data in sites.items():
                total = site_data.get("Count", site_data.get("Total", 0))
                unique = len(set(site_data.get("Descriptions", [])))
                dupes = total - unique
                dup_pct = (dupes / total) if total > 0 else 0.0
                summary_rows.append({
                    "Report Type": report_type,
                    "Site": site_name,
                    "Total Records": total,
                    "Unique Records": unique,
                    "Duplicates": dupes,
                    "Duplication Rate": dup_pct
                })

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
        else:
            df_summary = pd.DataFrame(columns=["Report Type", "Site", "Total Records", "Unique Records", "Duplicates", "Duplication Rate"])

        for col, header in enumerate(df_summary.columns):
            ws_summary.write(1, col, header, header_format)

        for r_idx, row in enumerate(df_summary.itertuples(index=False), start=2):
            ws_summary.write(r_idx, 0, row[0], data_format)
            ws_summary.write(r_idx, 1, row[1], data_format)
            ws_summary.write(r_idx, 2, row[2], data_format)
            ws_summary.write(r_idx, 3, row[3], data_format)
            ws_summary.write(r_idx, 4, row[4], data_format)
            ws_summary.write_number(r_idx, 5, float(row[5]), percent_format)

        for report_type, report_obj in all_reports.items():
            sheet_name = str(report_type)[:30]
            ws = workbook.add_worksheet(sheet_name)
            ws.set_column('A:A', 5)
            ws.set_column('B:B', 25)
            ws.set_column('C:C', 60)
            ws.set_column('D:G', 20)

            ws.merge_range('A1:G1', f'{report_name} - {report_type} - All Records (with Duplicates)', title_format)
            headers = ['#', 'Site', 'Description', 'Created Date', 'Expected Close Date', 'Status', 'Record ID']
            for col, header in enumerate(headers):
                ws.write(1, col, header, header_format)

            row = 2
            record_num = 1
            sites = report_obj.get("Sites", {})
            for site, site_data in sites.items():
                descriptions = site_data.get("Descriptions", [])
                created = site_data.get("Created Date (WET)", [])
                expected = site_data.get("Expected Close Date (WET)", [])
                status = site_data.get("Status", [])
                record_ids = site_data.get("Record_IDs", [])

                max_len = max(len(descriptions), len(created), len(expected), len(status), len(record_ids))
                for i in range(max_len):
                    desc = descriptions[i] if i < len(descriptions) else ""
                    cr = created[i] if i < len(created) else ""
                    ex = expected[i] if i < len(expected) else ""
                    stt = status[i] if i < len(status) else ""
                    rid = record_ids[i] if i < len(record_ids) else ""

                    ws.write(row, 0, record_num, data_format)
                    ws.write(row, 1, site, data_format)
                    ws.write(row, 2, desc, data_format)
                    ws.write(row, 3, cr, data_format)
                    ws.write(row, 4, ex, data_format)
                    ws.write(row, 5, stt, data_format)
                    ws.write(row, 6, rid, data_format)

                    row += 1
                    record_num += 1

    output.seek(0)
    return output


# ---------------------------
# UI helpers
# ---------------------------

def generate_report_title(prefix: str) -> str:
    now = datetime.now()
    return f"{prefix}: {now.strftime('%d_%B_%Y')}"


def show_duplicate_analysis(result_data: Dict, report_type: str):
    if not result_data or "error" in result_data:
        return
    sites = result_data.get(report_type, {}).get("Sites", {})
    analysis_data = []
    for site, site_data in sites.items():
        total_count = site_data.get("Count", site_data.get("Total", 0))
        unique_count = len(set(site_data.get("Descriptions", [])))
        duplicate_count = total_count - unique_count
        analysis_data.append({
            "Site": site,
            "Total Occurrences": total_count,
            "Unique Records": unique_count,
            "Duplicate Occurrences": duplicate_count,
            "Duplication Rate %": f"{(duplicate_count/total_count*100):.1f}%" if total_count > 0 else "0%"
        })
    if analysis_data:
        df_analysis = pd.DataFrame(analysis_data)
        st.write(f"### Duplicate Analysis - {report_type}")
        st.dataframe(df_analysis, use_container_width=True)


# ---------------------------
# Streamlit App Layout
# ---------------------------

st.title("Eligo NCR Report Generator (Updated)")

with st.sidebar.expander("Configuration"):
    email = st.text_input("Asite Email")
    password = st.text_input("Asite Password", type='password')
    project_name = st.text_input("Project Name", value="Eligo")
    form_name = st.text_input("Form Name", value="NCR")
    record_limit = st.number_input("Record Limit per page", min_value=100, max_value=5000, value=1000)
    use_api_key = st.text_input("IBM API Key (optional)")

if st.button("Fetch & Generate Reports"):
    if not email or not password:
        st.error("Please provide Asite credentials in the sidebar")
        st.stop()

    session_id = login_to_asite(email, password, session)
    if not session_id:
        st.error("Login failed - check credentials and logs")
        st.stop()

    header_info, raw_data = fetch_project_data(session_id, project_name, form_name, record_limit, session=session)
    if not raw_data:
        st.error("No data fetched")
        st.stop()

    df = process_json_data(raw_data)
    st.success(f"Processed {len(df)} records into DataFrame")

    # Generate main NCR reports (Closed & Open) preserving duplicates
    closed_results, closed_json = generate_ncr_report_for_eligo(df, "Closed")
    open_results, open_json = generate_ncr_report_for_eligo(df, "Open", Until_Date=datetime.today().strftime('%Y-%m-%d'))

    # Housekeeping & Safety
    housekeeping_res, housekeeping_json = generate_ncr_Housekeeping_report_for_eligo(df, "Open")
    safety_res, safety_json = generate_ncr_Safety_report_for_eligo(df, "Open")

    # Combine
    all_reports = {}
    if isinstance(closed_results, dict) and "error" not in closed_results:
        all_reports.update(closed_results)
    if isinstance(open_results, dict) and "error" not in open_results:
        all_reports.update(open_results)
    if isinstance(housekeeping_res, dict) and "error" not in housekeeping_res:
        all_reports.update(housekeeping_res)
    if isinstance(safety_res, dict) and "error" not in safety_res:
        all_reports.update(safety_res)

    if not all_reports:
        st.error("No reports generated")
        st.stop()

    # Create combined excel
    day = datetime.now().day
    month_name = datetime.now().strftime('%B')
    year = datetime.now().year
    try:
        excel_file = generate_combined_excel_report_for_eligo(all_reports, f"All_Reports_{day}_{month_name}_{year}")
        st.success("Excel generated successfully")
        st.download_button("Download Combined Excel", excel_file, file_name=f"Eligo_All_Reports_{day}_{month_name}_{year}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        logger.exception("Failed to generate combined excel report")
        st.error("Failed to generate combined Excel report - check logs")

    # Show duplicate analysis for housekeeping and safety if available
    show_duplicate_analysis(housekeeping_res, "Housekeeping")
    show_duplicate_analysis(safety_res, "Safety")


# End of file
