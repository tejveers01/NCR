import io
import streamlit as st
import requests
import json
import urllib.parse
import urllib3
import certifi
import pandas as pd  
from bs4 import BeautifulSoup
from datetime import datetime
import re
import logging
import os
from dotenv import load_dotenv
from io import BytesIO
import base64
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Tuple, Dict, Any, List, Set

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# WatsonX configuration
WATSONX_API_URL = os.getenv("WATSONX_API_URL")
MODEL_ID = os.getenv("MODEL_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")

# Check environment variables
if not all([API_KEY, WATSONX_API_URL, MODEL_ID, PROJECT_ID]):
    st.error("❌ Required environment variables (API_KEY, WATSONX_API_URL, MODEL_ID, PROJECT_ID) missing!")
    logger.error("Missing one or more required environment variables")
    st.stop()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API Endpoints
LOGIN_URL = "https://dms.asite.com/apilogin/"
SEARCH_URL = "https://adoddleak.asite.com/commonapi/formsearchapi/search"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"


# ============================================================================
# TOKEN & LOGIN FUNCTIONS
# ============================================================================

def get_access_token(API_KEY):
    """Generate access token for WatsonX API."""
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": API_KEY}
    try:
        response = requests.post(IAM_TOKEN_URL, headers=headers, data=data, verify=certifi.where(), timeout=50)
        if response.status_code == 200:
            token_info = response.json()
            logger.info("Access token generated successfully")
            return token_info['access_token']
        else:
            logger.error(f"Failed to get access token: {response.status_code} - {response.text}")
            st.error(f"❌ Failed to get access token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception getting access token: {str(e)}")
        st.error(f"❌ Error getting access token: {str(e)}")
        return None


def login_to_asite(email, password):
    """Login to Asite and retrieve session ID."""
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"emailId": email, "password": password}
    response = requests.post(LOGIN_URL, headers=headers, data=payload, verify=certifi.where(), timeout=50)
    if response.status_code == 200:
        try:
            session_id = response.json().get("UserProfile", {}).get("Sessionid")
            logger.info(f"Login successful, Session ID: {session_id}")
            return session_id
        except json.JSONDecodeError:
            logger.error("JSONDecodeError during login")
            st.error("❌ Failed to parse login response")
            return None
    logger.error(f"Login failed: {response.status_code}")
    st.error(f"❌ Login failed: {response.status_code}")
    return None


# ============================================================================
# DATA FETCHING & PROCESSING
# ============================================================================

def fetch_project_data(session_id, project_name, form_name, record_limit=1000):
    """Fetch project data from Asite with pagination."""
    headers = {
        "Accept": "application/json", 
        "Content-Type": "application/x-www-form-urlencoded", 
        "Cookie": f"ASessionID={session_id}"
    }
    all_data = []
    start_record = 1
    total_records = None
    
    start_time = datetime.now()
    st.write(f"🔄 Fetching data from Asite started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Started fetching data from Asite for project '{project_name}', form '{form_name}'")

    with st.spinner("Fetching data from Asite..."):
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
                response = requests.post(SEARCH_URL, headers=headers, data=encoded_payload, 
                                       verify=certifi.where(), timeout=50)
                response_json = response.json()
                
                if total_records is None:
                    total_records = response_json.get("responseHeader", {}).get("results-total", 0)
                
                form_list = response_json.get("FormList", {}).get("Form", [])
                all_data.extend(form_list)
                st.info(f"🔄 Fetched {len(all_data)} / {total_records} records")
                
                if start_record + record_limit - 1 >= total_records:
                    break
                start_record += record_limit
                
            except Exception as e:
                logger.error(f"Error fetching data: {str(e)}")
                st.error(f"❌ Error fetching data: {str(e)}")
                break

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    st.write(f"🔄 Fetching completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration:.1f}s)")
    logger.info(f"Finished fetching data. Total records: {len(all_data)}")

    return {"responseHeader": {"results": len(all_data), "total_results": total_records}}, all_data, encoded_payload


def process_json_data(json_data: List[Dict]) -> pd.DataFrame:
    """Process JSON data from Asite into a DataFrame."""
    data = []
    for item in json_data:
        form_details = item.get('FormDetails', {})
        created_date = form_details.get('FormCreationDate', None)
        expected_close_date = form_details.get('UpdateDate', None)
        form_status = form_details.get('FormStatus', None)
        
        discipline = None
        description = None
        custom_fields = form_details.get('CustomFields', {}).get('CustomField', [])
        
        for field in custom_fields:
            if field.get('FieldName') == 'CFID_DD_DISC':
                discipline = field.get('FieldValue', None)
            elif field.get('FieldName') == 'CFID_RTA_DES':
                description = BeautifulSoup(field.get('FieldValue', None) or '', "html.parser").get_text()

        days_diff = None
        if created_date and expected_close_date:
            try:
                created_date_obj = datetime.strptime(created_date.split('#')[0], "%d-%b-%Y")
                expected_close_date_obj = datetime.strptime(expected_close_date.split('#')[0], "%d-%b-%Y")
                days_diff = (expected_close_date_obj - created_date_obj).days
            except Exception as e:
                logger.error(f"Error calculating days difference: {str(e)}")
                days_diff = None

        data.append([days_diff, created_date, expected_close_date, description, form_status, discipline])

    df = pd.DataFrame(data, columns=['Days', 'Created Date (WET)', 'Expected Close Date (WET)', 
                                     'Description', 'Status', 'Discipline'])
    df['Created Date (WET)'] = pd.to_datetime(df['Created Date (WET)'].str.split('#').str[0], 
                                              format="%d-%b-%Y", errors='coerce')
    df['Expected Close Date (WET)'] = pd.to_datetime(df['Expected Close Date (WET)'].str.split('#').str[0], 
                                                      format="%d-%b-%Y", errors='coerce')
    
    if df.empty:
        logger.warning("DataFrame is empty after processing")
        st.warning("⚠️ No data processed. Check the API response.")
    
    return df


def clean_and_parse_json(text: str) -> Dict:
    """Extract and parse JSON from text response."""
    import re
    import json
    
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        try:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {json_str}")
            
    logger.error(f"Could not extract valid JSON from: {text}")
    return None


# ============================================================================
# TEXT EXTRACTION & PARSING
# ============================================================================

def extract_modules_from_description(description: str) -> List[str]:
    """Extract module numbers from description text."""
    if not description:
        return ["Common"]
    
    description_lower = description.lower()
    modules = set()

    # Pattern 1: Ranges like "Module 1 to 3"
    range_patterns = r"(?:module|mod|m)[-\s]*(\d+)\s*(?:to|-|–)\s*(\d+)"
    for start_str, end_str in re.findall(range_patterns, description_lower, re.IGNORECASE):
        try:
            start, end = int(start_str), int(end_str)
            if 0 < start <= end <= 50:
                modules.update(f"Module {i}" for i in range(start, end + 1))
        except ValueError:
            continue

    # Pattern 2: Individual modules
    individual_patterns = r"(?:module|mod|m)[-\s]*(\d{1,2})"
    for num_str in re.findall(individual_patterns, description_lower, re.IGNORECASE):
        try:
            num = int(num_str)
            if 0 < num <= 50:
                modules.add(f"Module {num}")
        except ValueError:
            continue

    return sorted(list(modules)) if modules else ["Common"]


def determine_tower_assignment(description: str) -> str:
    """Assign tower based on description keywords."""
    if not description:
        return "Common_Area"
    
    description_lower = description.lower()
    
    if any(phrase in description_lower for phrase in ["eligo clubhouse", "eligo-clubhouse", "eligo club"]):
        return "Eligo-Club"

    # Tower patterns
    tower_matches = re.findall(r"\b(?:tower|t)\s*[-\s(]*([fgh])\b", description_lower, re.IGNORECASE)
    
    if tower_matches:
        tower_letter = tower_matches[0].upper()
        return f"Eligo-Tower-{tower_letter}"
    
    return "Common_Area"


# ============================================================================
# REPORT GENERATION (IMPROVED - NO DEDUPLICATION)
# ============================================================================

@st.cache_data
def generate_ncr_report_for_eligo(df: pd.DataFrame, report_type: str, 
                                  start_date=None, end_date=None, Until_Date=None) -> Tuple[Dict[str, Any], str]:
    """
    Generate NCR report grouping by Tower WITHOUT hiding duplicates.
    Each occurrence is counted separately.
    """
    try:
        with st.spinner(f"Generating {report_type} NCR Report..."):
            if df is None or df.empty:
                st.error("❌ Input DataFrame is empty or None")
                return {"error": "Empty DataFrame"}, ""
            
            if report_type not in ["Open", "Closed"]:
                st.error(f"❌ Invalid report_type: {report_type}")
                return {"error": "Invalid report_type"}, ""
            
            df = df.copy()
            df = df[df['Created Date (WET)'].notna()]
            
            # ===== FILTER DATA =====
            if report_type == "Closed":
                try:
                    start_date = pd.to_datetime(start_date) if start_date else df['Created Date (WET)'].min()
                    end_date = pd.to_datetime(end_date) if end_date else df['Expected Close Date (WET)'].max()
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid date range: {str(e)}")
                    st.error(f"❌ Invalid date range: {str(e)}")
                    return {"error": "Invalid date range"}, ""

                df = df[df['Expected Close Date (WET)'].notna()]
                
                if 'Days' not in df.columns:
                    df['Days'] = (pd.to_datetime(df['Expected Close Date (WET)']) - 
                                 pd.to_datetime(df['Created Date (WET)'])).dt.days
                
                filtered_df = df[
                    (df['Status'] == 'Closed') &
                    (pd.to_datetime(df['Created Date (WET)']) >= start_date) &
                    (pd.to_datetime(df['Created Date (WET)']) <= end_date) &
                    (pd.to_numeric(df['Days'], errors='coerce') > 21)
                ].copy()
                
            else:  # Open
                if Until_Date is None:
                    st.error("❌ Open Until Date is required for Open NCR Report")
                    return {"error": "Open Until Date is required"}, ""
                
                try:
                    today = pd.to_datetime(Until_Date)
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid Open Until Date: {str(e)}")
                    st.error(f"❌ Invalid Open Until Date: {str(e)}")
                    return {"error": "Invalid Open Until Date"}, ""
                    
                filtered_df = df[df['Status'] == 'Open'].copy()
                filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(
                    filtered_df['Created Date (WET)'])).dt.days
                filtered_df = filtered_df[filtered_df['Days_From_Today'] > 21].copy()

            if filtered_df.empty:
                st.warning(f"No {report_type} NCRs found with duration > 21 days.")
                return {"error": f"No {report_type} records found"}, ""

            # ===== PROCESS RECORDS - NO DEDUPLICATION =====
            processed_data = filtered_df.to_dict(orient="records")
            all_results = {report_type: {"Sites": {}, "Grand_Total": 0}}
            
            st.write(f"Total records to process: {len(processed_data)} (including duplicates)")
            logger.info(f"Processing {len(processed_data)} records (duplicates preserved)")

            for idx, record in enumerate(processed_data):
                try:
                    cleaned_record = {
                        "Description": str(record.get("Description", "")),
                        "Discipline": str(record.get("Discipline", "")),
                        "Created Date (WET)": str(record.get("Created Date (WET)", "")),
                        "Expected Close Date (WET)": str(record.get("Expected Close Date (WET)", "")),
                        "Status": str(record.get("Status", "")),
                        "Days": int(record.get("Days", 0)) if pd.notna(record.get("Days")) else 0,
                        "Record_Index": idx  # Track original position
                    }
                    
                    if report_type == "Open":
                        cleaned_record["Days_From_Today"] = int(record.get("Days_From_Today", 0)) if pd.notna(record.get("Days_From_Today")) else 0

                    description = cleaned_record["Description"]
                    discipline = cleaned_record["Discipline"].strip().lower()
                    
                    # Skip invalid records
                    if discipline == "none" or not discipline or "hse" in discipline:
                        continue
                    
                    # Categorize discipline
                    if "structure" in discipline or "sw" in discipline:
                        cleaned_record["Discipline_Category"] = "SW"
                    elif "civil" in discipline or "finishing" in discipline or "fw" in discipline:
                        cleaned_record["Discipline_Category"] = "FW"
                    else:
                        cleaned_record["Discipline_Category"] = "MEP"

                    modules = extract_modules_from_description(description)
                    cleaned_record["Modules"] = modules
                    
                    tower = determine_tower_assignment(description)
                    cleaned_record["Tower"] = tower

                    # ===== ADD TO RESULTS (KEEP ALL OCCURRENCES) =====
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
                            "Total": 0,
                            "ModulesCount": {},
                            "Record_Indices": []  # Track which records belong here
                        }
                    
                    site_data = all_results[report_type]["Sites"][tower]
                    site_data["Descriptions"].append(cleaned_record["Description"])
                    site_data["Created Date (WET)"].append(cleaned_record["Created Date (WET)"])
                    site_data["Expected Close Date (WET)"].append(cleaned_record["Expected Close Date (WET)"])
                    site_data["Status"].append(cleaned_record["Status"])
                    site_data["Discipline"].append(cleaned_record["Discipline"])
                    site_data["Modules"].append(cleaned_record["Modules"])
                    site_data["Record_Indices"].append(idx)
                    
                    disc_cat = cleaned_record["Discipline_Category"]
                    if disc_cat in ["SW", "FW", "MEP"]:
                        site_data[disc_cat] += 1
                    site_data["Total"] += 1
                    
                    for mod in cleaned_record["Modules"]:
                        site_data["ModulesCount"][mod] = site_data["ModulesCount"].get(mod, 0) + 1
                    
                    all_results[report_type]["Grand_Total"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing record {idx}: {str(e)}")
                    continue

            # ===== DISPLAY SUMMARY TABLE =====
            table_data = []
            for site, data in all_results[report_type]["Sites"].items():
                row = {
                    "Site": site,
                    "SW Count": data["SW"],
                    "FW Count": data["FW"],
                    "MEP Count": data["MEP"],
                    "Total Records": data["Total"],
                    "Unique Descriptions": len(set(data["Descriptions"])),
                    "Duplicate Count": data["Total"] - len(set(data["Descriptions"]))
                }
                table_data.append(row)
            
            if table_data:
                df_table = pd.DataFrame(table_data)
                st.write(f"### {report_type} NCR Summary (Duplicates Preserved)")
                st.dataframe(df_table, use_container_width=True)

            return all_results, json.dumps(all_results, default=str)

    except Exception as e:
        error_msg = f"❌ Error in generate_ncr_report: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return {"error": str(e)}, ""


# ============================================================================
# EXCEL EXPORT WITH DUPLICATE TRACKING
# ============================================================================

def generate_ncr_excel_with_duplicates(all_results, report_type="Closed"):
    """Generate Excel file preserving all duplicate records."""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        title_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': 'yellow', 
            'border': 1, 'font_size': 12
        })
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        cell_format = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        description_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        
        # Summary Sheet
        ws_summary = workbook.add_worksheet('Summary')
        ws_summary.set_column('A:F', 15)
        
        ws_summary.merge_range('A1:F1', f'{report_type} NCR Report - All Records (No Deduplication)', title_format)
        
        headers = ['Site', 'SW Count', 'FW Count', 'MEP Count', 'Total', 'Unique Descriptions']
        for col, header in enumerate(headers):
            ws_summary.write(0, col, header, header_format)
        
        row = 1
        for site, data in all_results[report_type]["Sites"].items():
            ws_summary.write(row, 0, site, cell_format)
            ws_summary.write(row, 1, data["SW"], cell_format)
            ws_summary.write(row, 2, data["FW"], cell_format)
            ws_summary.write(row, 3, data["MEP"], cell_format)
            ws_summary.write(row, 4, data["Total"], cell_format)
            ws_summary.write(row, 5, len(set(data["Descriptions"])), cell_format)
            row += 1
        
        # Detail Sheet - ALL Records with Occurrence Number
        ws_detail = workbook.add_worksheet('All Records')
        ws_detail.set_column('A:A', 5)
        ws_detail.set_column('B:B', 20)
        ws_detail.set_column('C:C', 50)
        ws_detail.set_column('D:G', 18)
        
        ws_detail.merge_range('A1:G1', f'{report_type} NCR - All Records with Duplicates Highlighted', title_format)
        
        headers = ['#', 'Site', 'Description', 'Created Date', 'Expected Close Date', 'Status', 'Discipline']
        for col, header in enumerate(headers):
            ws_detail.write(0, col, header, header_format)
        
        row = 1
        record_counter = 1
        
        for site, data in all_results[report_type]["Sites"].items():
            for i, desc in enumerate(data["Descriptions"]):
                ws_detail.write(row, 0, record_counter, cell_format)
                ws_detail.write(row, 1, site, cell_format)
                ws_detail.write(row, 2, desc, description_format)
                ws_detail.write(row, 3, data["Created Date (WET)"][i] if i < len(data["Created Date (WET)"]) else "", cell_format)
                ws_detail.write(row, 4, data["Expected Close Date (WET)"][i] if i < len(data["Expected Close Date (WET)"]) else "", cell_format)
                ws_detail.write(row, 5, data["Status"][i] if i < len(data["Status"]) else "", cell_format)
                ws_detail.write(row, 6, data["Discipline"][i] if i < len(data["Discipline"]) else "", cell_format)
                row += 1
                record_counter += 1
    
    output.seek(0)
    return output


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_report_title(prefix):
    """Generate timestamped report title."""
    now = datetime.now()
    day = now.strftime("%d")
    month_name = now.strftime("%B")
    year = now.strftime("%Y")
    return f"{prefix}: {day}_{month_name}_{year}"
