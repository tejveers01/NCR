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
from typing import Tuple, Dict, Any

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

WATSONX_API_URL = os.getenv("WATSONX_API_URL")
MODEL_ID = os.getenv("MODEL_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")

if not all([API_KEY, WATSONX_API_URL, MODEL_ID, PROJECT_ID]):
    st.error("❌ Required environment variables missing!")
    logger.error("Missing required environment variables")
    st.stop()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOGIN_URL = "https://dms.asite.com/apilogin/"
SEARCH_URL = "https://adoddleak.asite.com/commonapi/formsearchapi/search"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def get_access_token(API_KEY):
    """Generate WatsonX access token"""
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": API_KEY}
    try:
        response = requests.post(IAM_TOKEN_URL, headers=headers, data=data, verify=certifi.where(), timeout=50)
        if response.status_code == 200:
            token_info = response.json()
            logger.info("✅ Access token generated successfully")
            return token_info['access_token']
        else:
            logger.error(f"❌ Failed to get access token: {response.status_code}")
            st.error(f"❌ Failed to get access token: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ Exception getting access token: {str(e)}")
        st.error(f"❌ Error getting access token: {str(e)}")
        return None

def login_to_asite(email, password):
    """Login to Asite and get session ID"""
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"emailId": email, "password": password}
    response = requests.post(LOGIN_URL, headers=headers, data=payload, verify=certifi.where(), timeout=50)
    if response.status_code == 200:
        try:
            session_id = response.json().get("UserProfile", {}).get("Sessionid")
            logger.info(f"✅ Login successful, Session ID: {session_id}")
            return session_id
        except json.JSONDecodeError:
            logger.error("❌ JSONDecodeError during login")
            st.error("❌ Failed to parse login response")
            return None
    logger.error(f"❌ Login failed: {response.status_code}")
    st.error(f"❌ Login failed: {response.status_code}")
    return None

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_project_data(session_id, project_name, form_name, record_limit=1000):
    """Fetch data from Asite API"""
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded", "Cookie": f"ASessionID={session_id}"}
    all_data = []
    start_record = 1
    total_records = None

    start_time = datetime.now()
    st.write(f"🔄 Fetching data from Asite started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Started fetching data for project '{project_name}' at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    with st.spinner("📥 Fetching data from Asite..."):
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
            response = requests.post(SEARCH_URL, headers=headers, data=encoded_payload, verify=certifi.where(), timeout=50)

            try:
                response_json = response.json()
                if total_records is None:
                    total_records = response_json.get("responseHeader", {}).get("results-total", 0)
                all_data.extend(response_json.get("FormList", {}).get("Form", []))
                st.info(f"🔄 Fetched {len(all_data)} / {total_records} records")
                if start_record + record_limit - 1 >= total_records:
                    break
                start_record += record_limit
            except Exception as e:
                logger.error(f"❌ Error fetching data: {str(e)}")
                st.error(f"❌ Error fetching data: {str(e)}")
                break

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    st.write(f"✅ Fetching completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration}s)")
    logger.info(f"Finished fetching data (Duration: {duration}s)")

    return {"responseHeader": {"results": len(all_data), "total_results": total_records}}, all_data, encoded_payload

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_json_data(json_data):
    """Process JSON data from Asite"""
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

    df = pd.DataFrame(data, columns=['Days', 'Created Date (WET)', 'Expected Close Date (WET)', 'Description', 'Status', 'Discipline'])
    df['Created Date (WET)'] = pd.to_datetime(df['Created Date (WET)'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')
    df['Expected Close Date (WET)'] = pd.to_datetime(df['Expected Close Date (WET)'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')
    logger.debug(f"DataFrame processed: {len(df)} rows, {df.columns.tolist()}")
    if df.empty:
        logger.warning("⚠️ DataFrame is empty after processing")
        st.warning("⚠️ No data processed. Check the API response.")
    return df

def clean_and_parse_json(text):
    """Clean and parse JSON from text"""
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
            logger.error(f"Failed to parse JSON: {json_str[:100]}")
            
    logger.error(f"Could not extract valid JSON")
    return None

# ============================================================================
# DUPLICATE TRACKING AND DISPLAY
# ============================================================================

def display_duplicate_report(duplicate_stats, report_name="Records"):
    """Display duplicate analysis in Streamlit"""
    if duplicate_stats["duplicate_records"] > 0:
        st.warning(f"⚠️ {report_name}: Found {duplicate_stats['duplicate_records']} duplicates out of {duplicate_stats['total_records']} total")
        
        with st.expander(f"📊 {report_name} Duplicate Analysis ({duplicate_stats['duplicate_records']} duplicates)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", duplicate_stats["total_records"])
            with col2:
                st.metric("Unique Records", duplicate_stats["unique_records"])
            with col3:
                st.metric("Duplicate Found", duplicate_stats["duplicate_records"])
            
            if duplicate_stats["duplicates"]:
                dup_display = []
                for dup in duplicate_stats["duplicates"]:
                    record = dup["record"]
                    dup_display.append({
                        "ID": dup["unique_id"][:40],
                        "Occurrence": dup["occurrence"],
                        "Description": str(record.get("Description", ""))[:40],
                        "Created Date": str(record.get("Created Date (WET)", ""))[:10],
                        "Status": record.get("Status", "")
                    })
                
                dup_df = pd.DataFrame(dup_display)
                st.dataframe(dup_df, use_container_width=True)
                
                if st.button(f"📥 Export {report_name} Duplicates", key=f"export_dup_{report_name}"):
                    csv = dup_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {report_name} Duplicates CSV",
                        data=csv,
                        file_name=f"{report_name.lower()}_duplicates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    else:
        st.success(f"✅ {report_name}: All {duplicate_stats['total_records']} records are unique (No duplicates)")

# ============================================================================
# MODULE AND TOWER FUNCTIONS
# ============================================================================

def extract_modules_from_description(description):
    """Extract module numbers from description"""
    description_lower = description.lower()
    modules = set()

    # Pattern 1: Ranges like "Module 1 to 3"
    range_patterns = r"(?:module|mod|m)[-\s]*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))\s*(?:to|-|–)\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))"
    for start_str, end_str in re.findall(range_patterns, description_lower, re.IGNORECASE):
        try:
            start, end = int(start_str), int(end_str)
            if 0 < start <= end <= 50:
                modules.update(f"Module {i}" for i in range(start, end + 1))
        except ValueError:
            continue

    # Pattern 2: Combinations like "Module 1 & 2"
    combination_patterns = [
        r"module[-\s]*(?:–|-)?\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))\s*[&and]+\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))",
        r"module[-\s]*(?:–|-)?\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))\s*[,&]\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))",
    ]
    for pattern in combination_patterns:
        for match in re.findall(pattern, description_lower, re.IGNORECASE):
            try:
                num1, num2 = int(match[0]), int(match[1])
                if 1 <= num1 <= 50:
                    modules.add(f"Module {num1}")
                if 1 <= num2 <= 50:
                    modules.add(f"Module {num2}")
            except ValueError:
                continue

    # Pattern 3: Grouped modules
    list_pattern = r"(?:module|mod|m)[-\s]*((?:\d+\s*(?:,|&|and)?\s*)+)(?!\s*(?:mm|th|rd|nd|st|floor))"
    for match in re.findall(list_pattern, description_lower, re.IGNORECASE):
        for num in re.findall(r"\b\d{1,2}\b", match):
            try:
                num = int(num)
                if 0 < num <= 50:
                    modules.add(f"Module {num}")
            except ValueError:
                continue

    # Pattern 4: Single modules
    individual_patterns = r"(?:module|mod|m)[-\s]*(\d{1,2})(?!\s*(?:mm|th|rd|nd|st|floor))"
    for num_str in re.findall(individual_patterns, description_lower, re.IGNORECASE):
        try:
            num = int(num_str)
            if 0 < num <= 50:
                modules.add(f"Module {num}")
        except ValueError:
            continue

    if "corridor" in description_lower and modules:
        return sorted(list(modules))

    if not modules:
        common_patterns = [r"steel\s+yard", r"qc\s+lab", r"cipl", r"nta\s+beam", r"non\s+tower"]
        for pattern in common_patterns:
            if re.search(pattern, description_lower, re.IGNORECASE):
                return ["Common"]
    
    return sorted(list(modules)) if modules else ["Common"]

def determine_tower_assignment(description):
    """Determine tower assignment from description"""
    description_lower = description.lower()
    
    if any(phrase in description_lower for phrase in ["eligo clubhouse", "eligo-clubhouse", "eligo club"]):
        return "Eligo-Club"

    tower_matches = re.findall(r"\b(?:tower|t)\s*[-\s(]*([fgh])\b", description_lower, re.IGNORECASE)
    tower_bracket_matches = re.findall(r"tower\s*\(\s*([fgh])\s*\)", description_lower, re.IGNORECASE)
    all_tower_matches = tower_matches + tower_bracket_matches
    
    has_module = re.search(r"module\s*[-\s]*\d+", description_lower, re.IGNORECASE)
    
    general_common = ["steel yard", "qc lab", "cipl", "nta beam", "non tower"]
    tower_common = ["lift lobby", "corridor", "staircase"]
    structural = ["lift wall", "shear wall", "beam", "column", "slab", "foundation"]
    
    is_general = any(ind in description_lower for ind in general_common)
    is_tower_common_area = any(ind in description_lower for ind in tower_common)
    is_structural = any(elem in description_lower for elem in structural)

    if is_general and not all_tower_matches:
        return "Common_Area"

    if all_tower_matches:
        tower_letter = all_tower_matches[0].upper()
        if is_structural or has_module:
            return f"Eligo-Tower-{tower_letter}"
        elif is_tower_common_area:
            return f"Eligo-Tower-{tower_letter}-CommonArea"
        else:
            return f"Eligo-Tower-{tower_letter}"
    else:
        return "Common_Area"

# ============================================================================
# NCR REPORT GENERATION - WITH DUPLICATE TRACKING
# ============================================================================

@st.cache_data
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type((requests.RequestException, ValueError, KeyError)))
def generate_ncr_report_for_eligo(df: pd.DataFrame, report_type: str, start_date=None, end_date=None, Until_Date=None) -> Tuple[Dict[str, Any], str]:
    """Generate NCR report with duplicate tracking"""
    try:
        with st.spinner(f"Generating {report_type} NCR Report..."):
            if df is None or df.empty:
                st.error("❌ Input DataFrame is empty or None")
                return {"error": "Empty DataFrame"}, ""
            
            if report_type not in ["Open", "Closed"]:
                st.error(f"❌ Invalid report_type: {report_type}")
                return {"error": "Invalid report_type"}, ""
            
            df = df.copy()
            required_columns = ['Created Date (WET)', 'Status']
            if not all(col in df.columns for col in required_columns):
                st.error(f"❌ Missing required columns")
                return {"error": "Missing columns"}, ""
            
            df = df[df['Created Date (WET)'].notna()]
            
            if report_type == "Closed":
                if 'Expected Close Date (WET)' not in df.columns:
                    st.error("❌ 'Expected Close Date (WET)' column required for Closed reports")
                    return {"error": "Missing Expected Close Date column"}, ""
                
                start_date = pd.to_datetime(start_date) if start_date else df['Created Date (WET)'].min()
                end_date = pd.to_datetime(end_date) if end_date else df['Expected Close Date (WET)'].max()
                
                df = df[df['Expected Close Date (WET)'].notna()]
                if 'Days' not in df.columns:
                    df['Days'] = (pd.to_datetime(df['Expected Close Date (WET)']) - pd.to_datetime(df['Created Date (WET)'])).dt.days
                
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
                
                today = pd.to_datetime(Until_Date)
                filtered_df = df[(df['Status'] == 'Open') & (df['Created Date (WET)'].notna())].copy()
                filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
                filtered_df = filtered_df[filtered_df['Days_From_Today'] > 21].copy()

            if filtered_df.empty:
                st.warning(f"No {report_type} NCRs found with duration > 21 days")
                return {"error": f"No {report_type} records found"}, ""

            filtered_df.loc[:, 'Created Date (WET)'] = filtered_df['Created Date (WET)'].astype(str)
            if 'Expected Close Date (WET)' in filtered_df.columns:
                filtered_df.loc[:, 'Expected Close Date (WET)'] = filtered_df['Expected Close Date (WET)'].astype(str)

            processed_data = filtered_df.to_dict(orient="records")
            
            # ========== DUPLICATE TRACKING START ==========
            cleaned_data = []
            duplicate_stats = {
                "total_records": len(processed_data),
                "unique_records": 0,
                "duplicate_records": 0,
                "duplicates": [],
                "record_count": {}
            }
            # ========== DUPLICATE TRACKING END ==========
            
            for record in processed_data:
                try:
                    cleaned_record = {
                        "Description": str(record.get("Description", "")),
                        "Discipline": str(record.get("Discipline", "")),
                        "Created Date (WET)": str(record.get("Created Date (WET)", "")),
                        "Expected Close Date (WET)": str(record.get("Expected Close Date (WET)", "")),
                        "Status": str(record.get("Status", "")),
                        "Days": int(record.get("Days", 0)) if pd.notna(record.get("Days")) else 0,
                    }
                    
                    if report_type == "Open":
                        cleaned_record["Days_From_Today"] = int(record.get("Days_From_Today", 0)) if pd.notna(record.get("Days_From_Today")) else 0

                    description = cleaned_record["Description"]
                    unique_id = f"{description}_{cleaned_record['Created Date (WET)']}_{cleaned_record['Status']}"
                    
                    # ========== TRACK DUPLICATES ==========
                    if unique_id in duplicate_stats["record_count"]:
                        duplicate_stats["record_count"][unique_id] += 1
                        duplicate_stats["duplicates"].append({
                            "unique_id": unique_id,
                            "occurrence": duplicate_stats["record_count"][unique_id],
                            "record": cleaned_record
                        })
                        logger.warning(f"Duplicate NCR found (#{duplicate_stats['record_count'][unique_id]}): {unique_id[:60]}")
                    else:
                        duplicate_stats["record_count"][unique_id] = 1
                    # ========== END TRACKING ==========
                    
                    modules = extract_modules_from_description(description)
                    cleaned_record["Modules"] = modules
                    
                    discipline = cleaned_record["Discipline"].strip().lower()
                    if discipline == "none" or not discipline:
                        continue
                    elif "hse" in discipline:
                        cleaned_record["Discipline_Category"] = "HSE"
                        continue
                    elif "structure" in discipline or "sw" in discipline:
                        cleaned_record["Discipline_Category"] = "SW"
                    elif "civil" in discipline or "finishing" in discipline or "fw" in discipline:
                        cleaned_record["Discipline_Category"] = "FW"
                    else:
                        cleaned_record["Discipline_Category"] = "MEP"

                    tower_assignment = determine_tower_assignment(description)
                    if isinstance(tower_assignment, tuple):
                        for tower in tower_assignment:
                            record_copy = cleaned_record.copy()
                            record_copy["Tower"] = tower
                            cleaned_data.append(record_copy)
                    else:
                        cleaned_record["Tower"] = tower_assignment
                        cleaned_data.append(cleaned_record)
                        
                except Exception as e:
                    logger.error(f"Error processing record: {str(e)}")
                    continue

            # ========== DISPLAY DUPLICATE REPORT ==========
            duplicate_stats["unique_records"] = len(duplicate_stats["record_count"])
            duplicate_stats["duplicate_records"] = len(duplicate_stats["duplicates"])
            display_duplicate_report(duplicate_stats, f"NCR - {report_type}")
            
            st.write(f"📊 Summary: {duplicate_stats['unique_records']} unique records (Total: {duplicate_stats['total_records']})")
            logger.info(f"NCR {report_type}: {len(cleaned_data)} records, {duplicate_stats['duplicate_records']} duplicates")

            if not cleaned_data:
                return {"error": "No valid records after processing"}, ""

            # Get access token
            access_token = get_access_token(API_KEY)
            if not access_token:
                return {"error": "Failed to obtain access token"}, ""

            all_results = {report_type: {"Sites": {}, "Grand_Total": 0}}
            chunk_size = int(os.getenv("CHUNK_SIZE", 15))

            for i in range(0, len(cleaned_data), chunk_size):
                chunk = cleaned_data[i:i + chunk_size]
                chunk_num = i // chunk_size + 1
                
                start_chunk_time = datetime.now()
                st.write(f"🔄 Processing chunk {chunk_num} ({len(chunk)} records) at {start_chunk_time.strftime('%H:%M:%S')}")
                logger.info(f"Started chunk {chunk_num}")
                
                prompt = (
                    f"CRITICAL: Process exactly {len(chunk)} records. Group by 'Tower' field. "
                    f"Count records by 'Discipline_Category' (SW/FW/MEP). "
                    f"Keep 'Common Description' modules as is. Return ONLY valid JSON:\n"
                    f'{{"Report_Type": {{"Sites": {{"Tower_Name": {{"Descriptions": [], "Created Date (WET)": [], '
                    f'"Expected Close Date (WET)": [], "Status": [], "Discipline": [], "Modules": [], "SW": 0, "FW": 0, "MEP": 0, '
                    f'"Total": 0, "ModulesCount": {{}}}}}}, "Grand_Total": 0}}}}\n\n'
                    f"Data: {json.dumps(chunk)}\n"
                )

                payload = {
                    "input": prompt,
                    "parameters": {
                        "decoding_method": "greedy",
                        "max_new_tokens": 8100,
                        "temperature": 0.0
                    },
                    "model_id": MODEL_ID,
                    "project_id": PROJECT_ID
                }
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {access_token}"
                }

                try:
                    response = requests.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=600)
                    
                    if response.status_code == 200:
                        api_result = response.json()
                        generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                        parsed_json = clean_and_parse_json(generated_text)
                        
                        if parsed_json and report_type in parsed_json:
                            chunk_result = parsed_json[report_type]
                            for site, data in chunk_result.get("Sites", {}).items():
                                if site not in all_results[report_type]["Sites"]:
                                    all_results[report_type]["Sites"][site] = {
                                        "Descriptions": [], "Created Date (WET)": [], "Expected Close Date (WET)": [],
                                        "Status": [], "Discipline": [], "Modules": [],
                                        "SW": 0, "FW": 0, "MEP": 0, "Total": 0, "ModulesCount": {}
                                    }
                                
                                all_results[report_type]["Sites"][site]["Descriptions"].extend(data.get("Descriptions", []))
                                all_results[report_type]["Sites"][site]["Created Date (WET)"].extend(data.get("Created Date (WET)", []))
                                all_results[report_type]["Sites"][site]["Expected Close Date (WET)"].extend(data.get("Expected Close Date (WET)", []))
                                all_results[report_type]["Sites"][site]["Status"].extend(data.get("Status", []))
                                all_results[report_type]["Sites"][site]["Discipline"].extend(data.get("Discipline", []))
                                all_results[report_type]["Sites"][site]["Modules"].extend(data.get("Modules", []))
                                all_results[report_type]["Sites"][site]["SW"] += data.get("SW", 0)
                                all_results[report_type]["Sites"][site]["FW"] += data.get("FW", 0)
                                all_results[report_type]["Sites"][site]["MEP"] += data.get("MEP", 0)
                                all_results[report_type]["Sites"][site]["Total"] += data.get("Total", 0)
                                
                                for module, count in data.get("ModulesCount", {}).items():
                                    all_results[report_type]["Sites"][site]["ModulesCount"][module] = all_results[report_type]["Sites"][site]["ModulesCount"].get(module, 0) + count
                            
                            all_results[report_type]["Grand_Total"] += chunk_result.get("Grand_Total", 0)
                            st.success(f"✅ Chunk {chunk_num} processed successfully")
                        else:
                            logger.error(f"Invalid JSON format for chunk {chunk_num}")
                            st.error(f"❌ Invalid JSON for chunk {chunk_num}")
                    else:
                        logger.error(f"WatsonX API error: {response.status_code}")
                        st.error(f"❌ WatsonX API error: {response.status_code}")
                        
                except requests.RequestException as e:
                    logger.error(f"Request error for chunk {chunk_num}: {str(e)}")
                    st.error(f"❌ Request error: {str(e)}")

            sites_to_remove = [site for site, data in all_results[report_type]["Sites"].items() if data["Total"] == 0]
            for site in sites_to_remove:
                del all_results[report_type]["Sites"][site]

            end_time = datetime.now()
            duration = (end_time - start_chunk_time).total_seconds()
            st.write(f"✅ Report generation completed at {end_time.strftime('%H:%M:%S')} (Duration: {duration}s)")
            logger.info(f"Finished NCR {report_type} report generation")

            return all_results, json.dumps(all_results)

    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return {"error": str(e)}, ""


# ============================================================================
# HOUSEKEEPING REPORT - WITH DUPLICATE TRACKING
# ============================================================================

@st.cache_data
def generate_ncr_Housekeeping_report_for_eligo(df, report_type, start_date=None, end_date=None, until_date=None):
    """Generate Housekeeping NCR report with duplicate tracking"""
    with st.spinner(f"Generating {report_type} Housekeeping NCR Report..."):
        try:
            today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
            
            housekeeping_keywords = [
                'housekeeping', 'cleaning', 'cleanliness', 'waste disposal', 'waste management', 'garbage', 'trash',
                'rubbish', 'debris', 'litter', 'dust', 'untidy', 'cluttered', 'accumulation of waste',
                'construction waste', 'pile of garbage', 'poor housekeeping', 'material storage',
                'construction debris', 'cleaning schedule', 'garbage collection', 'waste bins', 'dirty',
                'mess', 'unclean', 'disorderly', 'dirty floor', 'waste disposal area', 'waste collection',
                'cleaning protocol', 'sanitation', 'trash removal', 'waste accumulation', 'unkept area',
                'refuse collection', 'workplace cleanliness'
            ]

            def is_housekeeping_record(description):
                if description is None or not isinstance(description, str):
                    return False
                return any(keyword in description.lower() for keyword in housekeeping_keywords)

            if report_type == "Closed":
                filtered_df = df[
                    (df['Discipline'] == 'HSE') &
                    (df['Status'] == 'Closed') &
                    (df['Days'].notnull()) &
                    (df['Days'] > 7) &
                    (df['Description'].apply(is_housekeeping_record))
                ].copy()
                if start_date and end_date:
                    filtered_df = filtered_df[
                        (pd.to_datetime(filtered_df['Created Date (WET)']) >= pd.to_datetime(start_date)) &
                        (pd.to_datetime(filtered_df['Expected Close Date (WET)']) <= pd.to_datetime(end_date))
                    ].copy()
            else:  # Open
                filtered_df = df[
                    (df['Discipline'] == 'HSE') &
                    (df['Status'] == 'Open') &
                    (pd.to_datetime(df['Created Date (WET)']).notna()) &
                    (df['Description'].apply(is_housekeeping_record))
                ].copy()
                filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
                filtered_df = filtered_df[filtered_df['Days_From_Today'] > 7].copy()
                if until_date:
                    filtered_df = filtered_df[pd.to_datetime(filtered_df['Created Date (WET)']) <= pd.to_datetime(until_date)].copy()

            if filtered_df.empty:
                st.warning(f"No {report_type} Housekeeping records found")
                return {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}, ""

            filtered_df.loc[:, 'Created Date (WET)'] = filtered_df['Created Date (WET)'].astype(str)
            filtered_df.loc[:, 'Expected Close Date (WET)'] = filtered_df['Expected Close Date (WET)'].astype(str)

            processed_data = filtered_df.to_dict(orient="records")
            
            # ========== DUPLICATE TRACKING ==========
            cleaned_data = []
            seen_descriptions = {}
            duplicate_stats = {
                "total_records": len(processed_data),
                "unique_records": 0,
                "duplicate_records": 0,
                "duplicates": []
            }
            # ========== END TRACKING ==========
            
            for record in processed_data:
                description = str(record.get("Description", "")).strip()
                if description:
                    # ========== TRACK DUPLICATES ==========
                    if description in seen_descriptions:
                        seen_descriptions[description] += 1
                        duplicate_stats["duplicates"].append({
                            "description": description[:50],
                            "occurrence": seen_descriptions[description],
                            "created_date": str(record.get("Created Date (WET)", ""))[:10],
                            "status": str(record.get("Status", ""))
                        })
                        logger.warning(f"Housekeeping duplicate (#{seen_descriptions[description]}): {description[:40]}")
                    else:
                        seen_descriptions[description] = 1
                    # ========== END ==========
                    
                    cleaned_record = {
                        "Description": description,
                        "Created Date (WET)": str(record.get("Created Date (WET)", "")),
                        "Expected Close Date (WET)": str(record.get("Expected Close Date (WET)", "")),
                        "Status": str(record.get("Status", "")),
                        "Days": record.get("Days", 0),
                        "Discipline": "HSE",
                        "Tower": "External Development"
                    }
                    if report_type == "Open":
                        cleaned_record["Days_From_Today"] = record.get("Days_From_Today", 0)

                    desc_lower = description.lower()
                    tower_match = re.search(r"(tower|t)\s*-?\s*([A-Za-z])", desc_lower, re.IGNORECASE)
                    cleaned_record["Tower"] = f"Eligo-Tower-{tower_match.group(2).upper()}" if tower_match else "Common_Area"
                    
                    cleaned_data.append(cleaned_record)

            # ========== DISPLAY DUPLICATE REPORT ==========
            duplicate_stats["unique_records"] = len(seen_descriptions)
            duplicate_stats["duplicate_records"] = len(duplicate_stats["duplicates"])
            display_duplicate_report(duplicate_stats, f"Housekeeping - {report_type}")
            
            st.write(f"📊 Housekeeping: {duplicate_stats['unique_records']} unique records (Total: {duplicate_stats['total_records']})")
            logger.info(f"Housekeeping {report_type}: {len(cleaned_data)} records, {duplicate_stats['duplicate_records']} duplicates")

            if not cleaned_data:
                return {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}, ""

            access_token = get_access_token(API_KEY)
            if not access_token:
                return {"error": "Failed to obtain access token"}, ""

            result = {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}
            chunk_size = 10

            for i in range(0, len(cleaned_data), chunk_size):
                chunk = cleaned_data[i:i + chunk_size]
                chunk_num = i // chunk_size + 1
                
                st.write(f"🔄 Processing Housekeeping chunk {chunk_num}")

                prompt = (
                    f"Count Housekeeping NCRs by 'Tower' where 'Discipline' is 'HSE' and 'Days' > 7. "
                    f"Collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status' into arrays. "
                    f"Return ONLY valid JSON:\n"
                    f'{{"Housekeeping": {{"Sites": {{"Site_Name": {{"Descriptions": [], "Created Date (WET)": [], '
                    f'"Expected Close Date (WET)": [], "Status": [], "Count": 0}}}}, "Grand_Total": 0}}}}\n\n'
                    f"Data: {json.dumps(chunk)}\n"
                )

                payload = {
                    "input": prompt,
                    "parameters": {
                        "decoding_method": "greedy",
                        "max_new_tokens": 500,
                        "temperature": 0.0
                    },
                    "model_id": MODEL_ID,
                    "project_id": PROJECT_ID
                }
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {access_token}"
                }

                try:
                    response = requests.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=60)
                    
                    if response.status_code == 200:
                        api_result = response.json()
                        generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                        parsed_json = clean_and_parse_json(generated_text)
                        
                        if parsed_json and "Housekeeping" in parsed_json:
                            chunk_result = parsed_json["Housekeeping"]
                            for site, values in chunk_result.get("Sites", {}).items():
                                if site not in result["Housekeeping"]["Sites"]:
                                    result["Housekeeping"]["Sites"][site] = {
                                        "Count": 0,
                                        "Descriptions": [],
                                        "Created Date (WET)": [],
                                        "Expected Close Date (WET)": [],
                                        "Status": []
                                    }
                                result["Housekeeping"]["Sites"][site]["Descriptions"].extend(values.get("Descriptions", []))
                                result["Housekeeping"]["Sites"][site]["Created Date (WET)"].extend(values.get("Created Date (WET)", []))
                                result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"].extend(values.get("Expected Close Date (WET)", []))
                                result["Housekeeping"]["Sites"][site]["Status"].extend(values.get("Status", []))
                                result["Housekeeping"]["Sites"][site]["Count"] += values.get("Count", 0)
                            result["Housekeeping"]["Grand_Total"] += chunk_result.get("Grand_Total", 0)
                            st.success(f"✅ Housekeeping chunk {chunk_num} processed")
                except Exception as e:
                    logger.error(f"Error processing housekeeping chunk {chunk_num}: {str(e)}")
                    st.error(f"❌ Housekeeping chunk {chunk_num} error")

            return result, json.dumps(result)
            
        except Exception as e:
            logger.error(f"Housekeeping report error: {str(e)}")
            st.error(f"❌ Housekeeping Error: {str(e)}")
            return {"error": str(e)}, ""


# ============================================================================
# SAFETY REPORT - WITH DUPLICATE TRACKING
# ============================================================================

@st.cache_data
def generate_ncr_Safety_report_for_eligo(df, report_type, start_date=None, end_date=None, until_date=None):
    """Generate Safety NCR report with duplicate tracking"""
    with st.spinner(f"Generating {report_type} Safety NCR Report..."):
        try:
            today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
            
            safety_keywords = [
                'safety precautions', 'temporary electricity', 'safety norms', 'safety belt', 'helmet',
                'lifeline', 'guard rails', 'fall protection', 'PPE', 'electrical hazard', 'unsafe platform',
                'catch net', 'edge protection', 'TPI', 'scaffold', 'lifting equipment', 'dust suppression',
                'debris chute', 'spill control', 'crane operator', 'halogen lamps', 'fall catch net',
                'working at height', 'PPE kit', 'HSE norms', 'negligence in supervision', 'violation of HSE',
                'environmental contamination', 'fire hazard', 'non-tower area', 'nta'
            ]

            def is_safety_record(description):
                if description is None or not isinstance(description, str):
                    return False
                return any(keyword in description.lower() for keyword in safety_keywords)

            if report_type == "Closed":
                filtered_df = df[
                    (df['Discipline'] == 'HSE') &
                    (df['Status'] == 'Closed') &
                    (df['Days'].notnull()) &
                    (df['Days'] > 7) &
                    (df['Description'].apply(is_safety_record))
                ].copy()
                if start_date and end_date:
                    filtered_df = filtered_df[
                        (pd.to_datetime(filtered_df['Created Date (WET)']) >= pd.to_datetime(start_date)) &
                        (pd.to_datetime(filtered_df['Expected Close Date (WET)']) <= pd.to_datetime(end_date))
                    ].copy()
            else:  # Open
                filtered_df = df[
                    (df['Discipline'] == 'HSE') &
                    (df['Status'] == 'Open') &
                    (pd.to_datetime(df['Created Date (WET)']).notna()) &
                    (df['Description'].apply(is_safety_record))
                ].copy()
                filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
                filtered_df = filtered_df[filtered_df['Days_From_Today'] > 7].copy()
                if until_date:
                    filtered_df = filtered_df[pd.to_datetime(filtered_df['Created Date (WET)']) <= pd.to_datetime(until_date)].copy()

            if filtered_df.empty:
                st.warning(f"No {report_type} Safety records found")
                return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

            filtered_df.loc[:, 'Created Date (WET)'] = filtered_df['Created Date (WET)'].astype(str)
            filtered_df.loc[:, 'Expected Close Date (WET)'] = filtered_df['Expected Close Date (WET)'].astype(str)

            processed_data = filtered_df.to_dict(orient="records")
            
            # ========== DUPLICATE TRACKING ==========
            cleaned_data = []
            seen_descriptions = {}
            duplicate_stats = {
                "total_records": len(processed_data),
                "unique_records": 0,
                "duplicate_records": 0,
                "duplicates": []
            }
            # ========== END TRACKING ==========
            
            for record in processed_data:
                description = str(record.get("Description", "")).strip()
                if description:
                    # ========== TRACK DUPLICATES ==========
                    if description in seen_descriptions:
                        seen_descriptions[description] += 1
                        duplicate_stats["duplicates"].append({
                            "description": description[:50],
                            "occurrence": seen_descriptions[description],
                            "created_date": str(record.get("Created Date (WET)", ""))[:10],
                            "status": str(record.get("Status", ""))
                        })
                        logger.warning(f"Safety duplicate (#{seen_descriptions[description]}): {description[:40]}")
                    else:
                        seen_descriptions[description] = 1
                    # ========== END ==========
                    
                    days = record.get("Days", 0)
                    days_from_today = record.get("Days_From_Today", 0)
                    
                    cleaned_record = {
                        "Description": description,
                        "Created Date (WET)": str(record.get("Created Date (WET)", "")),
                        "Expected Close Date (WET)": str(record.get("Expected Close Date (WET)", "")),
                        "Status": str(record.get("Status", "")),
                        "Days": days,
                        "Discipline": str(record.get("Discipline", "")),
                        "Tower": "External Development"
                    }
                    if report_type == "Open":
                        cleaned_record["Days_From_Today"] = days_from_today

                    desc_lower = description.lower()
                    tower_match = re.search(r"(tower|t)\s*-?\s*([A-Za-z])", desc_lower, re.IGNORECASE)
                    cleaned_record["Tower"] = f"Eligo-Tower-{tower_match.group(2).upper()}" if tower_match else "Common_Area"
                    
                    cleaned_data.append(cleaned_record)

            # ========== DISPLAY DUPLICATE REPORT ==========
            duplicate_stats["unique_records"] = len(seen_descriptions)
            duplicate_stats["duplicate_records"] = len(duplicate_stats["duplicates"])
            display_duplicate_report(duplicate_stats, f"Safety - {report_type}")
            
            st.write(f"📊 Safety: {duplicate_stats['unique_records']} unique records (Total: {duplicate_stats['total_records']})")
            logger.info(f"Safety {report_type}: {len(cleaned_data)} records, {duplicate_stats['duplicate_records']} duplicates")

            if not cleaned_data:
                return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

            access_token = get_access_token(API_KEY)
            if not access_token:
                return {"error": "Failed to obtain access token"}, ""

            result = {"Safety": {"Sites": {}, "Grand_Total": 0}}
            chunk_size = 10

            for i in range(0, len(cleaned_data), chunk_size):
                chunk = cleaned_data[i:i + chunk_size]
                chunk_num = i // chunk_size + 1
                
                st.write(f"🔄 Processing Safety chunk {chunk_num}")

                prompt = (
                    f"Count Safety NCRs by 'Tower' where 'Discipline' is 'HSE' and description contains safety keywords. "
                    f"Collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status' into arrays. "
                    f"Return ONLY valid JSON:\n"
                    f'{{"Safety": {{"Sites": {{"Site_Name": {{"Descriptions": [], "Created Date (WET)": [], '
                    f'"Expected Close Date (WET)": [], "Status": [], "Count": 0}}}}, "Grand_Total": 0}}}}\n\n'
                    f"Data: {json.dumps(chunk)}\n"
                )

                payload = {
                    "input": prompt,
                    "parameters": {
                        "decoding_method": "greedy",
                        "max_new_tokens": 1500,
                        "temperature": 0.0
                    },
                    "model_id": MODEL_ID,
                    "project_id": PROJECT_ID
                }
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {access_token}"
                }

                try:
                    response = requests.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=60)
                    
                    if response.status_code == 200:
                        api_result = response.json()
                        generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                        parsed_json = clean_and_parse_json(generated_text)
                        
                        if parsed_json and "Safety" in parsed_json:
                            chunk_result = parsed_json["Safety"]
                            for site, values in chunk_result.get("Sites", {}).items():
                                if site not in result["Safety"]["Sites"]:
                                    result["Safety"]["Sites"][site] = {
                                        "Count": 0,
                                        "Descriptions": [],
                                        "Created Date (WET)": [],
                                        "Expected Close Date (WET)": [],
                                        "Status": []
                                    }
                                result["Safety"]["Sites"][site]["Descriptions"].extend(values.get("Descriptions", []))
                                result["Safety"]["Sites"][site]["Created Date (WET)"].extend(values.get("Created Date (WET)", []))
                                result["Safety"]["Sites"][site]["Expected Close Date (WET)"].extend(values.get("Expected Close Date (WET)", []))
                                result["Safety"]["Sites"][site]["Status"].extend(values.get("Status", []))
                                result["Safety"]["Sites"][site]["Count"] += values.get("Count", 0)
                            result["Safety"]["Grand_Total"] += chunk_result.get("Grand_Total", 0)
                            st.success(f"✅ Safety chunk {chunk_num} processed")
                except Exception as e:
                    logger.error(f"Error processing safety chunk {chunk_num}: {str(e)}")
                    st.error(f"❌ Safety chunk {chunk_num} error")

            return result, json.dumps(result)
            
        except Exception as e:
            logger.error(f"Safety report error: {str(e)}")
            st.error(f"❌ Safety Error: {str(e)}")
            return {"error": str(e)}, ""


# ============================================================================
# STREAMLIT MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="NCR Report Generator", layout="wide")
    st.title("📊 NCR Report Generator - With Duplicate Tracking")
    
    st.markdown("""
    This application generates NCR reports from Asite data with comprehensive duplicate tracking.
    **Features:**
    - ✅ Tracks and displays duplicate records
    - 📊 Shows duplicate statistics
    - 💾 Export duplicate reports
    - 🔍 Detailed logging for all duplicates
    """)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    project_name = st.sidebar.text_input("Project Name", value="Eligo")
    form_name = st.sidebar.text_input("Form Name", value="RTA")
    
    if st.sidebar.button("🔐 Login & Fetch Data"):
        session_id = login_to_asite(email, password)
        if session_id:
            header, data, payload = fetch_project_data(session_id, project_name, form_name)
            df = process_json_data(data)
            st.session_state.df = df
            st.success(f"✅ Fetched {len(df)} records")
    
    if "df" in st.session_state:
        df = st.session_state.df
        
        tab1, tab2, tab3, tab4 = st.tabs(["NCR Report", "Housekeeping", "Safety", "Settings"])
        
        with tab1:
            st.header("📋 NCR Report")
            col1, col2 = st.columns(2)
            with col1:
                report_type = st.radio("Report Type:", ["Open", "Closed"])
            with col2:
                until_date = st.date_input("Until Date:")
            
            if st.button("🔄 Generate NCR Report"):
                result, json_str = generate_ncr_report_for_eligo(df, report_type, until_date=until_date)
                if "error" not in result:
                    st.json(result)
        
        with tab2:
            st.header("🧹 Housekeeping Report")
            hk_type = st.radio("Housekeeping Type:", ["Open", "Closed"], key="hk_type")
            if st.button("🔄 Generate Housekeeping Report"):
                result, json_str = generate_ncr_Housekeeping_report_for_eligo(df, hk_type)
                if "error" not in result:
                    st.json(result)
        
        with tab3:
            st.header("🛡️ Safety Report")
            safety_type = st.radio("Safety Type:", ["Open", "Closed"], key="safety_type")
            if st.button("🔄 Generate Safety Report"):
                result, json_str = generate_ncr_Safety_report_for_eligo(df, safety_type)
                if "error" not in result:
                    st.json(result)
        
        with tab4:
            st.header("⚙️ Settings")
            st.write("Data Summary:")
            st.metric("Total Records", len(df))
            st.metric("Status Breakdown", df['Status'].value_counts().to_dict())


if __name__ == "__main__":
    main()
