
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
    st.error("❌ Missing environment variables. Please set API_KEY, WATSONX_API_URL, MODEL_ID, and PROJECT_ID in your .env file.")
    st.markdown("**Setup Instructions**:\n1. Create a `.env` file with the following:\n```\nAPI_KEY=your_api_key\nWATSONX_API_URL=your_url\nMODEL_ID=your_model_id\nPROJECT_ID=your_project_id\n```\n2. Restart the application.")
    logger.error("Missing one or more required environment variables")
    st.stop()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API Endpoints
LOGIN_URL = "https://dms.asite.com/apilogin/"
SEARCH_URL = "https://adoddleak.asite.com/commonapi/formsearchapi/search"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"


standard_sites = [
    "EWS Tower 1",
    "EWS Tower 2",
    "EWS Tower 3",
    "LIG Tower 1",
    "LIG Tower 2",
    "LIG Tower 3"
]


def project_dropdown():
    project_options = [
        "WAVE CITY CLUB @ PSP 14A",
        "EWS_LIG Veridia PH04",
        "GH-8 Phase-2 (ELIGO) Wave City",
        "GH-8 Phase-3 (EDEN) Wave City",
        "Wave Oakwood, Wave City"
    ]
    project_name = st.sidebar.selectbox(
        "Project Name",
        options=project_options,
        index=project_options.index("Wave Oakwood, Wave City") if "Wave Oakwood, Wave City" in project_options else 0,
        key="project_name_selectbox",
        help="Select a project to fetch data and generate individual reports."
    )
    form_name = st.sidebar.text_input(
        "Form Name",
        "Non Conformity Report",
        key="form_name_input",
        help="Enter the form name for the report."
    )
    return project_name, form_name, project_options

# Function to generate access token
def get_access_token(API_KEY):
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

# Login Function
def login_to_asite(email, password):
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

# Fetch Data Function
def fetch_project_data(session_id, project_name, form_name, record_limit=1000):
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded", "Cookie": f"ASessionID={session_id}"}
    all_data = []
    start_record = 1
    total_records = None

    # Capture start time
    start_time = datetime.now()
    st.write(f"🔄 Fetching data from Asite started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Started fetching data from Asite for project '{project_name}', form '{form_name}' at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    with st.spinner("Fetching data from Asite..."):
        while True:
            search_criteria = {"criteria": [{"field": "ProjectName", "operator": 1, "values": [project_name]}, {"field": "FormName", "operator": 1, "values": [form_name]}], "recordStart": start_record, "recordLimit": record_limit}
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
                logger.error(f"Error fetching data: {str(e)}")
                st.error(f"❌ Error fetching data: {str(e)}")
                break


    end_time = datetime.now()
    st.write(f"🔄 Fetching data from Asite completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {(end_time - start_time).total_seconds()} seconds)")
    logger.info(f"Finished fetching data from Asite for project '{project_name}', form '{form_name}' at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {(end_time - start_time).total_seconds()} seconds)")

    return {"responseHeader": {"results": len(all_data), "total_results": total_records}}, all_data, encoded_payload

# Process JSON Data
def process_json_data(json_data):
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
    logger.debug(f"DataFrame columns after processing: {df.columns.tolist()}")
    if df.empty:
        logger.warning("DataFrame is empty after processing")
        st.warning("⚠️ No data processed. Check the API response.")
    return df

# Clean and Parse JSON
def clean_and_parse_json(text):
    
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
            logger.error(f"Failed to parse JSON even after extraction: {json_str}")
            
    logger.error(f"Could not extract valid JSON from: {text}")
    return None
def assign_site(description: str, standard_sites: list) -> list:
    """
    Assign tower sites based on description keywords.
    Fixed to avoid cross-matching (EWS Tower 1 won't match EWS Tower 2, etc.)
    """
    if not isinstance(description, str) or not description.strip():
        return ["Common Area"]

    # Normalize description: lowercase, replace hyphens/commas with spaces
    desc = re.sub(r'[-,]', ' ', description.lower()).strip()

    # Initialize matched sites
    matched = set()

    # PRIORITY 1: Check for exact shorthand patterns first (most specific)
    shorthand_patterns = {
        r'\bews\s*1\b': "EWS Tower 1",
        r'\bews\s*2\b': "EWS Tower 2",
        r'\bews\s*3\b': "EWS Tower 3",
        r'\bligh?\s*1\b': "LIG Tower 1",
        r'\bligh?\s*2\b': "LIG Tower 2",
        r'\bligh?\s*3\b': "LIG Tower 3",
    }
    
    for pattern, site_name in shorthand_patterns.items():
        if re.search(pattern, desc) and site_name in standard_sites:
            matched.add(site_name)

    # PRIORITY 2: If no shorthand match, check for full tower names
    if not matched:
        ews_pattern = r'\bews\s+tower\s+([1-3])\b'
        ews_matches = re.findall(ews_pattern, desc)
        for tower_num in ews_matches:
            site_name = f"EWS Tower {tower_num}"
            if site_name in standard_sites:
                matched.add(site_name)

        lig_pattern = r'\bligh?\s+tower\s+([1-3])\b'
        lig_matches = re.findall(lig_pattern, desc)
        for tower_num in lig_matches:
            site_name = f"LIG Tower {tower_num}"
            if site_name in standard_sites:
                matched.add(site_name)

    # PRIORITY 3: Check for exact site names from standard_sites list
    if not matched:
        for site in standard_sites:
            if site == "Common Area":
                continue  # Skip Common Area in pattern matching
            site_lower = site.lower()
            site_pattern = re.escape(site_lower).replace(' ', r'\s+').replace('lig', 'ligh?')
            if re.search(rf'\b{site_pattern}\b', desc):
                matched.add(site)

    # Return matched sites or default to Common Area
    return sorted(matched) if matched else ["Common Area"]

@st.cache_data
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type((requests.RequestException, ValueError, KeyError)))
def generate_ncr_report_for_ews(df: pd.DataFrame, report_type: str, start_date=None, end_date=None, Until_Date=None) -> Tuple[Dict[str, Any], str]:
    try:
        with st.spinner(f"Generating {report_type} NCR Report..."):
            # Input validation
            if df is None or df.empty:
                error_msg = "❌ Input DataFrame is empty or None"
                st.error(error_msg)
                return {"error": "Empty DataFrame"}, ""
            
            if report_type not in ["Open", "Closed"]:
                error_msg = f"❌ Invalid report_type: {report_type}. Must be 'Open' or 'Closed'"
                st.error(error_msg)
                return {"error": "Invalid report_type"}, ""
            
            # Ensure the DataFrame has no NaT values in critical columns for filtering
            df = df.copy()
            
            # Check if required columns exist
            required_columns = ['Created Date (WET)', 'Status']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"❌ Missing required columns: {missing_columns}"
                st.error(error_msg)
                return {"error": f"Missing columns: {missing_columns}"}, ""
            
            # Clean the DataFrame
            df = df[df['Created Date (WET)'].notna()]
            
            if report_type == "Closed":
                try:
                    # Ensure Expected Close Date column exists for Closed reports
                    if 'Expected Close Date (WET)' not in df.columns:
                        error_msg = "❌ 'Expected Close Date (WET)' column is required for Closed reports"
                        st.error(error_msg)
                        return {"error": "Missing Expected Close Date column"}, ""
                    
                    start_date = pd.to_datetime(start_date) if start_date else df['Created Date (WET)'].min()
                    end_date = pd.to_datetime(end_date) if end_date else df['Expected Close Date (WET)'].max()
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid date range: {str(e)}")
                    st.error(f"❌ Invalid date range: {str(e)}")
                    return {"error": "Invalid date range"}, ""

                df = df[df['Expected Close Date (WET)'].notna()]
                
                # Ensure 'Days' column exists or calculate it
                if 'Days' not in df.columns:
                    try:
                        df['Days'] = (pd.to_datetime(df['Expected Close Date (WET)']) - pd.to_datetime(df['Created Date (WET)'])).dt.days
                    except Exception as e:
                        logger.error(f"Error calculating Days column: {str(e)}")
                        st.error(f"❌ Error calculating Days: {str(e)}")
                        return {"error": "Error calculating Days"}, ""
                
                filtered_df = df[
                    (df['Status'] == 'Closed') &
                    (pd.to_datetime(df['Created Date (WET)']) >= start_date) &
                    (pd.to_datetime(df['Created Date (WET)']) <= end_date) &
                    (pd.to_numeric(df['Days'], errors='coerce') > 21)
                ].copy()
                
            else:  # Open report
                if Until_Date is None:
                    logger.error("Open Until Date is required for Open NCR Report")
                    st.error("❌ Open Until Date is required for Open NCR Report")
                    return {"error": "Open Until Date is required"}, ""
                
                try:
                    today = pd.to_datetime(Until_Date)
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid Open Until Date: {str(e)}")
                    st.error(f"❌ Invalid Open Until Date: {str(e)}")
                    return {"error": "Invalid Open Until Date"}, ""
                    
                filtered_df = df[
                    (df['Status'] == 'Open') &
                    (df['Created Date (WET)'].notna())
                ].copy()
                
                try:
                    filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
                    filtered_df = filtered_df[filtered_df['Days_From_Today'] > 21].copy()
                except Exception as e:
                    logger.error(f"Error calculating Days_From_Today: {str(e)}")
                    st.error(f"❌ Error calculating Days_From_Today: {str(e)}")
                    return {"error": "Error calculating Days_From_Today"}, ""

            if filtered_df.empty:
                st.warning(f"No {report_type} NCRs found with duration > 21 days. Try adjusting the date range or criteria.")
                return {"error": f"No {report_type} records found with duration > 21 days"}, ""

            # Safe string conversion
            try:
                filtered_df.loc[:, 'Created Date (WET)'] = filtered_df['Created Date (WET)'].astype(str)
                if 'Expected Close Date (WET)' in filtered_df.columns:
                    filtered_df.loc[:, 'Expected Close Date (WET)'] = filtered_df['Expected Close Date (WET)'].astype(str)
            except Exception as e:
                logger.error(f"Error converting dates to string: {str(e)}")
                st.error(f"❌ Error converting dates: {str(e)}")
                return {"error": "Error converting dates"}, ""

            processed_data = filtered_df.to_dict(orient="records")
            
            cleaned_data = []
            unique_records = []

            # Define standard_sites before using assign_site
            standard_sites = [
                "EWS Tower 1",
                "EWS Tower 2",
                "EWS Tower 3",
                "LIG Tower 1",
                "LIG Tower 2",
                "LIG Tower 3"
            ]
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

                    description = cleaned_record["Description"].lower()

                    if not description:
                        continue 

                    # Initialize Discipline_Category
                    discipline = cleaned_record["Discipline"].strip().lower()
                    if discipline == "none" or not discipline:
                        logger.debug(f"Skipping record with invalid discipline: {discipline}")
                        continue
                    elif "hse" in discipline:
                        logger.debug(f"Skipping HSE record: {discipline}")
                        cleaned_record["Discipline_Category"] = "HSE"
                        continue  # Skip HSE records entirely
                    elif "structure" in discipline or "sw" in discipline:
                        cleaned_record["Discipline_Category"] = "SW"
                    elif "civil" in discipline or "finishing" in discipline or "fw" in discipline:
                        cleaned_record["Discipline_Category"] = "FW"
                    else:
                        cleaned_record["Discipline_Category"] = "MEP"

                        
                    unique_records.append(cleaned_record["Description"])

                    # Tower assignment using assign_site
                    matched_towers = assign_site(cleaned_record["Description"], standard_sites)
                    for tower in matched_towers:
                        tower_record = cleaned_record.copy()
                        tower_record["Tower"] = tower
                        cleaned_data.append(tower_record)
                        logger.debug(f"Assigned {tower}: {cleaned_record['Description']}")
                            
                except Exception as e:
                    logger.error(f"Error processing record: {record}, error: {str(e)}")
                    continue
            
            # Remove duplicates
            try:
                cleaned_data = [dict(t) for t in {tuple(sorted(d.items())) for d in cleaned_data}]
            except Exception as e:
                logger.error(f"Error removing duplicates: {str(e)}")
                # If deduplication fails, continue with original data
                pass

            if not cleaned_data:
                return {report_type: {"Sites": {}, "Grand_Total": 0}}, ""

            # Get access token
            try:
                access_token = get_access_token(API_KEY)
                if not access_token:
                    return {"error": "Failed to obtain access token"}, ""
            except Exception as e:
                logger.error(f"Error getting access token: {str(e)}")
                return {"error": f"Failed to obtain access token: {str(e)}"}, ""

            all_results = {report_type: {"Sites": {}, "Grand_Total": 0}}
            chunk_size = int(os.getenv("CHUNK_SIZE", 20))
            
            for i in range(0, len(cleaned_data), chunk_size):
                chunk = cleaned_data[i:i + chunk_size]
                
                # Capture and log start time
                start_time = datetime.now()
                st.write(f" 🔄 Processing chunk {i // chunk_size + 1}: Records {i} to {min(i + chunk_size, len(cleaned_data))} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Started chunk {i // chunk_size + 1} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Log data sent to WatsonX
                logger.info(f"Data sent to WatsonX for {report_type} chunk {i // chunk_size + 1}: {json.dumps(chunk, indent=2)}")

                # Log the total number of records being processed
                total_records = len(cleaned_data)
                st.write(f"Total {report_type} records to process: {total_records}")
                logger.info(f"Total {report_type} records to process: {total_records}")

                prompt = (
                    "IMPORTANT: RETURN ONLY A SINGLE VALID JSON OBJECT WITH THE EXACT FIELDS SPECIFIED BELOW. "
                    "DO NOT GENERATE ANY CODE (e.g., Python, JavaScript). "
                    "DO NOT INCLUDE ANY TEXT, EXPLANATIONS, OR MULTIPLE RESPONSES OUTSIDE THE JSON OBJECT. "
                    "DO NOT WRAP THE JSON IN CODE BLOCKS (e.g., ```json). "
                    "RETURN THE JSON OBJECT DIRECTLY.\n\n"
                    f"Task: Group the provided data by 'Tower' and collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', and 'Discipline' into arrays. "
                    f"Count the records by 'Discipline_Category' ('SW', 'FW', 'MEP'), calculate the 'Total' for each 'Tower'. "
                    f"Process ALL {len(chunk)} records provided in the data.\n"
                    f"Use 'Tower' values (e.g., 'EWS Tower 1', 'EWS Tower 2', 'EWS Tower 3', 'LIG Tower 1', 'LIG Tower 2', 'LIG Tower 3', 'Common_Area'), "
                    f"'Discipline_Category' values (e.g., 'SW', 'FW', 'MEP'). Count each record exactly once.\n\n"
                    "REQUIRED OUTPUT FORMAT (ONLY THESE FIELDS):\n"
                    "{\n"
                    f'  "{report_type}": {{\n'
                    '    "Sites": {\n'
                    '      "Site_Name1": {\n'
                    '        "Descriptions": ["description1", "description2"],\n'
                    '        "Created Date (WET)": ["date1", "date2"],\n'
                    '        "Expected Close Date (WET)": ["date1", "date2"],\n'
                    '        "Status": ["status1", "status2"],\n'
                    '        "Discipline": ["discipline1", "discipline2"],\n'
                    '        "SW": number,\n'
                    '        "FW": number,\n'
                    '        "MEP": number,\n'
                    '        "Total": number\n'
                    '      }\n'
                    '    },\n'
                    f'    "Grand_Total": {len(chunk)}\n'
                    '  }\n'
                    '}\n\n'
                    f"Data: {json.dumps(chunk)}\n"
                    f"IMPORTANT: Ensure the JSON is valid and contains all required fields. "    
                    f"Return the result strictly as a JSON object—no code, no explanations, only the JSON.Dont put <|eom_id|> or any other markers in the JSON output. Grand_Total must be {len(chunk)}."
                )

                payload = {
                    "input": prompt,
                    "parameters": {
                        "decoding_method": "greedy",
                        "max_new_tokens": 8100,
                        "min_new_tokens": 0,
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

                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["POST"]
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                http = requests.Session()
                http.mount("https://", adapter)

                try:
                    start_time = datetime.now()
                    response = http.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=600)
                    logger.info(f"WatsonX API call took {(datetime.now() - start_time).total_seconds()} seconds for {len(chunk)} records")
                    st.write(f"Debug - Response status code: {response.status_code}")

                    if response.status_code == 200:
                        api_result = response.json()
                        generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                        short_text = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
                        logger.debug(f"Parsed generated text: {generated_text}")

                        parsed_json = clean_and_parse_json(generated_text)
                        if parsed_json and report_type in parsed_json:
                            chunk_result = parsed_json[report_type]
                            st.write(f"Processing API response for chunk {i // chunk_size + 1} with {len(chunk)} records")
                            
                            for site, data in chunk_result["Sites"].items():
                                if site not in all_results[report_type]["Sites"]:
                                    all_results[report_type]["Sites"][site] = {
                                        "Descriptions": [],
                                        "Created Date (WET)": [],
                                        "Expected Close Date (WET)": [],
                                        "Status": [],
                                        "Discipline": [],
                                        "SW": 0,
                                        "FW": 0,
                                        "MEP": 0,
                                        "Total": 0
                                    }
                                all_results[report_type]["Sites"][site]["Descriptions"].extend(data["Descriptions"])
                                all_results[report_type]["Sites"][site]["Created Date (WET)"].extend(data["Created Date (WET)"])
                                all_results[report_type]["Sites"][site]["Expected Close Date (WET)"].extend(data["Expected Close Date (WET)"])
                                all_results[report_type]["Sites"][site]["Status"].extend(data["Status"])
                                all_results[report_type]["Sites"][site]["Discipline"].extend(data["Discipline"])
                                all_results[report_type]["Sites"][site]["SW"] += data["SW"]
                                all_results[report_type]["Sites"][site]["FW"] += data["FW"]
                                all_results[report_type]["Sites"][site]["MEP"] += data["MEP"]
                                all_results[report_type]["Sites"][site]["Total"] += data["Total"]
                            
                            # Use the actual number of records processed instead of API count
                            all_results[report_type]["Grand_Total"] += len(chunk)
                            st.write(f"Successfully processed chunk {i // chunk_size + 1} with {len(chunk)} records")
                        else:
                            logger.error("No valid JSON found in response")
                            st.error("❌ No valid JSON found in response")
                            st.write("Falling back to local count for this chunk")
                            local_result = clean_and_parse_json(chunk, report_type)
                            if local_result and report_type in local_result:
                                # Merge local_result into all_results
                                chunk_result = local_result[report_type]
                                for site, data in chunk_result["Sites"].items():
                                    if site not in all_results[report_type]["Sites"]:
                                        all_results[report_type]["Sites"][site] = {
                                            "Descriptions": [],
                                            "Created Date (WET)": [],
                                            "Expected Close Date (WET)": [],
                                            "Status": [],
                                            "Discipline": [],
                                            "SW": 0,
                                            "FW": 0,
                                            "MEP": 0,
                                            "Total": 0
                                        }
                                    all_results[report_type]["Sites"][site]["Descriptions"].extend(data["Descriptions"])
                                    all_results[report_type]["Sites"][site]["Created Date (WET)"].extend(data["Created Date (WET)"])
                                    all_results[report_type]["Sites"][site]["Expected Close Date (WET)"].extend(data["Expected Close Date (WET)"])
                                    all_results[report_type]["Sites"][site]["Status"].extend(data["Status"])
                                    all_results[report_type]["Sites"][site]["Discipline"].extend(data["Discipline"])
                                    all_results[report_type]["Sites"][site]["SW"] += data["SW"]
                                    all_results[report_type]["Sites"][site]["FW"] += data["FW"]
                                    all_results[report_type]["Sites"][site]["MEP"] += data["MEP"]
                                    all_results[report_type]["Sites"][site]["Total"] += data["Total"]
                                all_results[report_type]["Grand_Total"] += len(chunk)
                    else:
                        error_msg = f"❌ WatsonX API error: {response.status_code} - {response.text}"
                        st.error(error_msg)
                        logger.error(error_msg)
                        st.write("Falling back to local count for this chunk")
                        local_result = clean_and_parse_json(chunk, report_type)
                        if local_result and report_type in local_result:
                            chunk_result = local_result[report_type]
                            for site, data in chunk_result["Sites"].items():
                                if site not in all_results[report_type]["Sites"]:
                                    all_results[report_type]["Sites"][site] = {
                                        "Descriptions": [],
                                        "Created Date (WET)": [],
                                        "Expected Close Date (WET)": [],
                                        "Status": [],
                                        "Discipline": [],
                                        "SW": 0,
                                        "FW": 0,
                                        "MEP": 0,
                                        "Total": 0
                                    }
                                all_results[report_type]["Sites"][site]["Descriptions"].extend(data["Descriptions"])
                                all_results[report_type]["Sites"][site]["Created Date (WET)"].extend(data["Created Date (WET)"])
                                all_results[report_type]["Sites"][site]["Expected Close Date (WET)"].extend(data["Expected Close Date (WET)"])
                                all_results[report_type]["Sites"][site]["Status"].extend(data["Status"])
                                all_results[report_type]["Sites"][site]["Discipline"].extend(data["Discipline"])
                                all_results[report_type]["Sites"][site]["SW"] += data["SW"]
                                all_results[report_type]["Sites"][site]["FW"] += data["FW"]
                                all_results[report_type]["Sites"][site]["MEP"] += data["MEP"]
                                all_results[report_type]["Sites"][site]["Total"] += data["Total"]
                            all_results[report_type]["Grand_Total"] += len(chunk)
                        
                except requests.RequestException as e:
                    error_msg = f"❌ Request exception during WatsonX call: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.write("Falling back to local count for this chunk")
                    local_result = clean_and_parse_json(chunk, report_type)
                    if local_result and report_type in local_result:
                        chunk_result = local_result[report_type]
                        for site, data in chunk_result["Sites"].items():
                            if site not in all_results[report_type]["Sites"]:
                                all_results[report_type]["Sites"][site] = {
                                    "Descriptions": [],
                                    "Created Date (WET)": [],
                                    "Expected Close Date (WET)": [],
                                    "Status": [],
                                    "Discipline": [],
                                    "SW": 0,
                                    "FW": 0,
                                    "MEP": 0,
                                    "Total": 0
                                }
                            all_results[report_type]["Sites"][site]["Descriptions"].extend(data["Descriptions"])
                            all_results[report_type]["Sites"][site]["Created Date (WET)"].extend(data["Created Date (WET)"])
                            all_results[report_type]["Sites"][site]["Expected Close Date (WET)"].extend(data["Expected Close Date (WET)"])
                            all_results[report_type]["Sites"][site]["Status"].extend(data["Status"])
                            all_results[report_type]["Sites"][site]["Discipline"].extend(data["Discipline"])
                            all_results[report_type]["Sites"][site]["SW"] += data["SW"]
                            all_results[report_type]["Sites"][site]["FW"] += data["FW"]
                            all_results[report_type]["Sites"][site]["MEP"] += data["MEP"]
                            all_results[report_type]["Sites"][site]["Total"] += data["Total"]
                        all_results[report_type]["Grand_Total"] += len(chunk)
                except Exception as e:
                    error_msg = f"❌ Exception during WatsonX call: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.write("Falling back to local count for this chunk")
                    local_result = clean_and_parse_json(chunk, report_type)
                    if local_result and report_type in local_result:
                        chunk_result = local_result[report_type]
                        for site, data in chunk_result["Sites"].items():
                            if site not in all_results[report_type]["Sites"]:
                                all_results[report_type]["Sites"][site] = {
                                    "Descriptions": [],
                                    "Created Date (WET)": [],
                                    "Expected Close Date (WET)": [],
                                    "Status": [],
                                    "Discipline": [],
                                    "SW": 0,
                                    "FW": 0,
                                    "MEP": 0,
                                    "Total": 0
                                }
                            all_results[report_type]["Sites"][site]["Descriptions"].extend(data["Descriptions"])
                            all_results[report_type]["Sites"][site]["Created Date (WET)"].extend(data["Created Date (WET)"])
                            all_results[report_type]["Sites"][site]["Expected Close Date (WET)"].extend(data["Expected Close Date (WET)"])
                            all_results[report_type]["Sites"][site]["Status"].extend(data["Status"])
                            all_results[report_type]["Sites"][site]["Discipline"].extend(data["Discipline"])
                            all_results[report_type]["Sites"][site]["SW"] += data["SW"]
                            all_results[report_type]["Sites"][site]["FW"] += data["FW"]
                            all_results[report_type]["Sites"][site]["MEP"] += data["MEP"]
                            all_results[report_type]["Sites"][site]["Total"] += data["Total"]
                        all_results[report_type]["Grand_Total"] += len(chunk)

            # Capture and log end time after API call
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            duration_minutes = duration_seconds / 60

            st.write(
                f"🔄 Chunk {i // chunk_size + 1} model processing for {report_type} completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(Duration: {duration_seconds:.2f} seconds / {duration_minutes:.2f} minutes)"
            )

            logger.info(
                f"Finished model processing chunk {i // chunk_size + 1} for {report_type} at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(Duration: {duration_seconds:.2f} seconds / {duration_minutes:.2f} minutes)"
            )

            # Convert all_results to a DataFrame for table display
            table_data = []
            for site, data in all_results[report_type]["Sites"].items():
                row = {
                    "Site": site,
                    "SW Count": data["SW"],
                    "FW Count": data["FW"],
                    "MEP Count": data["MEP"],
                    "Total Records": data["Total"],
                    "Descriptions": "; ".join(data["Descriptions"]),
                    "Created Dates": "; ".join(data["Created Date (WET)"]),
                    "Expected Close Dates": "; ".join(data["Expected Close Date (WET)"]),
                    "Statuses": "; ".join(data["Status"]),
                    "Disciplines": "; ".join(data["Discipline"]),
                }
            table_data.append(row)
            
            if table_data:
                df_table = pd.DataFrame(table_data)
                st.write(f"Final {report_type} Results:")
                st.dataframe(df_table, use_container_width=True)
            else:
                st.write(f"No data available for {report_type} report.")

            return all_results, json.dumps(all_results)

    except TypeError as e:
        logger.error(f"TypeError in generate_ncr_report: {str(e)}")
        st.error(f"❌ Type Error: {str(e)}")
        return {"error": f"Type Error: {str(e)}"}, ""
    except Exception as e:
        logger.error(f"Unexpected error in generate_ncr_report: {str(e)}")
        st.error(f"❌ Unexpected Error: {str(e)}")
        return {"error": f"Unexpected Error: {str(e)}"}, ""
    
# Generate NCR Housekeeping Report
@st.cache_data
def generate_ncr_Housekeeping_report_for_ews(df, report_type, start_date=None, end_date=None, open_until_date=None):
    """Generate Housekeeping NCR report for Open or Closed records."""
    with st.spinner(f"Generating {report_type} Housekeeping NCR Report with WatsonX..."):
        today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
        closed_start = pd.to_datetime(start_date) if start_date else None
        closed_end = pd.to_datetime(end_date) if end_date else None
        open_until = pd.to_datetime(open_until_date)

        # Housekeeping and safety keywords
        housekeeping_keywords = [
            'housekeeping', 'cleaning', 'cleanliness', 'waste disposal', 'waste management', 'garbage', 'trash',
            'rubbish', 'debris', 'litter', 'dust', 'untidy', 'cluttered', 'accumulation of waste', 'construction waste',
            'pile of garbage', 'poor housekeeping', 'material storage', 'construction debris', 'cleaning schedule',
            'garbage collection', 'waste bins', 'dirty', 'mess', 'unclean', 'disorderly', 'dirty floor',
            'waste disposal area', 'waste collection', 'cleaning protocol', 'sanitation', 'trash removal',
            'waste accumulation', 'unkept area', 'refuse collection', 'workplace cleanliness'
        ]
        safety_keywords = ['safety precautions', 'PPE', 'fall protection', 'safety belts', 'barricades']

        def is_housekeeping_record(description):
            description_lower = description.lower()
            has_housekeeping = any(keyword in description_lower for keyword in housekeeping_keywords)
            has_safety = any(keyword in description_lower for keyword in safety_keywords)
            return has_housekeeping and not has_safety

        # Filter data
        if report_type == "Closed":
            filtered_df = df[
                (df['Discipline'] == 'HSE') &
                (df['Status'] == 'Closed') &
                (df['Days'].notnull()) &
                (df['Days'] > 7) &
                (df['Description'].apply(is_housekeeping_record))
            ].copy()
            if closed_start and closed_end:
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['Created Date (WET)']) >= closed_start) &
                    (pd.to_datetime(filtered_df['Expected Close Date (WET)']) <= closed_end)
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
            if open_until:
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['Created Date (WET)']) <= open_until)
                ].copy()

        if filtered_df.empty:
            return {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}, ""

        filtered_df.loc[:, 'Created Date (WET)'] = filtered_df['Created Date (WET)'].astype(str)
        filtered_df.loc[:, 'Expected Close Date (WET)'] = filtered_df['Expected Close Date (WET)'].astype(str)

        processed_data = filtered_df.to_dict(orient="records")
        
        cleaned_data = []
        seen_descriptions = set()
        for record in processed_data:
            description = str(record.get("Description", "")).strip()
            if description and description not in seen_descriptions:
                seen_descriptions.add(description)
                cleaned_record = {
                    "Description": description,
                    "Created Date (WET)": str(record.get("Created Date (WET)", "")),
                    "Expected Close Date (WET)": str(record.get("Expected Close Date (WET)", "")),
                    "Status": str(record.get("Status", "")),
                    "Days": record.get("Days", 0),
                    "Discipline": "HSE"
                }
                if report_type == "Open":
                    cleaned_record["Days_From_Today"] = record.get("Days_From_Today", 0)
                matched_towers = assign_site(description, standard_sites)
                for tower in matched_towers:
                    tower_record = cleaned_record.copy()
                    tower_record["Tower"] = tower
                    cleaned_data.append(tower_record)
                    logger.debug(f"Assigned {tower}: {description}")

        st.write(f"Total {report_type} records to process: {len(cleaned_data)}")
        logger.debug(f"Processed data: {json.dumps(cleaned_data, indent=2)}")

        if not cleaned_data:
            return {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}, ""

        access_token = get_access_token(API_KEY)
        if not access_token:
            return {"error": "Failed to obtain access token"}, ""

        result = {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}
        chunk_size = 10
        total_chunks = (len(cleaned_data) + chunk_size - 1) // chunk_size

        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504, 429, 408],
            allowed_methods=["POST"],
            raise_on_redirect=True,
            raise_on_status=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)

        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        error_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)

        for i in range(0, len(cleaned_data), chunk_size):
            chunk = cleaned_data[i:i + chunk_size]
            current_chunk = i // chunk_size + 1
            progress = min((current_chunk / total_chunks) * 100, 100)
            progress_bar.progress(int(progress))
            status_placeholder.write(f"Processed {current_chunk}/{total_chunks} chunks ({int(progress)}%)")
            logger.debug(f"Chunk data: {json.dumps(chunk, indent=2)}")

            prompt = (
                "Return a single valid JSON object with the exact fields specified below. Do not generate code, explanations, multiple responses, or wrap the JSON in code blocks. Process the provided data and return only the JSON object.\n\n"
                "Task: Count Housekeeping NCRs by site ('Tower' field) where 'Discipline' is 'HSE' and 'Days' > 7. The 'Description' must contain any of these housekeeping keywords (case-insensitive): 'housekeeping', 'cleaning', 'cleanliness', 'waste disposal', 'waste management', 'garbage', 'trash', 'rubbish', 'debris', 'litter', 'dust', 'untidy', 'cluttered', 'accumulation of waste', 'construction waste', 'pile of garbage', 'poor housekeeping', 'material storage', 'construction debris', 'cleaning schedule', 'garbage collection', 'waste bins', 'dirty', 'mess', 'unclean', 'disorderly', 'dirty floor', 'waste disposal area', 'waste collection', 'cleaning protocol', 'sanitation', 'trash removal', 'waste accumulation', 'unkept area', 'refuse collection', 'workplace cleanliness'. Exclude records primarily about safety issues (e.g., 'safety precautions', 'PPE', 'fall protection', 'safety belts', 'barricades'). Use 'Tower' values as they appear (e.g., 'Common Area'). Collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', and 'Status' into arrays for each site. Assign the count to 'Count' (No. of Housekeeping NCRs beyond 7 days). If no matches, set count to 0 for each site in the data. Return all sites present in the data.\n\n"
                "Output Format:\n"
                "{\n"
                '  "Housekeeping": {\n'
                '    "Sites": {\n'
                '      "Site_Name": {\n'
                '        "Descriptions": [],\n'
                '        "Created Date (WET)": [],\n'
                '        "Expected Close Date (WET)": [],\n'
                '        "Status": [],\n'
                '        "Count": 0\n'
                '      }\n'
                '    },\n'
                '    "Grand_Total": 0\n'
                '  }\n'
                '}\n\n'
                f"Data: {json.dumps(chunk)}\n"
            )

            payload = {
                "input": prompt,
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 500,
                    "min_new_tokens": 0,
                    "temperature": 0.001,
                    "n": 1
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
                logger.debug("Initiating WatsonX API call...")
                response = session.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=30)
                logger.info(f"WatsonX API response status: {response.status_code}")

                if response.status_code == 200:
                    api_result = response.json()
                    generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                    logger.debug(f"Generated text for chunk {current_chunk}: {generated_text}")

                    json_str = None
                    brace_count = 0
                    start_idx = None
                    for idx, char in enumerate(generated_text):
                        if char == '{':
                            if brace_count == 0:
                                start_idx = idx
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and start_idx is not None:
                                json_str = generated_text[start_idx:idx + 1]
                                break

                    if json_str:
                        try:
                            parsed_json = json.loads(json_str)
                            chunk_result = parsed_json.get("Housekeeping", {})
                            chunk_sites = chunk_result.get("Sites", {})
                            chunk_grand_total = chunk_result.get("Grand_Total", 0)

                            for site, values in chunk_sites.items():
                                if not isinstance(values, dict):
                                    logger.warning(f"Invalid site data for {site}: {values}")
                                    continue
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
                            result["Housekeeping"]["Grand_Total"] += chunk_grand_total
                            logger.debug(f"Successfully processed chunk {current_chunk}/{total_chunks}")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSONDecodeError for chunk {current_chunk}: {str(e)}")
                            error_placeholder.error(f"Failed to parse JSON for chunk {current_chunk}: {str(e)}")
                            for record in chunk:
                                if is_housekeeping_record(record["Description"]) and record.get("Days", 0) > 7 and record.get("Discipline") == "HSE":
                                    site = record["Tower"]
                                    if site not in result["Housekeeping"]["Sites"]:
                                        result["Housekeeping"]["Sites"][site] = {
                                            "Count": 0,
                                            "Descriptions": [],
                                            "Created Date (WET)": [],
                                            "Expected Close Date (WET)": [],
                                            "Status": []
                                        }
                                    result["Housekeeping"]["Sites"][site]["Descriptions"].append(record["Description"])
                                    result["Housekeeping"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                                    result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                                    result["Housekeeping"]["Sites"][site]["Status"].append(record["Status"])
                                    result["Housekeeping"]["Sites"][site]["Count"] += 1
                                    result["Housekeeping"]["Grand_Total"] += 1
                    else:
                        logger.error(f"No valid JSON for chunk {current_chunk}: {generated_text}")
                        error_placeholder.error(f"No valid JSON for chunk {current_chunk}")
                        for record in chunk:
                            if is_housekeeping_record(record["Description"]) and record.get("Days", 0) > 7 and record.get("Discipline") == "HSE":
                                site = record["Tower"]
                                if site not in result["Housekeeping"]["Sites"]:
                                    result["Housekeeping"]["Sites"][site] = {
                                        "Count": 0,
                                        "Descriptions": [],
                                        "Created Date (WET)": [],
                                        "Expected Close Date (WET)": [],
                                        "Status": []
                                    }
                                result["Housekeeping"]["Sites"][site]["Descriptions"].append(record["Description"])
                                result["Housekeeping"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                                result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                                result["Housekeeping"]["Sites"][site]["Status"].append(record["Status"])
                                result["Housekeeping"]["Sites"][site]["Count"] += 1
                                result["Housekeeping"]["Grand_Total"] += 1
                else:
                    logger.error(f"WatsonX API error for chunk {current_chunk}: {response.status_code} - {response.text}")
                    error_placeholder.error(f"WatsonX API error for chunk {current_chunk}: {response.status_code}")
                    for record in chunk:
                        if is_housekeeping_record(record["Description"]) and record.get("Days", 0) > 7 and record.get("Discipline") == "HSE":
                            site = record["Tower"]
                            if site not in result["Housekeeping"]["Sites"]:
                                result["Housekeeping"]["Sites"][site] = {
                                    "Count": 0,
                                    "Descriptions": [],
                                    "Created Date (WET)": [],
                                    "Expected Close Date (WET)": [],
                                    "Status": []
                                }
                            result["Housekeeping"]["Sites"][site]["Descriptions"].append(record["Description"])
                            result["Housekeeping"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                            result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                            result["Housekeeping"]["Sites"][site]["Status"].append(record["Status"])
                            result["Housekeeping"]["Sites"][site]["Count"] += 1
                            result["Housekeeping"]["Grand_Total"] += 1
            except requests.exceptions.ReadTimeout as e:
                logger.error(f"ReadTimeoutError for chunk {current_chunk}: {str(e)}")
                error_placeholder.error(f"Failed to connect to WatsonX API for chunk {current_chunk}: {str(e)}")
                for record in chunk:
                    if is_housekeeping_record(record["Description"]) and record.get("Days", 0) > 7 and record.get("Discipline") == "HSE":
                        site = record["Tower"]
                        if site not in result["Housekeeping"]["Sites"]:
                            result["Housekeeping"]["Sites"][site] = {
                                "Count": 0,
                                "Descriptions": [],
                                "Created Date (WET)": [],
                                "Expected Close Date (WET)": [],
                                "Status": []
                            }
                        result["Housekeeping"]["Sites"][site]["Descriptions"].append(record["Description"])
                        result["Housekeeping"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                        result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                        result["Housekeeping"]["Sites"][site]["Status"].append(record["Status"])
                        result["Housekeeping"]["Sites"][site]["Count"] += 1
                        result["Housekeeping"]["Grand_Total"] += 1
            except requests.exceptions.RequestException as e:
                logger.error(f"RequestException for chunk {current_chunk}: {str(e)}")
                error_placeholder.error(f"Failed to connect to WatsonX API for chunk {current_chunk}: {str(e)}")
                for record in chunk:
                    if is_housekeeping_record(record["Description"]) and record.get("Days", 0) > 7 and record.get("Discipline") == "HSE":
                        site = record["Tower"]
                        if site not in result["Housekeeping"]["Sites"]:
                            result["Housekeeping"]["Sites"][site] = {
                                "Count": 0,
                                "Descriptions": [],
                                "Created Date (WET)": [],
                                "Expected Close Date (WET)": [],
                                "Status": []
                            }
                        result["Housekeeping"]["Sites"][site]["Descriptions"].append(record["Description"])
                        result["Housekeeping"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                        result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                        result["Housekeeping"]["Sites"][site]["Status"].append(record["Status"])
                        result["Housekeeping"]["Sites"][site]["Count"] += 1
                        result["Housekeeping"]["Grand_Total"] += 1

        progress_bar.progress(100)
        status_placeholder.write(f"Processed {total_chunks}/{total_chunks} chunks (100%)")
        logger.debug(f"Final result before deduplication: {json.dumps(result, indent=2)}")

        for site in result["Housekeeping"]["Sites"]:
            result["Housekeeping"]["Sites"][site]["Descriptions"] = list(set(result["Housekeeping"]["Sites"][site]["Descriptions"]))
            result["Housekeeping"]["Sites"][site]["Created Date (WET)"] = list(set(result["Housekeeping"]["Sites"][site]["Created Date (WET)"]))
            result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"] = list(set(result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"]))
            result["Housekeeping"]["Sites"][site]["Status"] = list(set(result["Housekeeping"]["Sites"][site]["Status"]))
        
        logger.debug(f"Final result after deduplication: {json.dumps(result, indent=2)}")
        return result, json.dumps(result)
    
def clean_and_parse_json(generated_text):
    # Remove code block markers if present
    cleaned_text = re.sub(r'```json|```python|```', '', generated_text).strip()
    
    # First attempt: Try to parse the text directly as JSON
    try:
        for line in cleaned_text.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                return json.loads(line)
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSONDecodeError: {str(e)} - Cleaned response: {cleaned_text}")
    
    # Second attempt: If the response contains Python code with a print(json.dumps(...)),
    # extract the JSON from the output
    json_match = re.search(r'print\s*\(\s*json\.dumps\((.*?),\s*indent=2\s*\)\s*\)', cleaned_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return eval(json_str)  # Safely evaluate the JSON string as a Python dict
        except Exception as e:
            logger.error(f"Failed to evaluate extracted JSON: {str(e)} - Extracted JSON: {json_str}")
    
    logger.error(f"JSONDecodeError: Unable to parse response - Cleaned response: {cleaned_text}")
    return None

# Define standard_sites globally
standard_sites = [
    "EWS Tower 1",
    "EWS Tower 2",
    "EWS Tower 3",
    "LIG Tower 1",
    "LIG Tower 2",
    "LIG Tower 3"
]

@st.cache_data
def generate_ncr_Safety_report_for_ews(df, report_type, start_date=None, end_date=None, open_until_date=None):
    """Generate Safety NCR report for Open or Closed records."""
    with st.spinner(f"Generating {report_type} Safety NCR Report with WatsonX..."):
        try:
            today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
            closed_start = pd.to_datetime(start_date) if start_date else None
            closed_end = pd.to_datetime(end_date) if end_date else None
            open_until = pd.to_datetime(open_until_date) if open_until_date else None

            # Define safety keywords
            safety_keywords = [
                'safety precautions', 'ppe', 'helmet', 'safety shoes', 'safety shoe', 'jacket', 'gloves', 'safety harness',
                'precaution', 'safety gear', 'fall protection', 'scaffolding safety', 'ladder safety', 'hazard',
                'unsafe condition', 'safety violation', 'safety belt', 'lifeline', 'guard rails', 'electrical hazard',
                'unsafe platform', 'catch net', 'edge protection', 'scaffold', 'lifting equipment', 'temporary electricity',
                'dust suppression', 'debris chute', 'spill control', 'crane operator', 'fire hazard', 'barricading',
                'safety norms', 'lifeline is not fixed', 'third party inspection', 'tpic', 'no barrier', 'lock and key',
                'buzzer', 'gates at landing platforms', 'halogen lamps', 'fall catch net', 'environmental contamination'
            ]

            def is_safety_record(description):
                description_lower = description.lower()
                return any(keyword in description_lower for keyword in safety_keywords)

            # Filter data
            if report_type == "Closed":
                filtered_df = df[
                    (df['Discipline'] == 'HSE') &
                    (df['Status'] == 'Closed') &
                    (df['Days'].notnull()) &
                    (df['Days'] > 7) &
                    (df['Description'].apply(is_safety_record))
                ].copy()
                if closed_start and closed_end:
                    filtered_df = filtered_df[
                        (pd.to_datetime(filtered_df['Created Date (WET)']) >= closed_start) &
                        (pd.to_datetime(filtered_df['Expected Close Date (WET)']) <= closed_end)
                    ].copy()
            else:  # Open
                filtered_df = df[
                    (
                        (df['Discipline'] == 'HSE') |
                        (df['Description'].apply(is_safety_record))
                    ) &
                    (df['Status'] == 'Open') &
                    (pd.to_datetime(df['Created Date (WET)']).notna())
                ].copy()
                filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
                filtered_df = filtered_df[filtered_df['Days_From_Today'] > 7].copy()
                if open_until:
                    filtered_df = filtered_df[
                        (pd.to_datetime(filtered_df['Created Date (WET)']) <= open_until)
                    ].copy()

            if filtered_df.empty:
                return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

            filtered_df.loc[:, 'Created Date (WET)'] = pd.to_datetime(filtered_df['Created Date (WET)']).dt.strftime('%Y-%m-%d')
            filtered_df.loc[:, 'Expected Close Date (WET)'] = pd.to_datetime(filtered_df['Expected Close Date (WET)']).dt.strftime('%Y-%m-%d')

            processed_data = filtered_df.to_dict(orient="records")
            
            cleaned_data = []
            seen_descriptions = set()
            for record in processed_data:
                description = str(record.get("Description", "")).strip()
                if description and description not in seen_descriptions:
                    seen_descriptions.add(description)
                    cleaned_record = {
                        "Description": description,
                        "Created Date (WET)": str(record.get("Created Date (WET)", "")),
                        "Expected Close Date (WET)": str(record.get("Expected Close Date (WET)", "")),
                        "Status": str(record.get("Status", "")),
                        "Days": record.get("Days", 0),
                        "Discipline": "HSE"
                    }
                    if report_type == "Open":
                        cleaned_record["Days_From_Today"] = record.get("Days_From_Today", 0)
                    matched_towers = assign_site(description, standard_sites)
                    for tower in matched_towers:
                        tower_record = cleaned_record.copy()
                        tower_record["Tower"] = tower
                        cleaned_data.append(tower_record)
                        logger.debug(f"Assigned {tower}: {description}")

            st.write(f"Total {report_type} records to process: {len(cleaned_data)}")
            logger.debug(f"Processed data: {json.dumps(cleaned_data, indent=2)}")

            if not cleaned_data:
                return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

            access_token = get_access_token(API_KEY)
            if not access_token:
                return {"error": "Failed to obtain access token"}, ""

            result = {"Safety": {"Sites": {}, "Grand_Total": 0}}
            chunk_size = 10
            total_chunks = (len(cleaned_data) + chunk_size - 1) // chunk_size

            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=2,
                status_forcelist=[500, 502, 503, 504, 429, 408],
                allowed_methods=["POST"],
                raise_on_redirect=True,
                raise_on_status=True
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)

            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            error_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)

            for i in range(0, len(cleaned_data), chunk_size):
                chunk = cleaned_data[i:i + chunk_size]
                current_chunk = i // chunk_size + 1
                progress = min((current_chunk / total_chunks) * 100, 100)
                progress_bar.progress(int(progress))
                status_placeholder.write(f"Processed {current_chunk}/{total_chunks} chunks ({int(progress)}%)")
                logger.debug(f"Chunk data: {json.dumps(chunk, indent=2)}")

                prompt = (
                    "Return a single valid JSON object with the exact fields specified below. Do not generate code, explanations, multiple responses, or wrap the JSON in code blocks. Process the provided data and return only the JSON object.\n\n"
                    "Task: Count Safety NCRs by site ('Tower' field) where 'Discipline' is 'HSE' and 'Days' > 7 or 'Days_From_Today' > 7 for open records. The 'Description' must contain any of these safety keywords (case-insensitive): 'safety precautions', 'ppe', 'helmet', 'safety shoes', 'safety shoe', 'jacket', 'gloves', 'safety harness', 'precaution', 'safety gear', 'fall protection', 'scaffolding safety', 'ladder safety', 'hazard', 'unsafe condition', 'safety violation', 'safety belt', 'lifeline', 'guard rails', 'electrical hazard', 'unsafe platform', 'catch net', 'edge protection', 'scaffold', 'lifting equipment', 'temporary electricity', 'dust suppression', 'debris chute', 'spill control', 'crane operator', 'fire hazard', 'barricading', 'safety norms', 'lifeline is not fixed', 'third party inspection', 'tpic', 'no barrier', 'lock and key', 'buzzer', 'gates at landing platforms', 'halogen lamps', 'fall catch net', 'environmental contamination'. Use 'Tower' values as they appear (e.g., 'Common Area'). Collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', and 'Status' into arrays for each site. Assign the count to 'Count' (No. of Safety NCRs beyond 7 days). If no matches, set count to 0 for each site in the data. Return all sites present in the data.\n\n"
                    "Output Format:\n"
                    "{\n"
                    '  "Safety": {\n'
                    '    "Sites": {\n'
                    '      "Site_Name": {\n'
                    '        "Descriptions": [],\n'
                    '        "Created Date (WET)": [],\n'
                    '        "Expected Close Date (WET)": [],\n'
                    '        "Status": [],\n'
                    '        "Count": 0\n'
                    '      }\n'
                    '    },\n'
                    '    "Grand_Total": 0\n'
                    '  }\n'
                    '}\n\n'
                    f"Data: {json.dumps(chunk)}\n"
                )

                payload = {
                    "input": prompt,
                    "parameters": {
                        "decoding_method": "greedy",
                        "max_new_tokens": 500,
                        "min_new_tokens": 0,
                        "temperature": 0.001,
                        "n": 1
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
                    logger.debug("Initiating WatsonX API call...")
                    response = session.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=30)
                    logger.info(f"WatsonX API response status: {response.status_code}")

                    if response.status_code == 200:
                        api_result = response.json()
                        generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                        logger.debug(f"Generated text for chunk {current_chunk}: {generated_text}")

                        json_str = None
                        brace_count = 0
                        start_idx = None
                        for idx, char in enumerate(generated_text):
                            if char == '{':
                                if brace_count == 0:
                                    start_idx = idx
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0 and start_idx is not None:
                                    json_str = generated_text[start_idx:idx + 1]
                                    break

                        if json_str:
                            try:
                                parsed_json = json.loads(json_str)
                                chunk_result = parsed_json.get("Safety", {})
                                chunk_sites = chunk_result.get("Sites", {})
                                chunk_grand_total = chunk_result.get("Grand_Total", 0)

                                for site, values in chunk_sites.items():
                                    if not isinstance(values, dict):
                                        logger.warning(f"Invalid site data for {site}: {values}")
                                        continue
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
                                result["Safety"]["Grand_Total"] += chunk_grand_total
                                logger.debug(f"Successfully processed chunk {current_chunk}/{total_chunks}")
                            except json.JSONDecodeError as e:
                                logger.error(f"JSONDecodeError for chunk {current_chunk}: {str(e)}")
                                error_placeholder.error(f"Failed to parse JSON for chunk {current_chunk}: {str(e)}")
                                for record in chunk:
                                    if is_safety_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
                                        site = record["Tower"]
                                        if site not in result["Safety"]["Sites"]:
                                            result["Safety"]["Sites"][site] = {
                                                "Count": 0,
                                                "Descriptions": [],
                                                "Created Date (WET)": [],
                                                "Expected Close Date (WET)": [],
                                                "Status": []
                                            }
                                        result["Safety"]["Sites"][site]["Descriptions"].append(record["Description"])
                                        result["Safety"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                                        result["Safety"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                                        result["Safety"]["Sites"][site]["Status"].append(record["Status"])
                                        result["Safety"]["Sites"][site]["Count"] += 1
                                        result["Safety"]["Grand_Total"] += 1
                        else:
                            logger.error(f"No valid JSON for chunk {current_chunk}: {generated_text}")
                            error_placeholder.error(f"No valid JSON for chunk {current_chunk}")
                            for record in chunk:
                                if is_safety_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
                                    site = record["Tower"]
                                    if site not in result["Safety"]["Sites"]:
                                        result["Safety"]["Sites"][site] = {
                                            "Count": 0,
                                            "Descriptions": [],
                                            "Created Date (WET)": [],
                                            "Expected Close Date (WET)": [],
                                            "Status": []
                                        }
                                    result["Safety"]["Sites"][site]["Descriptions"].append(record["Description"])
                                    result["Safety"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                                    result["Safety"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                                    result["Safety"]["Sites"][site]["Status"].append(record["Status"])
                                    result["Safety"]["Sites"][site]["Count"] += 1
                                    result["Safety"]["Grand_Total"] += 1
                    else:
                        logger.error(f"WatsonX API error for chunk {current_chunk}: {response.status_code} - {response.text}")
                        error_placeholder.error(f"WatsonX API error for chunk {current_chunk}: {response.status_code}")
                        for record in chunk:
                            if is_safety_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
                                site = record["Tower"]
                                if site not in result["Safety"]["Sites"]:
                                    result["Safety"]["Sites"][site] = {
                                        "Count": 0,
                                        "Descriptions": [],
                                        "Created Date (WET)": [],
                                        "Expected Close Date (WET)": [],
                                        "Status": []
                                    }
                                result["Safety"]["Sites"][site]["Descriptions"].append(record["Description"])
                                result["Safety"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                                result["Safety"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                                result["Safety"]["Sites"][site]["Status"].append(record["Status"])
                                result["Safety"]["Sites"][site]["Count"] += 1
                                result["Safety"]["Grand_Total"] += 1
                except requests.exceptions.ReadTimeout as e:
                    logger.error(f"ReadTimeoutError for chunk {current_chunk}: {str(e)}")
                    error_placeholder.error(f"Failed to connect to WatsonX API for chunk {current_chunk}: {str(e)}")
                    for record in chunk:
                        if is_safety_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
                            site = record["Tower"]
                            if site not in result["Safety"]["Sites"]:
                                result["Safety"]["Sites"][site] = {
                                    "Count": 0,
                                    "Descriptions": [],
                                    "Created Date (WET)": [],
                                    "Expected Close Date (WET)": [],
                                    "Status": []
                                }
                            result["Safety"]["Sites"][site]["Descriptions"].append(record["Description"])
                            result["Safety"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                            result["Safety"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                            result["Safety"]["Sites"][site]["Status"].append(record["Status"])
                            result["Safety"]["Sites"][site]["Count"] += 1
                            result["Safety"]["Grand_Total"] += 1
                except requests.exceptions.RequestException as e:
                    logger.error(f"RequestException for chunk {current_chunk}: {str(e)}")
                    error_placeholder.error(f"Failed to connect to WatsonX API for chunk {current_chunk}: {str(e)}")
                    for record in chunk:
                        if is_safety_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
                            site = record["Tower"]
                            if site not in result["Safety"]["Sites"]:
                                result["Safety"]["Sites"][site] = {
                                    "Count": 0,
                                    "Descriptions": [],
                                    "Created Date (WET)": [],
                                    "Expected Close Date (WET)": [],
                                    "Status": []
                                }
                            result["Safety"]["Sites"][site]["Descriptions"].append(record["Description"])
                            result["Safety"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                            result["Safety"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                            result["Safety"]["Sites"][site]["Status"].append(record["Status"])
                            result["Safety"]["Sites"][site]["Count"] += 1
                            result["Safety"]["Grand_Total"] += 1

            progress_bar.progress(100)
            status_placeholder.write(f"Processed {total_chunks}/{total_chunks} chunks (100%)")
            logger.debug(f"Final result before deduplication: {json.dumps(result, indent=2)}")

            for site in result["Safety"]["Sites"]:
                result["Safety"]["Sites"][site]["Descriptions"] = list(set(result["Safety"]["Sites"][site]["Descriptions"]))
                result["Safety"]["Sites"][site]["Created Date (WET)"] = list(set(result["Safety"]["Sites"][site]["Created Date (WET)"]))
                result["Safety"]["Sites"][site]["Expected Close Date (WET)"] = list(set(result["Safety"]["Sites"][site]["Expected Close Date (WET)"]))
                result["Safety"]["Sites"][site]["Status"] = list(set(result["Safety"]["Sites"][site]["Status"]))
            
            logger.debug(f"Final result after deduplication: {json.dumps(result, indent=2)}")
            return result, json.dumps(result)
        except Exception as e:
            logger.error(f"Unexpected error in generate_ncr_Safety_report: {str(e)}")
            st.error(f"❌ Unexpected Error: {str(e)}")
            return {"error": f"Unexpected Error: {str(e)}"}, ""
        
@st.cache_data
def generate_consolidated_ncr_OpenClose_excel_for_ews(combined_result, report_title="NCR"):
    """
    Generate consolidated NCR report with proper Common Area handling
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formatting styles
        title_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'font_size': 12
        })
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        subheader_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1
        })
        cell_format = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        site_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1
        })
        tower_total_format = workbook.add_format({
            'bold': True, 'align': 'left', 'valign': 'vcenter', 'border': 1
        })
        
        # Generate date string
        now = datetime.now()
        date_part = now.strftime("%d_%B_%Y")
        
        # Standard sites including Common Area
        standard_sites = [
            "EWS Tower 1",
            "EWS Tower 2",
            "EWS Tower 3",
            "LIG Tower 1",
            "LIG Tower 2",
            "LIG Tower 3",
            "Common Area"
        ]
        
        worksheet = workbook.add_worksheet('NCR Report')
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:H', 12)
        
        # Extract data
        resolved_data = combined_result.get("NCR resolved beyond 21 days", {})
        open_data = combined_result.get("NCR open beyond 21 days", {})
        
        if not isinstance(resolved_data, dict) or "error" in resolved_data:
            resolved_data = {"Sites": {}}
        if not isinstance(open_data, dict) or "error" in open_data:
            open_data = {"Sites": {}}
            
        resolved_sites = resolved_data.get("Sites", {})
        open_sites = open_data.get("Sites", {})
        
        # Build comprehensive site mapping using assign_site
        site_mapping = {}
        all_site_keys = set(resolved_sites.keys()) | set(open_sites.keys())
        
        for original_key in all_site_keys:
            # Get descriptions to determine actual tower
            descriptions = []
            if original_key in resolved_sites:
                descriptions.extend(resolved_sites[original_key].get("Descriptions", []))
            if original_key in open_sites:
                descriptions.extend(open_sites[original_key].get("Descriptions", []))
            
            if descriptions:
                # Use first description to determine tower
                detected_sites = assign_site(descriptions[0], standard_sites)
                site_mapping[original_key] = detected_sites[0]
            else:
                # Fallback: try to match the key itself
                detected_sites = assign_site(original_key, standard_sites)
                site_mapping[original_key] = detected_sites[0]
        
        # Write header
        worksheet.merge_range('A1:H1', f"{report_title} {date_part}", title_format)
        row = 1
        worksheet.write(row, 0, 'Site', header_format)
        worksheet.merge_range(row, 1, row, 3, 'NCR resolved beyond 21 days', header_format)
        worksheet.merge_range(row, 4, row, 6, 'NCR open beyond 21 days', header_format)
        worksheet.write(row, 7, 'Total', header_format)
        
        row = 2
        categories = ['Finishing', 'Works', 'MEP']
        worksheet.write(row, 0, '', header_format)
        for i, cat in enumerate(categories):
            worksheet.write(row, i+1, cat, subheader_format)
        for i, cat in enumerate(categories):
            worksheet.write(row, i+4, cat, subheader_format)
        worksheet.write(row, 7, '', header_format)
        
        category_map = {
            'Finishing': ['FW', 'Civil Finishing', 'Finishing'],
            'Works': ['SW', 'Structure', 'Works'],
            'MEP': ['MEP']
        }
        
        row = 3
        
        # Process each standard site
        for site in standard_sites:
            resolved_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
            open_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
            
            # Find all original keys that map to this site
            matching_keys = [k for k, v in site_mapping.items() if v == site]
            
            # Process resolved NCRs
            for original_key in matching_keys:
                if original_key in resolved_sites:
                    site_data = resolved_sites[original_key]
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = site_data.get(key, 0)
                            if isinstance(value, (int, float)) and value > 0:
                                resolved_counts[display_cat] += value
            
            # Process open NCRs
            for original_key in matching_keys:
                if original_key in open_sites:
                    site_data = open_sites[original_key]
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = site_data.get(key, 0)
                            if isinstance(value, (int, float)) and value > 0:
                                open_counts[display_cat] += value
            
            site_total = sum(resolved_counts.values()) + sum(open_counts.values())
            
            # Write site row
            worksheet.write(row, 0, site, tower_total_format)
            for i, display_cat in enumerate(categories):
                worksheet.write(row, i+1, resolved_counts[display_cat], cell_format)
            for i, display_cat in enumerate(categories):
                worksheet.write(row, i+4, open_counts[display_cat], cell_format)
            worksheet.write(row, 7, site_total, cell_format)
            
            row += 1

        # Write detail sheets
        def write_detail_sheet(sheet_name, data, title):
            detail_worksheet = workbook.add_worksheet(sheet_name[:31])
            detail_worksheet.set_column('A:A', 20)
            detail_worksheet.set_column('B:B', 60)
            detail_worksheet.set_column('C:D', 20)
            detail_worksheet.set_column('E:F', 15)
            
            detail_worksheet.merge_range('A1:F1', f"{title} {date_part}", title_format)
            headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline']
            for col, header in enumerate(headers):
                detail_worksheet.write(1, col, header, header_format)
            
            row = 2
            for original_key, site_data in data.items():
                # Get the mapped site name
                mapped_site = site_mapping.get(original_key, "Common Area")
                
                descriptions = site_data.get("Descriptions", [])
                created_dates = site_data.get("Created Date (WET)", [])
                close_dates = site_data.get("Expected Close Date (WET)", [])
                statuses = site_data.get("Status", [])
                disciplines = site_data.get("Discipline", [])
                
                max_length = max(len(descriptions), len(created_dates), len(close_dates), 
                               len(statuses), len(disciplines))
                
                for i in range(max_length):
                    detail_worksheet.write(row, 0, mapped_site, site_format)
                    detail_worksheet.write(row, 1, descriptions[i] if i < len(descriptions) else "", cell_format)
                    detail_worksheet.write(row, 2, created_dates[i] if i < len(created_dates) else "", cell_format)
                    detail_worksheet.write(row, 3, close_dates[i] if i < len(close_dates) else "", cell_format)
                    detail_worksheet.write(row, 4, statuses[i] if i < len(statuses) else "", cell_format)
                    detail_worksheet.write(row, 5, disciplines[i] if i < len(disciplines) else "", cell_format)
                    row += 1

        if resolved_sites:
            write_detail_sheet("Closed NCR Details", resolved_sites, "Closed NCR Details")
        if open_sites:
            write_detail_sheet("Open NCR Details", open_sites, "Open NCR Details")

    output.seek(0)
    return output
    
@st.cache_data
def generate_consolidated_ncr_Housekeeping_excel_for_ews(combined_result, report_title="Housekeeping: Current Month"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        title_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': 'yellow', 'border': 1, 'font_size': 12
        })
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        cell_format = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 1
        })
        site_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1
        })
        description_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        
        report_type = "Closed" if "Closed" in report_title else "Open"
        now = datetime.now()  # April 25, 2025
        day = now.strftime("%d")
        month_name = now.strftime("%B")
        year = now.strftime("%Y")
        date_part = f"{day}_{month_name}_{year}"  # e.g., "25_April_2025"
        report_title = f"Housekeeping: {report_type} - {date_part}"

        def truncate_sheet_name(base_name, max_length=31):
            if len(base_name) > max_length:
                return base_name[:max_length - 3] + "..."
            return base_name

        summary_sheet_name = truncate_sheet_name(f'Housekeeping NCR Report {date_part}')
        details_sheet_name = truncate_sheet_name(f'Housekeeping NCR Details {date_part}')

        worksheet_summary = workbook.add_worksheet(summary_sheet_name)
        worksheet_summary.set_column('A:A', 20)
        worksheet_summary.set_column('B:B', 15)
        
        data = combined_result.get("Housekeeping", {}).get("Sites", {})
        
        standard_sites = [
            "EWS Tower 1",
            "EWS Tower 2",
            "EWS Tower 3",
            "LIG Tower 3",
            "LIG Tower 2",
            "LIG Tower 1"
        ]
        
        def normalize_site_name(site):
            if site in standard_sites:
                return site
            # Add any additional normalization logic here if needed
            return site

        site_mapping = {k: normalize_site_name(k) for k in data.keys()}
        sorted_sites = sorted(standard_sites)
        
        worksheet_summary.merge_range('A1:B1', report_title, title_format)
        row = 1
        worksheet_summary.write(row, 0, 'Site', header_format)
        worksheet_summary.write(row, 1, 'No. of Housekeeping NCRs beyond 7 days', header_format)
        
        row = 2
        for site in sorted_sites:
            worksheet_summary.write(row, 0, site, site_format)
            original_key = next((k for k, v in site_mapping.items() if v == site), None)
            if original_key and original_key in data:
                value = data[original_key].get("Count", 0)
            else:
                value = 0
            worksheet_summary.write(row, 1, value, cell_format)
            row += 1
        
        worksheet_details = workbook.add_worksheet(details_sheet_name)
        worksheet_details.set_column('A:A', 20)
        worksheet_details.set_column('B:B', 60)
        worksheet_details.set_column('C:D', 20)
        worksheet_details.set_column('E:E', 15)
        worksheet_details.set_column('F:F', 15)
        
        worksheet_details.merge_range('A1:F1', f"{report_title} - Details", title_format)
        
        headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline']
        row = 1
        for col, header in enumerate(headers):
            worksheet_details.write(row, col, header, header_format)
        
        row = 2
        for site in sorted_sites:
            original_key = next((k for k, v in site_mapping.items() if v == site), None)
            if original_key and original_key in data:
                site_data = data[original_key]
                descriptions = site_data.get("Descriptions", [])
                created_dates = site_data.get("Created Date (WET)", [])
                close_dates = site_data.get("Expected Close Date (WET)", [])
                statuses = site_data.get("Status", [])
                max_length = max(len(descriptions), len(created_dates), len(close_dates), len(statuses))
                for i in range(max_length):
                    worksheet_details.write(row, 0, site, site_format)
                    worksheet_details.write(row, 1, descriptions[i] if i < len(descriptions) else "", description_format)
                    worksheet_details.write(row, 2, created_dates[i] if i < len(created_dates) else "", cell_format)
                    worksheet_details.write(row, 3, close_dates[i] if i < len(close_dates) else "", cell_format)
                    worksheet_details.write(row, 4, statuses[i] if i < len(statuses) else "", cell_format)
                    worksheet_details.write(row, 5, "HSE", cell_format)
                    row += 1
        
        output.seek(0)
        return output
    
@st.cache_data
def generate_consolidated_ncr_Safety_excel_for_ews(combined_result, report_title=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        title_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': 'yellow', 'border': 1, 'font_size': 12
        })
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        cell_format = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 1
        })
        site_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1
        })
        description_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        
        now = datetime.now()  # April 25, 2025
        day = now.strftime("%d")
        month_name = now.strftime("%B")
        year = now.strftime("%Y")
        date_part = f"{month_name} {day}, {year}"  # e.g., "April 25, 2025"
        if report_title is None:
            report_title = f"Safety: {date_part} - Current Month"
        else:
            report_type = "Safety"
            report_title = f"{date_part}: {report_type}"

        def truncate_sheet_name(base_name, max_length=31):
            if len(base_name) > max_length:
                return base_name[:max_length - 3] + "..."
            return base_name

        summary_sheet_name = truncate_sheet_name(f'Safety NCR Report {date_part}')
        details_sheet_name = truncate_sheet_name(f'Safety NCR Details {date_part}')

        worksheet_summary = workbook.add_worksheet(summary_sheet_name)
        worksheet_summary.set_column('A:A', 20)
        worksheet_summary.set_column('B:B', 15)
        
        data = combined_result.get("Safety", {}).get("Sites", {})
        
        standard_sites = [
            "EWS Tower 1",
            "EWS Tower 2",
            "EWS Tower 3",
            "LIG Tower 3",
            "LIG Tower 2",
            "LIG Tower 1"
        ]
        
        def normalize_site_name(site):
            if site in standard_sites:
                return site
            # Add any additional normalization logic here if needed
            return site

        site_mapping = {k: normalize_site_name(k) for k in data.keys()}
        sorted_sites = sorted(standard_sites)
        
        worksheet_summary.merge_range('A1:B1', report_title, title_format)
        row = 1
        worksheet_summary.write(row, 0, 'Site', header_format)
        worksheet_summary.write(row, 1, 'No. of Safety NCRs beyond 7 days', header_format)
        
        row = 2
        for site in sorted_sites:
            worksheet_summary.write(row, 0, site, site_format)
            original_key = next((k for k, v in site_mapping.items() if v == site), None)
            if original_key and original_key in data:
                value = data[original_key].get("Count", 0)
            else:
                value = 0
            worksheet_summary.write(row, 1, value, cell_format)
            row += 1
        
        worksheet_details = workbook.add_worksheet(details_sheet_name)
        worksheet_details.set_column('A:A', 20)
        worksheet_details.set_column('B:B', 60)
        worksheet_details.set_column('C:D', 20)
        worksheet_details.set_column('E:E', 15)
        worksheet_details.set_column('F:F', 15)

        worksheet_details.merge_range('A1:F1', f"{report_title} - Details", title_format)
        
        headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline']
        row = 1
        for col, header in enumerate(headers):
            if col == 5:
                worksheet_details.write(row, col, header, title_format)
            else:
                worksheet_details.write(row, col, header, header_format)
        
        row = 2
        for site in sorted_sites:
            original_key = next((k for k, v in site_mapping.items() if v == site), None)
            if original_key and original_key in data:
                site_data = data[original_key]
                descriptions = site_data.get("Descriptions", [])
                created_dates = site_data.get("Created Date (WET)", [])
                close_dates = site_data.get("Expected Close Date (WET)", [])
                statuses = site_data.get("Status", [])
                max_length = max(len(descriptions), len(created_dates), len(close_dates), len(statuses))
                for i in range(max_length):
                    worksheet_details.write(row, 0, site, site_format)
                    worksheet_details.write(row, 1, descriptions[i] if i < len(descriptions) else "", description_format)
                    worksheet_details.write(row, 2, created_dates[i] if i < len(created_dates) else "", cell_format)
                    worksheet_details.write(row, 3, close_dates[i] if i < len(close_dates) else "", cell_format)
                    worksheet_details.write(row, 4, statuses[i] if i < len(statuses) else "", cell_format)
                    worksheet_details.write(row, 5, "HSE", cell_format)
                    row += 1
        
        output.seek(0)
        return output
    
@st.cache_data
@st.cache_data
def generate_combined_excel_report_for_ews(all_reports, filename_prefix="All_Reports"):
    """
    Combined Excel export with proper Common Area handling, tower detection, and colored rows
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # ---------- Formats ----------
        title_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter',
            'fg_color': 'yellow', 'border': 1, 'font_size': 12
        })
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter',
            'border': 1, 'text_wrap': True
        })
        subheader_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1
        })
        description_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        default_cell_format = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 1
        })
        default_site_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1
        })

        # ---------- Tower Colors ----------
        tower_colors = {
            "EWS Tower 1": '#F9E79F',  # Soft yellow
            "EWS Tower 2": '#A3CFFA',  # Blue
            "EWS Tower 3": '#F5C3C2',  # Pink
            "LIG Tower 1": '#C4E4B7',  # Green
            "LIG Tower 2": '#F5E8B7',  # Soft yellow
            "LIG Tower 3": '#D7B9F5',  # Soft lavender
            "Common Area": '#C7CEEA'   # Soft purple
        }

        tower_formats = {}
        for site, color in tower_colors.items():
            tower_formats[site] = workbook.add_format({
                'align': 'center', 'valign': 'vcenter',
                'border': 1,
                'fg_color': color
            })

        # ---------- Date ----------
        now = datetime.now()
        date_part = now.strftime("%d_%B_%Y")

        # ---------- Standard Sites ----------
        standard_sites = [
            "EWS Tower 1",
            "EWS Tower 2",
            "EWS Tower 3",
            "LIG Tower 1",
            "LIG Tower 2",
            "LIG Tower 3",
            "Common Area"
        ]

        # ---------- Helper ----------
        def detect_site_from_description(description):
            """Use assign_site() to find correct tower or Common Area."""
            result = assign_site(description, standard_sites)
            return result[0] if result else "Common Area"

        # ---------- NCR Data ----------
        combined_result = all_reports.get("Combined_NCR", {})
        resolved_data = combined_result.get("NCR resolved beyond 21 days", {})
        open_data = combined_result.get("NCR open beyond 21 days", {})

        resolved_sites = resolved_data.get("Sites", {}) if isinstance(resolved_data, dict) else {}
        open_sites = open_data.get("Sites", {}) if isinstance(open_data, dict) else {}

        site_mapping = {}
        all_keys = set(resolved_sites.keys()) | set(open_sites.keys())
        for k in all_keys:
            descs = []
            if k in resolved_sites:
                descs.extend(resolved_sites[k].get("Descriptions", []))
            if k in open_sites:
                descs.extend(open_sites[k].get("Descriptions", []))
            if descs:
                site_mapping[k] = detect_site_from_description(descs[0])
            else:
                site_mapping[k] = detect_site_from_description(k)

        # ---------- NCR Summary Sheet ----------
        worksheet = workbook.add_worksheet('NCR Report')
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:H', 14)

        # Row 0: Title
        worksheet.merge_range('A1:H1', f"NCR Summary - {date_part}", title_format)

        # Row 1: Column headers
        worksheet.write(1, 0, 'Site', header_format)
        worksheet.merge_range(1, 1, 1, 3, 'NCR resolved >21 days', header_format)
        worksheet.merge_range(1, 4, 1, 6, 'NCR open >21 days', header_format)
        worksheet.write(1, 7, 'Total', header_format)

        # Row 2: Subheaders
        row = 2
        categories = ['Finishing', 'Works', 'MEP']
        worksheet.write(row, 0, '', header_format)
        for i, cat in enumerate(categories):
            worksheet.write(row, i + 1, cat, subheader_format)
        for i, cat in enumerate(categories):
            worksheet.write(row, i + 4, cat, subheader_format)
        worksheet.write(row, 7, '', header_format)
        row = 3

        category_map = {
            'Finishing': ['FW', 'Civil Finishing', 'Finishing'],
            'Works': ['SW', 'Structure', 'Works'],
            'MEP': ['MEP']
        }

        # Write site data with coloring
        for site in standard_sites:
            resolved_counts = {cat: 0 for cat in categories}
            open_counts = {cat: 0 for cat in categories}
            matching_keys = [k for k, v in site_mapping.items() if v == site]

            for original_key in matching_keys:
                if original_key in resolved_sites:
                    sdata = resolved_sites[original_key]
                    for cat, keys in category_map.items():
                        for k in keys:
                            val = sdata.get(k, 0)
                            if isinstance(val, (int, float)):
                                resolved_counts[cat] += val
                if original_key in open_sites:
                    sdata = open_sites[original_key]
                    for cat, keys in category_map.items():
                        for k in keys:
                            val = sdata.get(k, 0)
                            if isinstance(val, (int, float)):
                                open_counts[cat] += val

            site_total = sum(resolved_counts.values()) + sum(open_counts.values())
            cell_format = tower_formats.get(site, default_cell_format)

            worksheet.write(row, 0, site, cell_format)
            for i, cat in enumerate(categories):
                worksheet.write(row, i + 1, resolved_counts[cat], cell_format)
                worksheet.write(row, i + 4, open_counts[cat], cell_format)
            worksheet.write(row, 7, site_total, cell_format)
            row += 1

        # ---------- NCR Details ----------
        def write_detail_sheet(sheet_name, data_dict, title):
            sheet = workbook.add_worksheet(sheet_name[:30])
            sheet.merge_range('A1:F1', f"{title} - {date_part}", title_format)
            headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline']
            for col, h in enumerate(headers):
                sheet.write(1, col, h, header_format)
            sheet.set_column('A:A', 20)
            sheet.set_column('B:B', 60)
            sheet.set_column('C:D', 20)
            sheet.set_column('E:F', 15)
            
            row = 2
            for key, sdata in data_dict.items():
                site = site_mapping.get(key, "Common Area")
                descs = sdata.get("Descriptions", [])
                created = sdata.get("Created Date (WET)", [])
                close = sdata.get("Expected Close Date (WET)", [])
                status = sdata.get("Status", [])
                disc = sdata.get("Discipline", [])
                maxlen = max(len(descs), len(created), len(close), len(status), len(disc))
                
                for i in range(maxlen):
                    sheet.write(row, 0, site, default_site_format)
                    sheet.write(row, 1, descs[i] if i < len(descs) else "", description_format)
                    sheet.write(row, 2, created[i] if i < len(created) else "", default_cell_format)
                    sheet.write(row, 3, close[i] if i < len(close) else "", default_cell_format)
                    sheet.write(row, 4, status[i] if i < len(status) else "", default_cell_format)
                    sheet.write(row, 5, disc[i] if i < len(disc) else "", default_cell_format)
                    row += 1

        if resolved_sites:
            write_detail_sheet("Closed NCR Details", resolved_sites, "Closed NCR Details")
        if open_sites:
            write_detail_sheet("Open NCR Details", open_sites, "Open NCR Details")

        # ---------- Safety & Housekeeping ----------
        def write_safety_housekeeping_report(report_type, data, report_title, sheet_type):
            sheet = workbook.add_worksheet(f"{report_type} NCR {sheet_type}"[:31])
            sheet.merge_range('A1:B1', f"{report_title} - {sheet_type}", title_format)
            sheet.write(1, 0, 'Site', header_format)
            sheet.write(1, 1, f'No. of {report_type} NCRs >7 days', header_format)
            sheet.set_column('A:A', 20)
            sheet.set_column('B:B', 15)
            row = 2

            sites_data = data.get(report_type, {}).get("Sites", {}) if isinstance(data, dict) else {}
            
            for site in standard_sites:
                sheet.write(row, 0, site, default_site_format)
                count = 0
                for k, s in sites_data.items():
                    descs = s.get("Descriptions", [])
                    detected = detect_site_from_description(descs[0]) if descs else detect_site_from_description(k)
                    if detected == site:
                        count += s.get("Count", 0)
                sheet.write(row, 1, count, default_cell_format)
                row += 1

        # Safety + Housekeeping reports
        write_safety_housekeeping_report("Safety", all_reports.get("Safety_NCR_Closed", {}), "Safety NCR", "Closed")
        write_safety_housekeeping_report("Safety", all_reports.get("Safety_NCR_Open", {}), "Safety NCR", "Open")
        write_safety_housekeeping_report("Housekeeping", all_reports.get("Housekeeping_NCR_Closed", {}), "Housekeeping NCR", "Closed")
        write_safety_housekeeping_report("Housekeeping", all_reports.get("Housekeeping_NCR_Open", {}), "Housekeeping NCR", "Open")

    output.seek(0)
    return output


# Streamlit UI


# Data Fetch Section (unchanged)


# Report Generation Section


# Generate Combined NCR Report


# Helper function to generate report title
def generate_report_title(prefix):
    now = datetime.now()  # Current date: April 25, 2025
    day = now.strftime("%d")
    month_name = now.strftime("%B")
    year = now.strftime("%Y")
    return f"{prefix}: {day}_{month_name}_{year}"

# Generate Safety NCR Report


# Generate Housekeeping NCR Report


# All Reports Button

        
