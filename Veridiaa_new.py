# -*- coding: utf-8 -*-
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
    
    # Capture end time
    end_time = datetime.now()
    st.write(f" 🔄 Fetching data from Asite completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {(end_time - start_time).total_seconds()} seconds)")
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
            logger.error(f"Failed to parse JSON even after extraction: {json_str}")
            
    logger.error(f"Could not extract valid JSON from: {text}")
    return None

# Generate NCR Report
@st.cache_data
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type((requests.RequestException, ValueError, KeyError)))
def generate_ncr_report_for_veridia(df: pd.DataFrame, report_type: str, start_date=None, end_date=None, Until_Date=None) -> Tuple[Dict[str, Any], str]:
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
            
            df = df.copy()
            
            # Check if required columns exist
            required_columns = ['Created Date (WET)', 'Status']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"❌ Missing required columns: {missing_columns}"
                st.error(error_msg)
                return {"error": f"Missing columns: {missing_columns}"}, ""
            
            df = df[df['Created Date (WET)'].notna()]
            
            if report_type == "Closed":
                if 'Expected Close Date (WET)' not in df.columns:
                    st.error("❌ 'Expected Close Date (WET)' column is required for Closed reports")
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
            else:
                if Until_Date is None:
                    st.error("❌ Open Until Date is required for Open NCR Report")
                    return {"error": "Open Until Date is required"}, ""
                
                today = pd.to_datetime(Until_Date)
                filtered_df = df[
                    (df['Status'] == 'Open') &
                    (df['Created Date (WET)'].notna())
                ].copy()
                filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
                filtered_df = filtered_df[filtered_df['Days_From_Today'] > 21].copy()

            if filtered_df.empty:
                st.warning(f"No {report_type} NCRs found with duration > 21 days.")
                return {"error": f"No {report_type} records found with duration > 21 days"}, ""

            filtered_df.loc[:, 'Created Date (WET)'] = filtered_df['Created Date (WET)'].astype(str)
            if 'Expected Close Date (WET)' in filtered_df.columns:
                filtered_df.loc[:, 'Expected Close Date (WET)'] = filtered_df['Expected Close Date (WET)'].astype(str)

            processed_data = filtered_df.to_dict(orient="records")
            
            cleaned_data = []
            unique_records = []
            common_pattern = re.compile(r"common area|flat\s*no", re.IGNORECASE)

            for record in processed_data:
                try:
                    cleaned_record = {
                        "Description": str(record.get("Description", "")),
                        "Discipline": str(record.get("Discipline", "")),
                        "Created Date (WET)": str(record.get("Created Date (WET)", "")),
                        "Expected Close Date (WET)": str(record.get("Expected Close Date (WET)", "")),
                        "Status": str(record.get("Status", "")),
                        "Days": int(record.get("Days", 0)) if pd.notna(record.get("Days")) else 0,
                        "Tower": "External Development"
                    }
                    
                    if report_type == "Open":
                        cleaned_record["Days_From_Today"] = int(record.get("Days_From_Today", 0)) if pd.notna(record.get("Days_From_Today")) else 0

                    description = cleaned_record["Description"].lower().strip()
                    if not description:
                        continue
                    # ✅ Prevent duplicate counting of same NCR description (Tower 7 overcount fix)
                    # if description in unique_records:
                    #     continue
                    # unique_records.append(description)  

                    # ✅ FIXED MODULE EXTRACTION LOGIC (Tower 7 issue solved)
                    if common_pattern.search(description):
                        cleaned_record["Modules"] = ["Common"]
                    else:
                        modules = set()

                        # Handle ranges like "T-7 M1-3", "Tower 5, Module 1 to 4"
                        range_pattern = r"(?:tower|t)?\s*-?\s*\d*\s*(?:module|mod|m)[-\s]*(\d+)\s*(?:to|-|–)\s*(\d+)"
                        for start_str, end_str in re.findall(range_pattern, description, re.IGNORECASE):
                            try:
                                start, end = int(start_str), int(end_str)
                                if 0 < start <= end <= 50:
                                    modules.update(f"M{i}" for i in range(start, end + 1))
                            except ValueError:
                                continue

                      # Handle lists like "Module 2&3", "M-7 & 6", "Mod 1,2,3"
                        list_pattern = r"(?:module|mod|m)[-\s]*((?:\d+\s*(?:,|&|and)?\s*)+)(?=\b|[^a-z])"
                        for match in re.findall(list_pattern, description, re.IGNORECASE):
                            for num in re.findall(r"\b\d{1,2}\b(?!\s*(?:mm|th|rd|nd|st|floor))", match, re.IGNORECASE):
                                try:
                                    num = int(num)
                                    if 0 < num <= 50:
                                        modules.add(f"M{num}")
                                except ValueError:
                                    continue

                        # Handle individual single mentions like "M-2", "Mod 5"
                        if not modules:
                            for num in re.findall(r"(?:module|mod|m)[-\s]*(\d{1,2})(?!\s*(?:mm|th|rd|nd|st|floor))", description, re.IGNORECASE):
                                try:
                                    num = int(num)
                                    if 0 < num <= 50:
                                        modules.add(f"M{num}")
                                except ValueError:
                                    continue
                                
                        cleaned_record["Modules"] = sorted(list(modules)) if modules else ["Common"]

                    # Discipline categorization
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

                    unique_records.append(cleaned_record["Description"])

                    # Tower categorization (unchanged)
                    if any(phrase in description for phrase in ["veridia clubhouse", "veridia-clubhouse", "veridia club"]):
                        cleaned_record["Tower"] = "Veridia-Club"
                        cleaned_data.append(cleaned_record)
                    else:
                        tower_matches = re.findall(r"(tower|t)\s*-?\s*(\d+)", description, re.IGNORECASE)
                        multiple_tower_pattern = re.search(
                            r"(tower|t)\s*-?\s*(\d+)\s*([,&]|and)\s*(tower|t)?\s*-?\s*(\d+)",
                            description,
                            re.IGNORECASE
                        )
                        flat_no_pattern = re.search(r"flat\s*no", description, re.IGNORECASE)
                        
                        if multiple_tower_pattern:
                            tower1 = multiple_tower_pattern.group(2).zfill(2)
                            tower2 = multiple_tower_pattern.group(5).zfill(2)
                            for tower_num in [tower1, tower2]:
                                tower_record = cleaned_record.copy()
                                tower_record["Tower"] = f"Veridia-Tower-{tower_num}-CommonArea"
                                cleaned_data.append(tower_record)
                        elif flat_no_pattern and tower_matches:
                            tower_num = tower_matches[0][1].zfill(2)
                            cleaned_record["Tower"] = f"Veridia-Tower-{tower_num}"
                            cleaned_data.append(cleaned_record)
                        elif "common area" in description or not tower_matches:
                            cleaned_record["Tower"] = "Common_Area"
                            cleaned_data.append(cleaned_record)
                        else:
                            tower_num = tower_matches[0][1].zfill(2)
                            cleaned_record["Tower"] = f"Veridia-Tower-{tower_num}"
                            cleaned_data.append(cleaned_record)
                            
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
                    f"Task: Group the provided data by 'Tower' and collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline', and 'Modules' into arrays. "
                    f"Count the records by 'Discipline_Category' ('SW', 'FW', 'MEP'), calculate the 'Total' for each 'Tower', and count occurrences of each module within 'Modules' (e.g., M1, M2). "
                    f"Process ALL {len(chunk)} records provided in the data.\n"
                    f"Use 'Tower' values (e.g., 'Veridia-Tower-04-CommonArea', 'Veridia-Tower-07-CommonArea', 'Common_Area'), "
                    f"'Discipline_Category' values (e.g., 'SW', 'FW', 'MEP'), and provided 'Modules' values. Count each record exactly once.\n\n"
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
                    '        "Modules": [["module1a", "module1b"], ["module2"]],\n'
                    '        "SW": number,\n'
                    '        "FW": number,\n'
                    '        "MEP": number,\n'
                    '        "Total": number,\n'
                    '        "ModulesCount": {"module1": count1, "module2": count2}\n'
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
                        "max_new_tokens": 5100,
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
                    response = http.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=1000)
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
                                        "Modules": [],
                                        "SW": 0,
                                        "FW": 0,
                                        "MEP": 0,
                                        "Total": 0,
                                        "ModulesCount": {}
                                    }
                                all_results[report_type]["Sites"][site]["Descriptions"].extend(data["Descriptions"])
                                all_results[report_type]["Sites"][site]["Created Date (WET)"].extend(data["Created Date (WET)"])
                                all_results[report_type]["Sites"][site]["Expected Close Date (WET)"].extend(data["Expected Close Date (WET)"])
                                all_results[report_type]["Sites"][site]["Status"].extend(data["Status"])
                                all_results[report_type]["Sites"][site]["Discipline"].extend(data["Discipline"])
                                all_results[report_type]["Sites"][site]["Modules"].extend(data["Modules"])
                                all_results[report_type]["Sites"][site]["SW"] += data["SW"]
                                all_results[report_type]["Sites"][site]["FW"] += data["FW"]
                                all_results[report_type]["Sites"][site]["MEP"] += data["MEP"]
                                all_results[report_type]["Sites"][site]["Total"] += data["Total"]
                                for module, count in data["ModulesCount"].items():
                                    all_results[report_type]["Sites"][site]["ModulesCount"][module] = all_results[report_type]["Sites"][site]["ModulesCount"].get(module, 0) + count
                                            
                            
                            # Use the actual number of records processed instead of API count
                            all_results[report_type]["Grand_Total"] += len(chunk)
                            st.write(f"Successfully processed chunk {i // chunk_size + 1} with {len(chunk)} records")
                        else:
                            logger.error("No valid JSON found in response")
                            st.error("❌ No valid JSON found in response")
                            st.write("Falling back to local count for this chunk")
                            process_chunk_locally(chunk, all_results, report_type)
                    else:
                        error_msg = f"❌ WatsonX API error: {response.status_code} - {response.text}"
                        st.error(error_msg)
                        logger.error(error_msg)
                        st.write("Falling back to local count for this chunk")
                        process_chunk_locally(chunk, all_results, report_type)
                        
                except requests.RequestException as e:
                    error_msg = f"❌ Request exception during WatsonX call: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.write("Falling back to local count for this chunk")
                    process_chunk_locally(chunk, all_results, report_type)
                except Exception as e:
                    error_msg = f"❌ Exception during WatsonX call: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.write("Falling back to local count for this chunk")
                    process_chunk_locally(chunk, all_results, report_type)

            # Capture and log end time after API call
            end_time = datetime.now()
            st.write(f"🔄 Chunk {i // chunk_size + 1} model processing for {report_type} completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {(end_time - start_time).total_seconds()} seconds)")
            logger.info(f"Finished model processing chunk {i // chunk_size + 1} for {report_type} at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {(end_time - start_time).total_seconds()} seconds)")

            # Convert all_results to a DataFrame for table display
            table_data = []
            for site, data in all_results[report_type]["Sites"].items():
                row = {
                    "Site": site,
                    "SW Count": data["SW"],
                    "FW Count": data["FW"],
                    "MEP Count": data["MEP"],
                    "Total Records": data["Total"],
                    "Modules Count": json.dumps(data["ModulesCount"], indent=2),
                    "Descriptions": "; ".join(data["Descriptions"]),
                    "Created Dates": "; ".join(data["Created Date (WET)"]),
                    "Expected Close Dates": "; ".join(data["Expected Close Date (WET)"]),
                    "Statuses": "; ".join(data["Status"]),
                    "Disciplines": "; ".join(data["Discipline"]),
                    "Modules": "; ".join([", ".join(m) for m in data["Modules"]])
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

@st.cache_data
def process_chunk_locally(chunk, all_results, report_type):
    """Helper function to process chunks locally when API fails"""
    try:
        for record in chunk:
            tower = record.get("Tower", "Unknown")
            discipline = record.get("Discipline_Category", "Unknown")
            modules = record.get("Modules", ["Common"])
            
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
                    "ModulesCount": {}
                }
            
            all_results[report_type]["Sites"][tower]["Descriptions"].append(record.get("Description", ""))
            all_results[report_type]["Sites"][tower]["Created Date (WET)"].append(record.get("Created Date (WET)", ""))
            all_results[report_type]["Sites"][tower]["Expected Close Date (WET)"].append(record.get("Expected Close Date (WET)", ""))
            all_results[report_type]["Sites"][tower]["Status"].append(record.get("Status", ""))
            all_results[report_type]["Sites"][tower]["Discipline"].append(record.get("Discipline", ""))
            all_results[report_type]["Sites"][tower]["Modules"].append(modules)
            
            if discipline in ["SW", "FW", "MEP"]:
                all_results[report_type]["Sites"][tower][discipline] += 1
            
            all_results[report_type]["Sites"][tower]["Total"] += 1
            
            for module in modules:
                all_results[report_type]["Sites"][tower]["ModulesCount"][module] = all_results[report_type]["Sites"][tower]["ModulesCount"].get(module, 0) + 1
                
            all_results[report_type]["Grand_Total"] += 1
            
    except Exception as e:
        logger.error(f"Error in local processing: {str(e)}")
        raise

# Generate NCR Housekeeping Report
@st.cache_data
def generate_ncr_Housekeeping_report_for_veridia(df, report_type, start_date=None, end_date=None, open_until_date=None):
    with st.spinner(f"Generating {report_type} Housekeeping NCR Report with WatsonX..."):
        today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
        closed_start = pd.to_datetime(start_date) if start_date else None
        closed_end = pd.to_datetime(end_date) if end_date else None
        open_until = pd.to_datetime(open_until_date)

        if report_type == "Closed":
            filtered_df = df[
                (df['Discipline'] == 'HSE') &
                (df['Status'] == 'Closed') &
                (df['Days'].notnull()) &
                (df['Days'] > 7)
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
                (pd.to_datetime(df['Created Date (WET)']).notna())
            ].copy()
            filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
            filtered_df = filtered_df[filtered_df['Days_From_Today'] > 7].copy()
            if open_until:
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['Created Date (WET)']) <= open_until)
                ].copy()

        if filtered_df.empty:
            return {"error": f"No {report_type} records found with duration > 7 days"}, ""

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
                    "Tower": "External Development"
                }

                desc_lower = description.lower()
                if any(phrase in desc_lower for phrase in ["veridia clubhouse", "veridia-clubhouse", "veridia club"]):
                    cleaned_record["Tower"] = "Veridia-Club"
                    logger.debug(f"Matched 'Veridia Clubhouse', setting Tower to Veridia-Club")
                else:
                    tower_match = re.search(r"(tower|t)\s*-?\s*(\d+|2021|28)", desc_lower, re.IGNORECASE)
                    cleaned_record["Tower"] = f"Veridia-Tower{tower_match.group(2).zfill(2)}" if tower_match else "Common_Area"
                    logger.debug(f"Tower set to {cleaned_record['Tower']}")

                cleaned_data.append(cleaned_record)

        st.write(f"Total {report_type} records to process: {len(cleaned_data)}")
        logger.debug(f"Processed data: {json.dumps(cleaned_data, indent=2)}")

        if not cleaned_data:
            return {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}, ""

        access_token = get_access_token(API_KEY)
        if not access_token:
            return {"error": "Failed to obtain access token"}, ""

        result = {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}
        chunk_size = 1
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
                "IMPORTANT: RETURN ONLY A SINGLE VALID JSON OBJECT WITH THE EXACT FIELDS SPECIFIED BELOW. "
                "DO NOT GENERATE ANY CODE (e.g., Python, JavaScript). "
                "DO NOT INCLUDE ANY TEXT, EXPLANATIONS, OR MULTIPLE RESPONSES OUTSIDE THE JSON OBJECT. "
                "DO NOT WRAP THE JSON IN CODE BLOCKS (e.g., ```json). "
                "RETURN THE JSON OBJECT DIRECTLY.\n\n"
                "Return the result strictly as a single JSON object—no code, no explanations, no string literal like this ```, only the JSON."
                "DO NOT INCLUDE EXAMPLES, EXPLANATIONS, COMMENTS, OR ANY ADDITIONAL TEXT BEYOND THE JSON OBJECT. "
                "DO NOT WRAP THE JSON IN CODE BLOCKS (e.g., ```). "
                "DO NOT GENERATE EXAMPLE OUTPUTS FOR OTHER SCENARIOS. "
                "ONLY PROCESS THE PROVIDED DATA AND RETURN THE RESULT.\n\n"
                "Task: For Housekeeping NCRs, count EVERY record in the provided data by site ('Tower' field) where 'Discipline' is 'HSE' and 'Days' is greater than 7. "
                "The 'Description' MUST be counted if it contains ANY of the following housekeeping issues (match these keywords exactly as provided, case-insensitive): "
                "'housekeeping','cleaning','cleanliness','waste disposal','waste management','garbage','trash','rubbish','debris','litter','dust','untidy',"
                "'cluttered','accumulation of waste','construction waste','pile of garbage','poor housekeeping','material storage','construction debris',"
                "'cleaning schedule','garbage collection','waste bins','dirty','mess','unclean','disorderly','dirty floor','waste disposal area',"
                "'waste collection','cleaning protocol','sanitation','trash removal','waste accumulation','unkept area','refuse collection','workplace cleanliness'. "
                "Use the 'Tower' values exactly as they appear in the data (e.g., 'Veridia-Club', 'Veridia-Tower01', 'Common_Area'). "
                "Collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', and 'Status' into arrays for each site. "
                "Assign each count to the 'Count' key, representing 'No. of Housekeeping NCRs beyond 7 days'. "
                "If no matches are found for a site, set its count to 0, but ensure all present sites in the data are listed. "
                "INCLUDE ONLY records where housekeeping is the PRIMARY concern and EXCLUDE records that are primarily about safety issues (e.g., descriptions focusing on 'safety precautions', 'PPE', 'fall protection').\n\n"
                "REQUIRED OUTPUT FORMAT (use this structure with the actual results):\n"
                "{\n"
                '  "Housekeeping": {\n'
                '    "Sites": {\n'
                '      "Site_Name1": {\n'
                '        "Descriptions": ["description1", "description2"],\n'
                '        "Created Date (WET)": ["date1", "date2"],\n'
                '        "Expected Close Date (WET)": ["date1", "date2"],\n'
                '        "Status": ["status1", "status2"],\n'
                '        "Count": number\n'
                '      },\n'
                '      "Site_Name2": {\n'
                '        "Descriptions": ["description1", "description2"],\n'
                '        "Created Date (WET)": ["date1", "date2"],\n'
                '        "Expected Close Date (WET)": ["date1", "date2"],\n'
                '        "Status": ["status1", "status2"],\n'
                '        "Count": number\n'
                '      }\n'
                '    },\n'
                '    "Grand_Total": number\n'
                '  }\n'
                '}\n\n'
                f"IMPORTANT: Ensure the JSON is valid and contains all required fields. "    
                f"Return the result strictly as a JSON object—no code, no explanations, only the JSON.Dont put <|eom_id|> or any other markers in the JSON output."
                f"Data: {json.dumps(chunk)}\n"
            )

            payload = {
                "input": prompt,
                "parameters": {"decoding_method": "greedy", "max_new_tokens": 8100, "min_new_tokens": 0, "temperature": 0.001},
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
                response = session.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=900)
                logger.info(f"WatsonX API response status: {response.status_code}")

                if response.status_code == 200:
                    api_result = response.json()
                    generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                    logger.debug(f"Generated text for chunk {current_chunk}: {generated_text}")

                    if generated_text:
                        # Extract the JSON portion by finding the first complete JSON object
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
                                        logger.warning(f"Invalid site data for {site}: {values}, converting to dict")
                                        values = {
                                            "Count": int(values) if isinstance(values, (int, float)) else 0,
                                            "Descriptions": [],
                                            "Created Date (WET)": [],
                                            "Expected Close Date (WET)": [],
                                            "Status": []
                                        }
                                    
                                    if site not in result["Housekeeping"]["Sites"]:
                                        result["Housekeeping"]["Sites"][site] = {
                                            "Count": 0,
                                            "Descriptions": [],
                                            "Created Date (WET)": [],
                                            "Expected Close Date (WET)": [],
                                            "Status": []
                                        }
                                    
                                    if "Descriptions" in values and values["Descriptions"]:
                                        if not isinstance(values["Descriptions"], list):
                                            values["Descriptions"] = [str(values["Descriptions"])]
                                        result["Housekeeping"]["Sites"][site]["Descriptions"].extend(values["Descriptions"])
                                    
                                    if "Created Date (WET)" in values and values["Created Date (WET)"]:
                                        if not isinstance(values["Created Date (WET)"], list):
                                            values["Created Date (WET)"] = [str(values["Created Date (WET)"])]
                                        result["Housekeeping"]["Sites"][site]["Created Date (WET)"].extend(values["Created Date (WET)"])
                                    
                                    if "Expected Close Date (WET)" in values and values["Expected Close Date (WET)"]:
                                        if not isinstance(values["Expected Close Date (WET)"], list):
                                            values["Expected Close Date (WET)"] = [str(values["Expected Close Date (WET)"])]
                                        result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"].extend(values["Expected Close Date (WET)"])
                                    
                                    if "Status" in values and values["Status"]:
                                        if not isinstance(values["Status"], list):
                                            values["Status"] = [str(values["Status"])]
                                        result["Housekeeping"]["Sites"][site]["Status"].extend(values["Status"])
                                    
                                    count = values.get("Count", 0)
                                    if not isinstance(count, (int, float)):
                                        count = 0
                                    result["Housekeeping"]["Sites"][site]["Count"] += count
                                
                                result["Housekeeping"]["Grand_Total"] += chunk_grand_total
                                logger.debug(f"Successfully processed chunk {current_chunk}/{total_chunks}")
                            except json.JSONDecodeError as e:
                                logger.error(f"JSONDecodeError for chunk {current_chunk}: {str(e)} - Raw: {json_str}")
                                error_placeholder.error(f"Failed to parse JSON for chunk {current_chunk}: {str(e)}")
                                # Fallback: Manually process the chunk
                                for record in chunk:
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
                                logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
                        else:
                            logger.error(f"No valid JSON found in response for chunk {current_chunk}: {generated_text}")
                            error_placeholder.error(f"No valid JSON found in response for chunk {current_chunk}")
                            # Fallback: Manually process the chunk
                            for record in chunk:
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
                            logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
                    else:
                        logger.error(f"Empty WatsonX response for chunk {current_chunk}")
                        error_placeholder.error(f"Empty WatsonX response for chunk {current_chunk}")
                        # Fallback: Manually process the chunk
                        for record in chunk:
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
                        logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
                else:
                    logger.error(f"WatsonX API error for chunk {current_chunk}: {response.status_code} - {response.text}")
                    error_placeholder.error(f"WatsonX API error for chunk {current_chunk}: {response.status_code} - {response.text}")
                    # Fallback: Manually process the chunk
                    for record in chunk:
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
                    logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
            except requests.exceptions.ReadTimeout as e:
                logger.error(f"ReadTimeoutError after retries for chunk {current_chunk}: {str(e)}")
                error_placeholder.error(f"Failed to connect to WatsonX API for chunk {current_chunk} after retries due to timeout: {str(e)}")
                # Fallback: Manually process the chunk
                for record in chunk:
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
                logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
            except requests.exceptions.RequestException as e:
                logger.error(f"RequestException for chunk {current_chunk}: {str(e)}")
                error_placeholder.error(f"Failed to connect to WatsonX API for chunk {current_chunk}: {str(e)}")
                # Fallback: Manually process the chunk
                for record in chunk:
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
                logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
            except Exception as e:
                logger.error(f"Unexpected error during WatsonX API call for chunk {current_chunk}: {str(e)}")
                error_placeholder.error(f"Unexpected error during WatsonX API call for chunk {current_chunk}: {str(e)}")
                # Fallback: Manually process the chunk
                for record in chunk:
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
                logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")

        progress_bar.progress(100)
        status_placeholder.write(f"Processed {total_chunks}/{total_chunks} chunks (100%)")
        logger.debug(f"Final result before deduplication: {json.dumps(result, indent=2)}")

        for site in result["Housekeeping"]["Sites"]:
            if "Descriptions" in result["Housekeeping"]["Sites"][site]:
                result["Housekeeping"]["Sites"][site]["Descriptions"] = list(set(result["Housekeeping"]["Sites"][site]["Descriptions"]))
            if "Created Date (WET)" in result["Housekeeping"]["Sites"][site]:
                result["Housekeeping"]["Sites"][site]["Created Date (WET)"] = list(set(result["Housekeeping"]["Sites"][site]["Created Date (WET)"]))
            if "Expected Close Date (WET)" in result["Housekeeping"]["Sites"][site]:
                result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"] = list(set(result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"]))
            if "Status" in result["Housekeeping"]["Sites"][site]:
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
    json_match = re.search(r'print$$ json\.dumps\((.*?),\s*indent=2 $$\)', cleaned_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return eval(json_str)  # Safely evaluate the JSON string as a Python dict
        except Exception as e:
            logger.error(f"Failed to evaluate extracted JSON: {str(e)} - Extracted JSON: {json_str}")
    
    logger.error(f"JSONDecodeError: Unable to parse response - Cleaned response: {cleaned_text}")
    return None


@st.cache_data
def generate_ncr_Safety_report_for_veridia(df, report_type, start_date=None, end_date=None, open_until_date=None):
    with st.spinner(f"Generating {report_type} Safety NCR Report with WatsonX..."):
        today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
        closed_start = pd.to_datetime(start_date) if start_date else None
        closed_end = pd.to_datetime(end_date) if end_date else None
        open_until = pd.to_datetime(open_until_date)

        if report_type == "Closed":
            filtered_df = df[
                (df['Discipline'] == 'HSE') &
                (df['Status'] == 'Closed') &
                (df['Days'].notnull()) &
                (df['Days'] > 7)
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
                (pd.to_datetime(df['Created Date (WET)']).notna())
            ].copy()
            filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
            filtered_df = filtered_df[filtered_df['Days_From_Today'] > 7].copy()
            if open_until:
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['Created Date (WET)']) <= open_until)
                ].copy()

        if filtered_df.empty:
            return {"error": f"No {report_type} records found with duration > 7 days"}, ""

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
                    "Tower": "External Development"
                }

                desc_lower = description.lower()
                if any(phrase in desc_lower for phrase in ["veridia clubhouse", "veridia-clubhouse", "veridia club"]):
                    cleaned_record["Tower"] = "Veridia-Club"
                    logger.debug(f"Matched 'Veridia Clubhouse', setting Tower to Veridia-Club")
                else:
                    tower_match = re.search(r"(tower|t)\s*-?\s*(\d+|2021|28)", desc_lower, re.IGNORECASE)
                    cleaned_record["Tower"] = f"Veridia-Tower{tower_match.group(2).zfill(2)}" if tower_match else "Common_Area"
                    logger.debug(f"Tower set to {cleaned_record['Tower']}")

                cleaned_data.append(cleaned_record)

        st.write(f"Total {report_type} records to process: {len(cleaned_data)}")
        logger.debug(f"Processed data: {json.dumps(cleaned_data, indent=2)}")

        if not cleaned_data:
            return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

        access_token = get_access_token(API_KEY)
        if not access_token:
            return {"error": "Failed to obtain access token"}, ""

        result = {"Safety": {"Sites": {}, "Grand_Total": 0}}
        chunk_size = 1
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
                "IMPORTANT: RETURN ONLY A SINGLE VALID JSON OBJECT WITH THE EXACT FIELDS SPECIFIED BELOW. "
                "DO NOT GENERATE ANY CODE (e.g., Python, JavaScript). "
                "DO NOT INCLUDE ANY TEXT, EXPLANATIONS, OR MULTIPLE RESPONSES OUTSIDE THE JSON OBJECT. "
                "DO NOT WRAP THE JSON IN CODE BLOCKS (e.g., ```json). "
                "RETURN THE JSON OBJECT DIRECTLY.\n\n"
                "ONLY PROCESS THE PROVIDED DATA AND RETURN THE RESULT.\n\n"
                "Task: For Safety NCRs, count EVERY record in the provided data by site ('Tower' field) where 'Discipline' is 'HSE' and 'Days' is greater than 7. "
                "The 'Description' MUST be counted if it contains ANY of the following construction safety issues (match these keywords exactly as provided, case-insensitive): "
                "'safety precautions','temporary electricity','on-site labor is working without wearing safety belt','safety norms','Missing Cabin Glass – Tower Crane',"
                "'Crane Operator cabin front glass','site on priority basis lifeline is not fixed at the working place','operated only after Third Party Inspection and certification crane operated without TPIC',"
                "'We have found that safety precautions are not taken seriously at site Tower crane operator cabin front glass is missing while crane operator is working inside cabin.',"
                "'no barrier around','Lock and Key arrangement to restrict unauthorized operations, buzzer while operation, gates at landing platforms, catch net in the vicinity', "
                "'safety precautions are not taken seriously','firecase','Health and Safety Plan','noticed that submission of statistics report is regularly delayed',"
                "'crane operator cabin front glass is missing while crane operator is working inside cabin','labor is working without wearing safety belt', 'barricading', 'tank', 'safety shoes', "
                "'safety belt', 'helmet', 'lifeline', 'guard rails', 'fall protection', 'PPE', 'electrical hazard', 'unsafe platform', 'catch net', 'edge protection', 'TPI', 'scaffold', "
                "'lifting equipment', 'temporary electricity', 'dust suppression', 'debris chute', 'spill control', 'crane operator', 'halogen lamps', 'fall catch net', 'environmental contamination', 'fire hazard'. "
                "Use the 'Tower' values exactly as they appear in the data (e.g., 'Veridia-Club', 'Veridia-Tower01', 'Common_Area'). "
                "Collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', and 'Status' into arrays for each site. "
                "Assign each count to the 'Count' key, representing 'No. of Safety NCRs beyond 7 days'. "
                "If no matches are found for a site, set its count to 0, but ensure all present sites in the data are listed. "
                "EXCLUDE records where 'housekeeping' is the PRIMARY safety concern (e.g., descriptions focusing solely on 'housekeeping' or 'cleaning').\n\n"
                "REQUIRED OUTPUT FORMAT (use this structure with the actual results):\n"
                "{\n"
                '  "Safety": {\n'
                '    "Sites": {\n'
                '      "Site_Name1": {\n'
                '        "Descriptions": ["description1", "description2"],\n'
                '        "Created Date (WET)": ["date1", "date2"],\n'
                '        "Expected Close Date (WET)": ["date1", "date2"],\n'
                '        "Status": ["status1", "status2"],\n'
                '        "Count": number\n'
                '      },\n'
                '      "Site_Name2": {\n'
                '        "Descriptions": ["description1", "description2"],\n'
                '        "Created Date (WET)": ["date1", "date2"],\n'
                '        "Expected Close Date (WET)": ["date1", "date2"],\n'
                '        "Status": ["status1", "status2"],\n'
                '        "Count": number\n'
                '      }\n'
                '    },\n'
                '    "Grand_Total": number\n'
                '  }\n'
                '}\n\n'
                f"IMPORTANT: Ensure the JSON is valid and contains all required fields. "    
                f"Return the result strictly as a JSON object—no code, no explanations, only the JSON.Dont put <|eom_id|> or any other markers in the JSON output."
                f"Data: {json.dumps(chunk)}\n"
            )

            payload = {
                "input": prompt,
                "parameters": {"decoding_method": "greedy", "max_new_tokens": 8100, "min_new_tokens": 0, "temperature": 0.001},
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
                response = session.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=900)
                logger.info(f"WatsonX API response status: {response.status_code}")

                if response.status_code == 200:
                    api_result = response.json()
                    generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                    logger.debug(f"Generated text for chunk {current_chunk}: {generated_text}")

                    if generated_text:
                        # Extract the JSON portion by finding the first complete JSON object
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
                                        logger.warning(f"Invalid site data for {site}: {values}, converting to dict")
                                        values = {
                                            "Count": int(values) if isinstance(values, (int, float)) else 0,
                                            "Descriptions": [],
                                            "Created Date (WET)": [],
                                            "Expected Close Date (WET)": [],
                                            "Status": []
                                        }
                                    
                                    if site not in result["Safety"]["Sites"]:
                                        result["Safety"]["Sites"][site] = {
                                            "Count": 0,
                                            "Descriptions": [],
                                            "Created Date (WET)": [],
                                            "Expected Close Date (WET)": [],
                                            "Status": []
                                        }
                                    
                                    if "Descriptions" in values and values["Descriptions"]:
                                        if not isinstance(values["Descriptions"], list):
                                            values["Descriptions"] = [str(values["Descriptions"])]
                                        result["Safety"]["Sites"][site]["Descriptions"].extend(values["Descriptions"])
                                    
                                    if "Created Date (WET)" in values and values["Created Date (WET)"]:
                                        if not isinstance(values["Created Date (WET)"], list):
                                            values["Created Date (WET)"] = [str(values["Created Date (WET)"])]
                                        result["Safety"]["Sites"][site]["Created Date (WET)"].extend(values["Created Date (WET)"])
                                    
                                    if "Expected Close Date (WET)" in values and values["Expected Close Date (WET)"]:
                                        if not isinstance(values["Expected Close Date (WET)"], list):
                                            values["Expected Close Date (WET)"] = [str(values["Expected Close Date (WET)"])]
                                        result["Safety"]["Sites"][site]["Expected Close Date (WET)"].extend(values["Expected Close Date (WET)"])
                                    
                                    if "Status" in values and values["Status"]:
                                        if not isinstance(values["Status"], list):
                                            values["Status"] = [str(values["Status"])]
                                        result["Safety"]["Sites"][site]["Status"].extend(values["Status"])
                                    
                                    count = values.get("Count", 0)
                                    if not isinstance(count, (int, float)):
                                        count = 0
                                    result["Safety"]["Sites"][site]["Count"] += count
                                
                                result["Safety"]["Grand_Total"] += chunk_grand_total
                                logger.debug(f"Successfully processed chunk {current_chunk}/{total_chunks}")
                            except json.JSONDecodeError as e:
                                logger.error(f"JSONDecodeError for chunk {current_chunk}: {str(e)} - Raw: {json_str}")
                                error_placeholder.error(f"Failed to parse JSON for chunk {current_chunk}: {str(e)}")
                                # Fallback: Manually process the chunk
                                for record in chunk:
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
                                logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
                        else:
                            logger.error(f"No valid JSON found in response for chunk {current_chunk}: {generated_text}")
                            error_placeholder.error(f"No valid JSON found in response for chunk {current_chunk}")
                            # Fallback: Manually process the chunk
                            for record in chunk:
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
                            logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
                    else:
                        logger.error(f"Empty WatsonX response for chunk {current_chunk}")
                        error_placeholder.error(f"Empty WatsonX response for chunk {current_chunk}")
                        # Fallback: Manually process the chunk
                        for record in chunk:
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
                        logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
                else:
                    logger.error(f"WatsonX API error for chunk {current_chunk}: {response.status_code} - {response.text}")
                    error_placeholder.error(f"WatsonX API error for chunk {current_chunk}: {response.status_code} - {response.text}")
                    # Fallback: Manually process the chunk
                    for record in chunk:
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
                    logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
            except requests.exceptions.ReadTimeout as e:
                logger.error(f"ReadTimeoutError after retries for chunk {current_chunk}: {str(e)}")
                error_placeholder.error(f"Failed to connect to WatsonX API for chunk {current_chunk} after retries due to timeout: {str(e)}")
                # Fallback: Manually process the chunk
                for record in chunk:
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
                logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
            except requests.exceptions.RequestException as e:
                logger.error(f"RequestException for chunk {current_chunk}: {str(e)}")
                error_placeholder.error(f"Failed to connect to WatsonX API for chunk {current_chunk}: {str(e)}")
                # Fallback: Manually process the chunk
                for record in chunk:
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
                logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")
            except Exception as e:
                logger.error(f"Unexpected error during WatsonX API call for chunk {current_chunk}: {str(e)}")
                error_placeholder.error(f"Unexpected error during WatsonX API call for chunk {current_chunk}: {str(e)}")
                # Fallback: Manually process the chunk
                for record in chunk:
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
                logger.debug(f"Fallback processed chunk {current_chunk}/{total_chunks}")

        progress_bar.progress(100)
        status_placeholder.write(f"Processed {total_chunks}/{total_chunks} chunks (100%)")
        logger.debug(f"Final result before deduplication: {json.dumps(result, indent=2)}")

        for site in result["Safety"]["Sites"]:
            if "Descriptions" in result["Safety"]["Sites"][site]:
                result["Safety"]["Sites"][site]["Descriptions"] = list(result["Safety"]["Sites"][site]["Descriptions"])
            if "Created Date (WET)" in result["Safety"]["Sites"][site]:
                result["Safety"]["Sites"][site]["Created Date (WET)"] = list(result["Safety"]["Sites"][site]["Created Date (WET)"])
            if "Expected Close Date (WET)" in result["Safety"]["Sites"][site]:
                result["Safety"]["Sites"][site]["Expected Close Date (WET)"] = list(result["Safety"]["Sites"][site]["Expected Close Date (WET)"])
            if "Status" in result["Safety"]["Sites"][site]:
                result["Safety"]["Sites"][site]["Status"] = list(result["Safety"]["Sites"][site]["Status"])
        
        logger.debug(f"Final result after deduplication: {json.dumps(result, indent=2)}")
        return result, json.dumps(result)
    
@st.cache_data
def generate_consolidated_ncr_OpenClose_excel_for_veridia(combined_result, report_title="NCR"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formatting styles
        title_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': 'yellow', 'border': 1, 'font_size': 12
        })
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        subheader_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1
        })
        cell_format_base = {
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        }
        site_format_base = {
            'align': 'left', 'valign': 'vcenter', 'border': 1
        }
        
        # Watercolor-style colors for each tower
        tower_colors = {
            "Veridia-Tower 02": '#F9E79F',  # Soft yellow
            "Veridia-Tower 03": '#A3CFFA',  # Blue
            "Veridia-Tower 04": '#F5C3C2',  # Pink
            "Veridia-Tower 05": '#C4E4B7',  # Green
            "Veridia-Tower 06": '#F5E8B7',  # Soft yellow
            "Veridia-Tower 07": '#D7B9F5',  # Soft lavender
            "Veridia-Club": '#E3B5A4',      # Peach
            "Veridia-Commercial": '#B5EAD7', # Mint
            "External Development area": '#FFD1DC', # Rose
            "Common_Area": '#C7CEEA'         # Periwinkle
        }
        
        # Create format dictionaries for each tower
        tower_formats = {}
        for site, color in tower_colors.items():
            tower_formats[site] = {
                'tower_total': workbook.add_format({
                    'bold': True, 'align': 'left', 'valign': 'vcenter', 'border': 1, 'fg_color': color
                }),
                'site': workbook.add_format({
                    'align': 'left', 'valign': 'vcenter', 'border': 1, 'fg_color': '#FFFFFF'
                }),
                'cell': workbook.add_format({
                    'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True, 'fg_color': '#FFFFFF'
                })
            }
        
        default_cell_format = workbook.add_format(cell_format_base)
        default_site_format = workbook.add_format(site_format_base)
        
        # Generate date string
        now = datetime.now()
        day = now.strftime("%d")
        month_name = now.strftime("%B")
        year = now.strftime("%Y")
        date_part = f"{day}_{month_name}_{year}"
        
        def truncate_sheet_name(base_name, max_length=31):
            if len(base_name) > max_length:
                return base_name[:max_length - 3] + "..."
            return base_name

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
        
        standard_sites = [
            "Veridia-Club","Veridia-Tower 02", "Veridia-Tower 03", "Veridia-Tower 04", "Veridia-Tower 05",
            "Veridia-Tower 06", "Veridia-Tower 07", "Veridia-Commercial", "External Development area", "Common_Area"
        ]
        
        def normalize_site_name(site):
            if site in ["Veridia-Club", "Veridia-Commercial", "External Development area", "Common_Area"]:
                return site
            match = re.search(r'(?:tower|t)[- ]?(\d+)', site, re.IGNORECASE)
            if match:
                num = match.group(1).zfill(2)
                if "CommonArea" in site or "Common Area" in site:
                    return f"Veridia-Tower {num} CommonArea"
                return f"Veridia-Tower {num}"
            return site

        site_mapping = {k: normalize_site_name(k) for k in (resolved_sites.keys() | open_sites.keys())}
        sorted_sites = sorted(standard_sites, key=lambda x: (x != "Veridia-Club", x))
        
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
            'Finishing': ['FW','Civil Finishing ', 'Finishing'],
            'Structure Works': ['SW', 'Structure', 'Works'],
            'MEP': ['MEP']
        }
        tower_levels = ['M8', 'M7', 'M6', 'M5', 'M4', 'M3', 'M2', 'M1']
        row = 3
        site_totals = {}
        
        for site in sorted_sites:
            original_resolved_key = next((k for k, v in site_mapping.items() if v == site), None)
            original_open_key = next((k for k, v in site_mapping.items() if v == site), None)
            
            # Skip tower-specific CommonAreas as separate sites
            if "CommonArea" in site and site != "Common_Area":
                continue
                
            formats = tower_formats.get(site, {})
            tower_total_format = formats.get('tower_total', workbook.add_format({
                'bold': True, 'align': 'left', 'valign': 'vcenter', 'border': 1, 'fg_color': '#D3D3D3'
            }))
            site_format = formats.get('site', default_site_format)
            cell_format = formats.get('cell', default_cell_format)
            
            resolved_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
            open_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
            
            if "Tower" in site and site != "Common_Area":
                resolved_module_counts = {level: {'Finishing': 0, 'Works': 0, 'MEP': 0} for level in tower_levels}
                open_module_counts = {level: {'Finishing': 0, 'Works': 0, 'MEP': 0} for level in tower_levels}
                resolved_common_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
                open_common_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
                
                # Process tower modules
                if original_resolved_key and original_resolved_key in resolved_sites:
                    disciplines = resolved_sites[original_resolved_key].get("Discipline", [])
                    modules = resolved_sites[original_resolved_key].get("Modules", [])
                    for i, discipline in enumerate(disciplines):
                        module_list = modules[i] if i < len(modules) else []
                        cat = 'Finishing' if discipline == 'FW' else 'Works' if discipline == 'SW' else 'MEP'
                        for module in module_list:
                            if module in tower_levels:
                                resolved_module_counts[module][cat] += 1
                            elif module == "Common":
                                resolved_common_counts[cat] += 1
                
                if original_open_key and original_open_key in open_sites:
                    disciplines = open_sites[original_open_key].get("Discipline", [])
                    modules = open_sites[original_open_key].get("Modules", [])
                    for i, discipline in enumerate(disciplines):
                        module_list = modules[i] if i < len(modules) else []
                        cat = 'Finishing' if discipline == 'FW' else 'Works' if discipline == 'SW' else 'MEP'
                        for module in module_list:
                            if module in tower_levels:
                                open_module_counts[module][cat] += 1
                            elif module == "Common":
                                open_common_counts[cat] += 1
                
                # Process tower-specific CommonArea
                tower_num = site.split("Tower ")[1]
                common_site_key = f"Veridia-Tower-{tower_num}-CommonArea"
                resolved_common = resolved_sites.get(common_site_key, {})
                open_common = open_sites.get(common_site_key, {})
                
                for display_cat, possible_keys in category_map.items():
                    for key in possible_keys:
                        value = resolved_common.get(key, 0)
                        if value > 0:
                            resolved_common_counts[display_cat] += value
                        value = open_common.get(key, 0)
                        if value > 0:
                            open_common_counts[display_cat] += value
                
                # Aggregate module and CommonArea counts
                for cat in categories:
                    resolved_counts[cat] = sum(resolved_module_counts[level][cat] for level in tower_levels) + resolved_common_counts[cat]
                    open_counts[cat] = sum(open_module_counts[level][cat] for level in tower_levels) + open_common_counts[cat]
            
            elif site != "Common_Area":
                if original_resolved_key and original_resolved_key in resolved_sites:
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = resolved_sites[original_resolved_key].get(key, 0)
                            if value > 0:
                                resolved_counts[display_cat] = value
                                break
                if original_open_key and original_open_key in open_sites:
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = open_sites[original_open_key].get(key, 0)
                            if value > 0:
                                open_counts[display_cat] = value
                                break
            
            else:  # Common_Area
                if original_resolved_key and original_resolved_key in resolved_sites:
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = resolved_sites[original_resolved_key].get(key, 0)
                            if value > 0:
                                resolved_counts[display_cat] = value
                                break
                if original_open_key and original_open_key in open_sites:
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = open_sites[original_open_key].get(key, 0)
                            if value > 0:
                                open_counts[display_cat] = value
                                break
            
            site_total = sum(resolved_counts.values()) + sum(open_counts.values())
            
            worksheet.write(row, 0, site, tower_total_format)
            for i, display_cat in enumerate(categories):
                worksheet.write(row, i+1, resolved_counts[display_cat], cell_format)
            for i, display_cat in enumerate(categories):
                worksheet.write(row, i+4, open_counts[display_cat], cell_format)
            worksheet.write(row, 7, site_total, cell_format)
            site_totals[site] = site_total
            row += 1
            
            # Add module rows for towers
            if "Tower" in site and site != "Common_Area":
                for level in tower_levels:
                    level_total = sum(resolved_module_counts[level].values()) + sum(open_module_counts[level].values())
                    # Always write the module row, even if counts are zero
                    worksheet.write(row, 0, level, site_format)
                    for i, display_cat in enumerate(categories):
                        worksheet.write(row, i+1, resolved_module_counts[level][display_cat], cell_format)
                    for i, display_cat in enumerate(categories):
                        worksheet.write(row, i+4, open_module_counts[level][display_cat], cell_format)
                    worksheet.write(row, 7, level_total, cell_format)
                    row += 1
                
                # Add tower-specific CommonArea row
                common_total = sum(resolved_common_counts.values()) + sum(open_common_counts.values())
                if common_total > 0:
                    common_desc = f"Tower {tower_num} Common_Area"
                    worksheet.write(row, 0, common_desc, site_format)
                    for i, display_cat in enumerate(categories):
                        worksheet.write(row, i+1, resolved_common_counts[display_cat], cell_format)
                    for i, display_cat in enumerate(categories):
                        worksheet.write(row, i+4, open_common_counts[display_cat], cell_format)
                    worksheet.write(row, 7, common_total, cell_format)
                    row += 1

        def write_detail_sheet(sheet_name, data, title):
            truncated_sheet_name = truncate_sheet_name(f"{sheet_name} {date_part}")
            detail_worksheet = workbook.add_worksheet(truncated_sheet_name)
            detail_worksheet.set_column('A:A', 20)
            detail_worksheet.set_column('B:B', 60)
            detail_worksheet.set_column('C:D', 20)
            detail_worksheet.set_column('E:E', 15)
            detail_worksheet.set_column('F:F', 15)
            detail_worksheet.merge_range('A1:F1', f"{title} {date_part}", title_format)
            headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline']
            for col, header in enumerate(headers):
                detail_worksheet.write(1, col, header, header_format)
            row = 2
            for site, site_data in data.items():
                normalized_site = site_mapping.get(site, site)
                descriptions = site_data.get("Descriptions", [])
                created_dates = site_data.get("Created Date (WET)", [])
                close_dates = site_data.get("Expected Close Date (WET)", [])
                statuses = site_data.get("Status", [])
                disciplines = site_data.get("Discipline", [])
                max_length = max(len(descriptions), len(created_dates), len(close_dates), len(statuses), len(disciplines))
                for i in range(max_length):
                    detail_worksheet.write(row, 0, normalized_site, default_site_format)
                    detail_worksheet.write(row, 1, descriptions[i] if i < len(descriptions) else "", default_cell_format)
                    detail_worksheet.write(row, 2, created_dates[i] if i < len(created_dates) else "", default_cell_format)
                    detail_worksheet.write(row, 3, close_dates[i] if i < len(close_dates) else "", default_cell_format)
                    detail_worksheet.write(row, 4, statuses[i] if i < len(statuses) else "", default_cell_format)
                    detail_worksheet.write(row, 5, disciplines[i] if i < len(disciplines) else "", default_cell_format)
                    row += 1

        if resolved_sites:
            write_detail_sheet("Closed NCR Details", resolved_sites, "Closed NCR Details")
        if open_sites:
            write_detail_sheet("Open NCR Details", open_sites, "Open NCR Details")

        output.seek(0)
        return output
    
@st.cache_data
def generate_consolidated_ncr_Housekeeping_excel_for_veridia(combined_result, report_title="Housekeeping: Current Month"):
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
            "Veridia-Club", "Veridia-Tower01", "Veridia-Tower02", "Veridia-Tower03", "Veridia-Tower04",
            "Veridia-Tower05", "Veridia-Tower06", "Veridia-Tower07", "Common_Area", "Veridia-Commercial", "External Development"
        ]
        
        def normalize_site_name(site):
            if site in standard_sites:
                return site
            match = re.search(r'(?:tower|t)[- ]?(\d+|2021|28)', site, re.IGNORECASE)
            if match:
                num = match.group(1).zfill(2)
                return f"Veridia-Tower{num}"
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
def generate_consolidated_ncr_Safety_excel_for_veridia(combined_result, report_title=None):
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
            "Veridia-Club", "Veridia-Tower01", "Veridia-Tower02", "Veridia-Tower03", "Veridia-Tower04",
            "Veridia-Tower05", "Veridia-Tower06", "Veridia-Tower07", "Common_Area", "Veridia-Commercial", "External Development"
        ]
        
        def normalize_site_name(site):
            if site in standard_sites:
                return site
            match = re.search(r'(?:tower|t)[- ]?(\d+|2021|28)', site, re.IGNORECASE)
            if match:
                num = match.group(1).zfill(2)
                return f"Veridia-Tower{num}"
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

def generate_combined_excel_report_for_veridia(all_reports, filename_prefix="All_Reports"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formatting
        title_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': 'yellow', 'border': 1, 'font_size': 12
        })
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        subheader_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1
        })
        cell_format_base = {
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        }
        site_format_base = {
            'align': 'left', 'valign': 'vcenter', 'border': 1
        }
        
        # Watercolor-style colors for Combined NCR
        tower_colors = {
            "Veridia-Tower 02": '#C4E4B7',  # Green
            "Veridia-Tower 03": '#A3CFFA',  # Blue
            "Veridia-Tower 04": '#F5C3C2',  # Pink
            "Veridia-Tower 05": '#C4E4B7',  # Green
            "Veridia-Tower 06": '#F5E8B7',  # Soft yellow
            "Veridia-Tower 07": '#D7B9F5',  # Soft lavender
            "Veridia-Club": '#E3B5A4',      # Peach
            "Veridia-Commercial": '#B5EAD7', # Mint
            "External Development area": '#FFD1DC', # Rose
            "Common_Area": '#C7CEEA'         # Periwinkle
        }
        
        tower_formats = {}
        for site, color in tower_colors.items():
            tower_formats[site] = {
                'tower_total': workbook.add_format({
                    'bold': True, 'align': 'left', 'valign': 'vcenter', 'border': 1, 'fg_color': color
                }),
                'site': workbook.add_format({
                    'align': 'left', 'valign': 'vcenter', 'border': 1, 'fg_color': '#FFFFFF'
                }),
                'cell': workbook.add_format({
                    'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True, 'fg_color': '#FFFFFF'
                })
            }
        
        default_cell_format = workbook.add_format(cell_format_base)
        default_site_format = workbook.add_format(site_format_base)
        
        # Formatting for Safety/Housekeeping
        description_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        
        # Date handling
        now = datetime.now()  # Current date: May 29, 2025
        day = now.strftime("%d")
        month_name = now.strftime("%B")
        year = now.strftime("%Y")
        date_part = f"{day}_{month_name}_{year}"  # e.g., "29_May_2025"
        
        def truncate_sheet_name(base_name, max_length=31):
            if len(base_name) > max_length:
                return base_name[:max_length - 3] + "..."
            return base_name

        # 1. Combined NCR Report
        combined_result = all_reports.get("Combined_NCR", {})
        report_title_ncr = f"NCR: {date_part}"
        
        worksheet = workbook.add_worksheet('NCR Report')
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:H', 12)
        
        resolved_data = combined_result.get("NCR resolved beyond 21 days", {})
        open_data = combined_result.get("NCR open beyond 21 days", {})
        if not isinstance(resolved_data, dict) or "error" in resolved_data:
            resolved_data = {"Sites": {}}
        if not isinstance(open_data, dict) or "error" in open_data:
            open_data = {"Sites": {}}
            
        resolved_sites = resolved_data.get("Sites", {})
        open_sites = open_data.get("Sites", {})
        
        standard_sites = [
            "Veridia-Club", "Veridia-Tower 02", "Veridia-Tower 03", "Veridia-Tower 04", "Veridia-Tower 05",
            "Veridia-Tower 06", "Veridia-Tower 07", "Veridia-Commercial", "External Development area", "Common_Area"
        ]
        
        def normalize_site_name(site):
            if site in ["Veridia-Club", "Veridia-Commercial", "External Development area", "Common_Area"]:
                return site
            match = re.search(r'(?:tower|t)[- ]?(\d+)', site, re.IGNORECASE)
            if match:
                num = match.group(1).zfill(2)
                if "CommonArea" in site or "Common Area" in site:
                    return f"Veridia-Tower {num} CommonArea"
                return f"Veridia-Tower {num}"
            return site

        site_mapping = {k: normalize_site_name(k) for k in (resolved_sites.keys() | open_sites.keys())}
        sorted_sites = sorted(standard_sites, key=lambda x: (x != "Veridia-Club", x))
        
        worksheet.merge_range('A1:H1', report_title_ncr, title_format)
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
        tower_levels = ['M8', 'M7', 'M6', 'M5', 'M4', 'M3', 'M2', 'M1']
        row = 3
        site_totals = {}
        
        for site in sorted_sites:
            original_resolved_key = next((k for k, v in site_mapping.items() if v == site), None)
            original_open_key = next((k for k, v in site_mapping.items() if v == site), None)
            
            if "CommonArea" in site and site != "Common_Area":
                continue
                
            formats = tower_formats.get(site, {})
            tower_total_format = formats.get('tower_total', workbook.add_format({
                'bold': True, 'align': 'left', 'valign': 'vcenter', 'border': 1, 'fg_color': '#D3D3D3'
            }))
            site_format = formats.get('site', default_site_format)
            cell_format = formats.get('cell', default_cell_format)
            
            resolved_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
            open_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
            
            if "Tower" in site and site != "Common_Area":
                resolved_module_counts = {level: {'Finishing': 0, 'Works': 0, 'MEP': 0} for level in tower_levels}
                open_module_counts = {level: {'Finishing': 0, 'Works': 0, 'MEP': 0} for level in tower_levels}
                resolved_common_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
                open_common_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
                
                if original_resolved_key and original_resolved_key in resolved_sites:
                    disciplines = resolved_sites[original_resolved_key].get("Discipline", [])
                    modules = resolved_sites[original_resolved_key].get("Modules", [])
                    for i, discipline in enumerate(disciplines):
                        module_list = modules[i] if i < len(modules) else []
                        cat = 'Finishing' if discipline == 'FW' else 'Works' if discipline == 'SW' else 'MEP'
                        for module in module_list:
                            if module in tower_levels:
                                resolved_module_counts[module][cat] += 1
                            elif module == "Common":
                                resolved_common_counts[cat] += 1
                
                if original_open_key and original_open_key in open_sites:
                    disciplines = open_sites[original_open_key].get("Discipline", [])
                    modules = open_sites[original_open_key].get("Modules", [])
                    for i, discipline in enumerate(disciplines):
                        module_list = modules[i] if i < len(modules) else []
                        cat = 'Finishing' if discipline == 'FW' else 'Works' if discipline == 'SW' else 'MEP'
                        for module in module_list:
                            if module in tower_levels:
                                open_module_counts[module][cat] += 1
                            elif module == "Common":
                                open_common_counts[cat] += 1
                
                # Process tower-specific CommonArea
                tower_num = site.split("Tower ")[1]
                common_site_key = f"Veridia-Tower-{tower_num}-CommonArea"
                resolved_common = resolved_sites.get(common_site_key, {})
                open_common = open_sites.get(common_site_key, {})
                
                for display_cat, possible_keys in category_map.items():
                    for key in possible_keys:
                        value = resolved_common.get(key, 0)
                        if value > 0:
                            resolved_common_counts[display_cat] += value
                        value = open_common.get(key, 0)
                        if value > 0:
                            open_common_counts[display_cat] += value
                
                # Aggregate module and CommonArea counts
                for cat in categories:
                    resolved_counts[cat] = sum(resolved_module_counts[level][cat] for level in tower_levels) + resolved_common_counts[cat]
                    open_counts[cat] = sum(open_module_counts[level][cat] for level in tower_levels) + open_common_counts[cat]
            
            elif site != "Common_Area":
                if original_resolved_key and original_resolved_key in resolved_sites:
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = resolved_sites[original_resolved_key].get(key, 0)
                            if value > 0:
                                resolved_counts[display_cat] = value
                                break
                if original_open_key and original_open_key in open_sites:
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = open_sites[original_open_key].get(key, 0)
                            if value > 0:
                                open_counts[display_cat] = value
                                break
            
            else:  # Common_Area
                if original_resolved_key and original_resolved_key in resolved_sites:
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = resolved_sites[original_resolved_key].get(key, 0)
                            if value > 0:
                                resolved_counts[display_cat] = value
                                break
                if original_open_key and original_open_key in open_sites:
                    for display_cat, possible_keys in category_map.items():
                        for key in possible_keys:
                            value = open_sites[original_open_key].get(key, 0)
                            if value > 0:
                                open_counts[display_cat] = value
                                break
            
            site_total = sum(resolved_counts.values()) + sum(open_counts.values())
            
            worksheet.write(row, 0, site, tower_total_format)
            for i, display_cat in enumerate(categories):
                worksheet.write(row, i+1, resolved_counts[display_cat], cell_format)
            for i, display_cat in enumerate(categories):
                worksheet.write(row, i+4, open_counts[display_cat], cell_format)
            worksheet.write(row, 7, site_total, cell_format)
            site_totals[site] = site_total
            row += 1
            
            if "Tower" in site and site != "Common_Area":
                for level in tower_levels:
                    # Calculate level-specific total correctly
                    level_total = 0
                    worksheet.write(row, 0, level, site_format)
                    for i, display_cat in enumerate(categories):
                        resolved_value = resolved_module_counts[level][display_cat]
                        worksheet.write(row, i+1, resolved_value, cell_format)
                        level_total += resolved_value
                    for i, display_cat in enumerate(categories):
                        open_value = open_module_counts[level][display_cat]
                        worksheet.write(row, i+4, open_value, cell_format)
                        level_total += open_value
                    worksheet.write(row, 7, level_total, cell_format)
                    row += 1
                
                # Add tower-specific CommonArea row
                common_total = 0
                common_desc = f"Tower {tower_num} Common_Area"
                worksheet.write(row, 0, common_desc, site_format)
                for i, display_cat in enumerate(categories):
                    resolved_value = resolved_common_counts[display_cat]
                    worksheet.write(row, i+1, resolved_value, cell_format)
                    common_total += resolved_value
                for i, display_cat in enumerate(categories):
                    open_value = open_common_counts[display_cat]
                    worksheet.write(row, i+4, open_value, cell_format)
                    common_total += open_value
                worksheet.write(row, 7, common_total, cell_format)
                row += 1

        # Combined NCR Detail Sheets
        def write_detail_sheet(sheet_name, data, title):
            truncated_sheet_name = truncate_sheet_name(f"{sheet_name} {date_part}")
            detail_worksheet = workbook.add_worksheet(truncated_sheet_name)
            detail_worksheet.set_column('A:A', 20)
            detail_worksheet.set_column('B:B', 60)
            detail_worksheet.set_column('C:D', 20)
            detail_worksheet.set_column('E:E', 15)
            detail_worksheet.set_column('F:F', 15)
            detail_worksheet.merge_range('A1:F1', f"{title} {date_part}", title_format)
            headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline']
            for col, header in enumerate(headers):
                detail_worksheet.write(1, col, header, header_format)
            row = 2
            for site, site_data in data.items():
                normalized_site = site_mapping.get(site, site)
                descriptions = site_data.get("Descriptions", [])
                created_dates = site_data.get("Created Date (WET)", [])
                close_dates = site_data.get("Expected Close Date (WET)", [])
                statuses = site_data.get("Status", [])
                disciplines = site_data.get("Discipline", [])
                max_length = max(len(descriptions), len(created_dates), len(close_dates), len(statuses), len(disciplines))
                for i in range(max_length):
                    detail_worksheet.write(row, 0, normalized_site, default_site_format)
                    detail_worksheet.write(row, 1, descriptions[i] if i < len(descriptions) else "", description_format)
                    detail_worksheet.write(row, 2, created_dates[i] if i < len(created_dates) else "", default_cell_format)
                    detail_worksheet.write(row, 3, close_dates[i] if i < len(close_dates) else "", default_cell_format)
                    detail_worksheet.write(row, 4, statuses[i] if i < len(statuses) else "", default_cell_format)
                    detail_worksheet.write(row, 5, disciplines[i] if i < len(disciplines) else "", default_cell_format)
                    row += 1

        if resolved_sites:
            write_detail_sheet("Closed NCR Details", resolved_sites, "Closed NCR Details")
        if open_sites:
            write_detail_sheet("Open NCR Details", open_sites, "Open NCR Details")

        # 2. Safety and Housekeeping Reports
        def write_safety_housekeeping_report(report_type, data, report_title, sheet_type):
            worksheet = workbook.add_worksheet(truncate_sheet_name(f'{report_type} NCR {sheet_type} {date_part}'))
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:B', 15)
            worksheet.merge_range('A1:B1', f"{report_title} - {sheet_type}", title_format)
            row = 1
            worksheet.write(row, 0, 'Site', header_format)
            worksheet.write(row, 1, f'No. of {report_type} NCRs beyond 7 days', header_format)
            sites_data = data.get(report_type, {}).get("Sites", {})
            site_mapping = {k: normalize_site_name(k) for k in sites_data.keys()}
            row = 2
            for site in sorted_sites:
                worksheet.write(row, 0, site, default_site_format)
                original_key = next((k for k, v in site_mapping.items() if v == site), None)
                value = sites_data[original_key].get("Count", 0) if original_key and original_key in sites_data else 0
                worksheet.write(row, 1, value, default_cell_format)
                row += 1
            worksheet_details = workbook.add_worksheet(truncate_sheet_name(f'{report_type} NCR {sheet_type} Details {date_part}'))
            worksheet_details.set_column('A:A', 20)
            worksheet_details.set_column('B:B', 60)
            worksheet_details.set_column('C:D', 20)
            worksheet_details.set_column('E:E', 15)
            worksheet_details.set_column('F:F', 15)
            worksheet_details.merge_range('A1:F1', f"{report_title} - {sheet_type} Details", title_format)
            headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline']
            row = 1
            for col, header in enumerate(headers):
                worksheet_details.write(row, col, header, header_format)
            row = 2
            for site in sorted_sites:
                original_key = next((k for k, v in site_mapping.items() if v == site), None)
                if original_key and original_key in sites_data:
                    site_data = sites_data[original_key]
                    descriptions = site_data.get("Descriptions", [])
                    created_dates = site_data.get("Created Date (WET)", [])
                    close_dates = site_data.get("Expected Close Date (WET)", [])
                    statuses = site_data.get("Status", [])
                    max_length = max(len(descriptions), len(created_dates), len(close_dates), len(statuses))
                    for i in range(max_length):
                        worksheet_details.write(row, 0, site, default_site_format)
                        worksheet_details.write(row, 1, descriptions[i] if i < len(descriptions) else "", description_format)
                        worksheet_details.write(row, 2, created_dates[i] if i < len(created_dates) else "", default_cell_format)
                        worksheet_details.write(row, 3, close_dates[i] if i < len(close_dates) else "", default_cell_format)
                        worksheet_details.write(row, 4, statuses[i] if i < len(statuses) else "", default_cell_format)
                        worksheet_details.write(row, 5, "HSE", default_cell_format)
                        row += 1

        safety_closed_data = all_reports.get("Safety_NCR_Closed", {})
        report_title_safety = f"Safety NCR: {date_part}"
        write_safety_housekeeping_report("Safety", safety_closed_data, report_title_safety, "Closed")
        safety_open_data = all_reports.get("Safety_NCR_Open", {})
        write_safety_housekeeping_report("Safety", safety_open_data, report_title_safety, "Open")
        housekeeping_closed_data = all_reports.get("Housekeeping_NCR_Closed", {})
        report_title_housekeeping = f"Housekeeping NCR: {date_part}"
        write_safety_housekeeping_report("Housekeeping", housekeeping_closed_data, report_title_housekeeping, "Closed")
        housekeeping_open_data = all_reports.get("Housekeeping_NCR_Open", {})
        write_safety_housekeeping_report("Housekeeping", housekeeping_open_data, report_title_housekeeping, "Open")

    output.seek(0)
    return output

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

        
