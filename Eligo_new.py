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
    st.error("❌ Required environment variables (API_KEY, WATSONX_API_URL, MODEL_ID, PROJECT_ID) missing!")
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


@st.cache_data
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type((requests.RequestException, ValueError, KeyError)))
def generate_ncr_report_for_eligo(df: pd.DataFrame, report_type: str, start_date=None, end_date=None, Until_Date=None) -> Tuple[Dict[str, Any], str]:
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
            unique_records = set()  # Use set to track unique records

            def extract_modules_from_description(description):
                """Extract module numbers from description, prioritizing modules over common areas."""
                description_lower = description.lower()
                modules = set()

                # Pattern 1: Handle ranges like "Module 1 to 3" or "Module 1-3"
                range_patterns = r"(?:module|mod|m)[-\s]*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))\s*(?:to|-|–)\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))"
                for start_str, end_str in re.findall(range_patterns, description_lower, re.IGNORECASE):
                    try:
                        start, end = int(start_str), int(end_str)
                        if 0 < start <= end <= 50:
                            modules.update(f"Module {i}" for i in range(start, end + 1))
                    except ValueError:
                        continue

                # Pattern 2: Handle combinations like "Module 1 & 2" or "Module – 6 & 7"
                combination_patterns = [
                    r"module[-\s]*(?:–|-)?\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))\s*[&and]+\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))",
                    r"module[-\s]*(?:–|-)?\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))\s*[,&]\s*(\d+)(?!\s*(?:mm|th|rd|nd|st|floor))",
                ]
                for pattern in combination_patterns:
                    combo_matches = re.findall(pattern, description_lower, re.IGNORECASE)
                    for match in combo_matches:
                        try:
                            num1 = int(match[0])
                            num2 = int(match[1])
                            if 1 <= num1 <= 50:
                                modules.add(f"Module {num1}")
                            if 1 <= num2 <= 50:
                                modules.add(f"Module {num2}")
                        except ValueError:
                            continue

                # Pattern 3:  Handle grouped modules like "Module 2&3", "M-7 & 6", "Mod 1,2,3"
                list_pattern = r"(?:module|mod|m)[-\s]*((?:\d+\s*(?:,|&|and)?\s*)+)(?!\s*(?:mm|th|rd|nd|st|floor))"
                for match in re.findall(list_pattern, description_lower, re.IGNORECASE):
                    for num in re.findall(r"\b\d{1,2}\b(?!\s*(?:mm|th|rd|nd|st|floor))", match, re.IGNORECASE):
                        try:
                            num = int(num)
                            if 0 < num <= 50:
                                modules.add(f"Module {num}")
                        except ValueError:
                            continue

                # Pattern 4:# --- 3️⃣ Handle single modules like "Module 3", "M-6", etc.

                individual_patterns = r"(?:module|mod|m)[-\s]*(\d{1,2})(?!\s*(?:mm|th|rd|nd|st|floor))"
                for num_str in re.findall(individual_patterns, description_lower, re.IGNORECASE):
                    try:
                        num = int(num_str)
                        if 0 < num <= 50:
                            modules.add(f"Module {num}")
                    except ValueError:
                            continue

                # Handle corridor with modules
                if "corridor" in description_lower and modules:
                    return sorted(list(modules))

                # FIXED: Check for common areas only if NO modules found AND has specific common indicators
                if not modules:
                    # More specific common area patterns that truly indicate common areas
                    specific_common_patterns = [
                        r"steel\s+yard", r"qc\s+lab", r"cipl", r"nta\s+beam",
                        r"non\s+tower", r"foundation\s+level(?!\s+flat)",
                    ]
                    
                    # Check for truly common areas (not tower-specific common areas)
                    for pattern in specific_common_patterns:
                        if re.search(pattern, description_lower, re.IGNORECASE):
                            return ["Common"]
                    
                    # For tower-specific descriptions without clear module numbers,
                    # try to infer from context but be more conservative
                    if any(word in description_lower for word in ["housekeeping", "steel scrap", "first floor level"]):
                        # These could be tower-specific issues, return empty to let tower assignment handle it
                        return ["Common"]  # Will be handled by tower assignment logic
                
                return sorted(list(modules)) if modules else ["Common"]

            def determine_tower_assignment(description):
                """Assign tower based on description, prioritizing explicit tower mentions."""
                description_lower = description.lower()
                if any(phrase in description_lower for phrase in ["eligo clubhouse", "eligo-clubhouse", "eligo club"]):
                    return "Eligo-Club"

                # Tower patterns - Added pattern to catch "Tower (F)" with brackets
                tower_matches = re.findall(r"\b(?:tower|t)\s*[-\s(]*([fgh])\b", description_lower, re.IGNORECASE)
                tower_bracket_matches = re.findall(r"tower\s*\(\s*([fgh])\s*\)", description_lower, re.IGNORECASE)
                    
                # Combine both patterns
                all_tower_matches = tower_matches + tower_bracket_matches
                    
                multiple_tower_pattern = re.search(
                    r"\btower\s*[-\s(]*([a-h])\b\s*(?:,|&|and)\s*(?:tower\s*[-\s(]*)?([a-h])\b",
                    description_lower, re.IGNORECASE
                )
                    
                # Check for specific module mentions
                has_module = re.search(r"module\s*[-\s]*\d+", description_lower, re.IGNORECASE)
                flat_no_pattern = re.search(r"flat\s*no", description_lower, re.IGNORECASE)
                unit_pattern = re.search(r"unit\s*\d+", description_lower, re.IGNORECASE)
                    
                # Common area indicators (areas truly common to all towers)
                general_common_indicators = [
                    "steel yard", "qc lab", "cipl", "nta beam", "non tower"
                ]
                    
                # Tower-specific common area indicators
                tower_common_indicators = [
                    "lift lobby", "corridor", "staircase"
                ]
                    
                # Structural elements that belong to specific tower (not common areas)
                tower_structural_elements = [
                    "lift wall", "shear wall", "beam", "column", "slab", "foundation"
                ]
                    
                # Check if it's a general common area (no tower assignment)
                is_general_common = any(indicator in description_lower for indicator in general_common_indicators)
                is_tower_common = any(indicator in description_lower for indicator in tower_common_indicators)
                is_structural_element = any(element in description_lower for element in tower_structural_elements)

                # If it's a general common area and no specific tower is mentioned, return Common_Area
                if is_general_common and not all_tower_matches:
                    return "Common_Area"

                # Handle multiple tower assignments
                if multiple_tower_pattern:
                    tower1 = multiple_tower_pattern.group(1).upper()
                    tower2 = multiple_tower_pattern.group(2).upper() if multiple_tower_pattern.group(2) else None
                    if tower2 and tower1 != tower2:
                        # For structural elements or specific modules/flats, assign to tower directly
                        if is_structural_element or has_module or flat_no_pattern or unit_pattern:
                            return (f"Eligo-Tower-{tower1}", f"Eligo-Tower-{tower2}")
                        elif is_tower_common:
                            return (f"Eligo-Tower-{tower1}-CommonArea", f"Eligo-Tower-{tower2}-CommonArea")
                        else:
                            return (f"Eligo-Tower-{tower1}", f"Eligo-Tower-{tower2}")
                    else:
                        # Single tower from multiple pattern
                        if is_structural_element or has_module or flat_no_pattern or unit_pattern:
                            return f"Eligo-Tower-{tower1}"
                        elif is_tower_common:
                            return f"Eligo-Tower-{tower1}-CommonArea"
                        else:
                            return f"Eligo-Tower-{tower1}"

                # Handle single tower assignments
                elif all_tower_matches:
                    tower_letter = all_tower_matches[0].upper()
                    # PRIORITY 1: If it's a structural element, assign to tower directly
                    if is_structural_element:
                        return f"Eligo-Tower-{tower_letter}"
                    # PRIORITY 2: If tower is mentioned with module/flat/unit, it's tower-specific, not common area
                    elif has_module or flat_no_pattern or unit_pattern:
                        return f"Eligo-Tower-{tower_letter}"
                    # PRIORITY 3: Only assign to CommonArea if it's truly common area indicators AND no structural elements
                    elif is_tower_common and not is_structural_element:
                        return f"Eligo-Tower-{tower_letter}-CommonArea"
                    # PRIORITY 4: Default to tower-specific for any tower mention
                    else:
                        return f"Eligo-Tower-{tower_letter}"
                    
                # If no tower is mentioned, it's a general common area
                else:
                    return "Common_Area"

            def process_chunk_locally(chunk, all_results, report_type):
                """Process a chunk of data locally, grouping by Tower and calculating metrics."""
                try:
                    # Define standard sites
                    standard_sites = {
                        "Tower F": ["F1", "F2", "Common Description"],
                        "Tower G": ["G1", "G2", "G3", "Common Description"],
                        "Tower H": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "Common Description"]
                    }

                    for record in chunk:
                        site = record["Tower"]
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
                        
                        site_data = all_results[report_type]["Sites"][site]
                        site_data["Descriptions"].append(record["Description"])
                        site_data["Created Date (WET)"].append(record["Created Date (WET)"])
                        site_data["Expected Close Date (WET)"].append(record.get("Expected Close Date (WET)", ""))
                        site_data["Status"].append(record["Status"])
                        site_data["Discipline"].append(record["Discipline"])
                        
                        # Convert modules to tower-specific format based on standard sites
                        tower_letter = site.split("-")[-1][0] if "Tower" in site and not site.endswith("CommonArea") else None
                        modules = record["Modules"]
                        formatted_modules = []
                        if tower_letter:
                            valid_modules = standard_sites.get(f"Tower {tower_letter}", [])
                            formatted_modules = [
                                f"{tower_letter}{mod.split()[-1]}" if mod != "Common" and f"{tower_letter}{mod.split()[-1]}" in valid_modules
                                else "Common Description" if mod == "Common" and "Common Description" in valid_modules
                                else mod
                                for mod in modules
                            ]
                        else:
                            formatted_modules = modules
                        site_data["Modules"].append(formatted_modules)
                        
                        # Update discipline counts
                        disc_cat = record["Discipline_Category"]
                        if disc_cat in ["SW", "FW", "MEP"]:
                            site_data[disc_cat] += 1
                        site_data["Total"] += 1
                        
                        # Update module counts
                        for mod in formatted_modules:
                            site_data["ModulesCount"][mod] = site_data["ModulesCount"].get(mod, 0) + 1
                    
                    all_results[report_type]["Grand_Total"] += len(chunk)
                except Exception as e:
                    logger.error(f"Error in process_chunk_locally: {str(e)}")

            # Process each record with unique identification
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
                    
                    # CREATE UNIQUE IDENTIFIER FOR EACH RECORD
                    unique_id = f"{description}_{cleaned_record['Created Date (WET)']}_{cleaned_record['Status']}"
                    
                    # # Skip if already processed
                    if unique_id in unique_records:
                        logger.debug(f"Skipping duplicate record: {unique_id}")
                        continue
                    
                    unique_records.add(unique_id)
                    
                    # Extract modules
                    modules = extract_modules_from_description(description)
                    cleaned_record["Modules"] = modules
                    
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

                    
                    # Determine tower assignment
                    tower_assignment = determine_tower_assignment(description)
                    
                    # Handle multiple tower assignments
                    if isinstance(tower_assignment, tuple):
                        for tower in tower_assignment:
                            record_copy = cleaned_record.copy()
                            record_copy["Tower"] = tower
                            cleaned_data.append(record_copy)
                    else:
                        cleaned_record["Tower"] = tower_assignment
                        cleaned_data.append(cleaned_record)
                        
                except Exception as e:
                    logger.error(f"Error processing record: {record}, error: {str(e)}")
                    continue

            if not cleaned_data:
                return {report_type: {"Sites": {}, "Grand_Total": 0}}, ""

            # Log the total records being processed
            st.write(f"Total unique records to process: {len(cleaned_data)}")
            logger.info(f"Processing {len(cleaned_data)} unique records for {report_type} report")

            # Get access token
            try:
                access_token = get_access_token("IS5GyEBD3wWrNYG_eF57TBL-fW1KNdskezaQKPbA7Kxm")
                if not access_token:
                    return {"error": "Failed to obtain access token"}, ""
            except Exception as e:
                logger.error(f"Error getting access token: {str(e)}")
                return {"error": f"Failed to obtain access token: {str(e)}"}, ""

            all_results = {report_type: {"Sites": {}, "Grand_Total": 0}}
            chunk_size = int(os.getenv("CHUNK_SIZE", 15))

            for i in range(0, len(cleaned_data), chunk_size):
                chunk = cleaned_data[i:i + chunk_size]
                chunk_num = i // chunk_size + 1
                
                # Capture and log start time
                start_time = datetime.now()
                st.write(f" 🔄 Processing chunk {chunk_num}: Records {i} to {min(i + chunk_size, len(cleaned_data))} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} (Total: {len(chunk)} records)")
                logger.info(f"Started chunk {chunk_num} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Log data sent to WatsonX
                logger.info(f"Data sent to WatsonX for {report_type} chunk {chunk_num}: {json.dumps(chunk, indent=2)}")

                # Log the total number of records being processed
                total_records = len(cleaned_data)
                st.write(f"Total {report_type} records to process: {total_records}")
                logger.info(f"Total {report_type} records to process: {total_records}")

                # Enhanced prompt with better validation
                prompt = (
                    "CRITICAL INSTRUCTIONS - READ CAREFULLY:\n"
                    "1. RETURN ONLY A SINGLE VALID JSON OBJECT\n"
                    "2. DO NOT GENERATE CODE, EXPLANATIONS, OR TEXT OUTSIDE JSON\n"
                    "3. DO NOT WRAP JSON IN CODE BLOCKS\n"
                    "4. PROCESS EVERY SINGLE RECORD PROVIDED\n"
                    "5. MAINTAIN EXACT COUNT CONSISTENCY\n\n"
                    
                    f"Task: Process exactly {len(chunk)} records. Group by 'Tower' field and collect all data into arrays. "
                    f"Count records by 'Discipline_Category' (SW/FW/MEP). Convert Module format: 'Module 1' → tower-specific (e.g., F1, G1, H1). "
                    f"Keep 'Common Description' modules as 'Common Description'.\n\n"
                    
                    "REQUIRED JSON STRUCTURE:\n"
                    "{\n"
                    f"  \"{report_type}\": {{\n"
                    '    "Sites": {\n'
                    '      "Tower_Name": {\n'
                    '        "Descriptions": ["desc1", "desc2"],\n'
                    '        "Created Date (WET)": ["date1", "date2"],\n'
                    '        "Expected Close Date (WET)": ["date1", "date2"],\n'
                    '        "Status": ["status1", "status2"],\n'
                    '        "Discipline": ["disc1", "disc2"],\n'
                    '        "Modules": [["H1"], ["H2", "H3"]],\n'
                    '        "SW": count,\n'
                    '        "FW": count,\n'
                    '        "MEP": count,\n'
                    '        "Total": count,\n'
                    '        "ModulesCount": {"H1": count1, "H2": count2}\n'
                    '      }\n'
                    '    },\n'
                    f'    "Grand_Total": {len(chunk)}\n'
                    '  }\n'
                    '}\n\n'
                    
                    f"VALIDATION REQUIREMENTS:\n"
                    f"- Grand_Total must equal {len(chunk)}\n"
                    f"- Sum of all Tower Totals must equal {len(chunk)}\n"
                    f"- Each array must have same length as Tower Total\n"
                    f"- No empty towers/sites in output\n\n"
                    
                    f"Data to process ({len(chunk)} records):\n{json.dumps(chunk, indent=2)}\n\n"
                    
                    f"RETURN ONLY THE JSON OBJECT. NO OTHER TEXT."
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
                    
                    if response.status_code == 200:
                        api_result = response.json()
                        generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                        
                        # Log response for debugging
                        logger.debug(f"API Response for chunk {chunk_num}: {generated_text[:500]}...")
                        
                        parsed_json = clean_and_parse_json(generated_text)
                        
                        if parsed_json and report_type in parsed_json:
                            chunk_result = parsed_json[report_type]
                            
                            # VALIDATION: Check if counts match
                            api_grand_total = chunk_result.get("Grand_Total", 0)
                            expected_total = len(chunk)
                            
                            if api_grand_total != expected_total:
                                logger.warning(f"Count mismatch in chunk {chunk_num}: API returned {api_grand_total}, expected {expected_total}")
                                st.warning(f"Count mismatch in chunk {chunk_num}: API returned {api_grand_total}, expected {expected_total}. Using local processing.")
                                process_chunk_locally(chunk, all_results, report_type)
                            else:
                                # Validate individual tower counts
                                total_records_in_sites = sum(site_data.get("Total", 0) for site_data in chunk_result["Sites"].values())
                                
                                if total_records_in_sites != expected_total:
                                    logger.warning(f"Site totals don't match in chunk {chunk_num}: Site sum {total_records_in_sites}, expected {expected_total}")
                                    process_chunk_locally(chunk, all_results, report_type)
                                else:
                                    # Merge results
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
                                        
                                        site_data = all_results[report_type]["Sites"][site]
                                        site_data["Descriptions"].extend(data["Descriptions"])
                                        site_data["Created Date (WET)"].extend(data["Created Date (WET)"])
                                        site_data["Expected Close Date (WET)"].extend(data["Expected Close Date (WET)"])
                                        site_data["Status"].extend(data["Status"])
                                        site_data["Discipline"].extend(data["Discipline"])
                                        site_data["Modules"].extend(data["Modules"])
                                        site_data["SW"] += data["SW"]
                                        site_data["FW"] += data["FW"]
                                        site_data["MEP"] += data["MEP"]
                                        site_data["Total"] += data["Total"]
                                        
                                        for module, count in data["ModulesCount"].items():
                                            site_data["ModulesCount"][module] = site_data["ModulesCount"].get(module, 0) + count
                                    
                                    all_results[report_type]["Grand_Total"] += len(chunk)
                                    st.success(f"Successfully processed chunk {chunk_num} with {len(chunk)} records")
                        else:
                            logger.error(f"No valid JSON found in response for chunk {chunk_num}")
                            st.write("Falling back to local processing")
                            process_chunk_locally(chunk, all_results, report_type)
                    else:
                        error_msg = f"❌ WatsonX API error for chunk {chunk_num}: {response.status_code} - {response.text}"
                        st.error(error_msg)
                        logger.error(error_msg)
                        st.write("Falling back to local processing")
                        process_chunk_locally(chunk, all_results, report_type)
                        
                except requests.RequestException as e:
                    error_msg = f"❌ Request exception during WatsonX call for chunk {chunk_num}: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.write("Falling back to local processing")
                    process_chunk_locally(chunk, all_results, report_type)
                except Exception as e:
                    error_msg = f"❌ Exception during WatsonX call for chunk {chunk_num}: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.write("Falling back to local processing")
                    process_chunk_locally(chunk, all_results, report_type)

            # Remove empty sites from final results
            sites_to_remove = []
            for site_name, site_data in all_results[report_type]["Sites"].items():
                if site_data["Total"] == 0 or not site_data["Descriptions"]:
                    sites_to_remove.append(site_name)
            
            for site_name in sites_to_remove:
                del all_results[report_type]["Sites"][site_name]

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

    except Exception as e:
        error_msg = f"❌ Unexpected error in generate_ncr_report: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)


@st.cache_data
def process_chunk_locally(chunk, all_results, report_type):
    """Helper function to process chunks locally when API fails - MODIFIED FOR ELIGO AND LETTER TOWERS"""
    try:
        for record in chunk:
            tower = record.get("Tower", "Unknown")
            discipline = record.get("Discipline_Category", "Unknown")
            modules = record.get("Modules", ["Common"])
            
            # Convert modules to tower-specific format
            tower_specific_modules = []
            if tower.startswith("Eligo-Tower-"):
                tower_letter = tower.split("-")[2][0]  # Extract F, G, or H
                for module in modules:
                    if module.startswith("Module "):
                        module_num = module.split(" ")[1]
                        tower_specific_modules.append(f"{tower_letter}{module_num}")
                    else:
                        tower_specific_modules.append(module)
            else:
                tower_specific_modules = modules
            
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
            all_results[report_type]["Sites"][tower]["Modules"].append(tower_specific_modules)
            
            if discipline in ["SW", "FW", "MEP"]:
                all_results[report_type]["Sites"][tower][discipline] += 1
            
            all_results[report_type]["Sites"][tower]["Total"] += 1
            
            for module in tower_specific_modules:
                all_results[report_type]["Sites"][tower]["ModulesCount"][module] = all_results[report_type]["Sites"][tower]["ModulesCount"].get(module, 0) + 1
                
            all_results[report_type]["Grand_Total"] += 1
            
    except Exception as e:
        logger.error(f"Error in local processing: {str(e)}")
        raise

@st.cache_data

def generate_ncr_Housekeeping_report_for_eligo(df, report_type, start_date=None, end_date=None, until_date=None):
    """Generate Housekeeping NCR report for Open or Closed records."""
    with st.spinner(f"Generating {report_type} Housekeeping NCR Report with WatsonX..."):
        try:
            today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
            closed_start = pd.to_datetime(start_date) if start_date else None
            closed_end = pd.to_datetime(end_date) if end_date else None
            open_until = pd.to_datetime(until_date) if until_date else None

            # Define housekeeping keywords
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
                # Handle None or non-string descriptions
                if description is None or not isinstance(description, str):
                    logger.debug(f"Invalid description encountered: {description}")
                    return False
                description_lower = description.lower()
                return any(keyword in description_lower for keyword in housekeeping_keywords)

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
                        "Discipline": "HSE",
                        "Tower": "External Development"
                    }
                    if report_type == "Open":
                        cleaned_record["Days_From_Today"] = record.get("Days_From_Today", 0)

                    desc_lower = description.lower()
                    tower_match = re.search(r"(tower|t)\s*-?\s*([A-Za-z])", desc_lower, re.IGNORECASE)
                    cleaned_record["Tower"] = f"Eden-Tower{tower_match.group(2).zfill(2)}" if tower_match else "Common_Area"
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
                    "Task: Count Housekeeping NCRs by site ('Tower' field) where 'Discipline' is 'HSE' and 'Days' > 7 or 'Days_From_Today' > 7 for open records. The 'Description' must contain any of these housekeeping keywords (case-insensitive): "
                    "'housekeeping', 'cleaning', 'cleanliness', 'waste disposal', 'waste management', 'garbage', 'trash', 'rubbish', 'debris', 'litter', 'dust', 'untidy', 'cluttered', "
                    "'accumulation of waste', 'construction waste', 'pile of garbage', 'poor housekeeping', 'material storage', 'construction debris', 'cleaning schedule', 'garbage collection', "
                    "'waste bins', 'dirty', 'mess', 'unclean', 'disorderly', 'dirty floor', 'waste disposal area', 'waste collection', 'cleaning protocol', 'sanitation', 'trash removal', "
                    "'waste accumulation', 'unkept area', 'refuse collection', 'workplace cleanliness'. "
                    "Use 'Tower' values as they appear (e.g., 'Eligo-TowerG ', 'Eligo-TowerH' ,'Eligo-TowerF', 'Common_Area'). Collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', and 'Status' into arrays for each site. "
                    "Assign the count to 'Count' (No. of Housekeeping NCRs beyond 7 days). If no matches, set count to 0 for each site in the data. Return all sites present in the data.\n\n"
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
                                logger.debug(f"Extracted JSON string: {json_str}")
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
                                    if is_housekeeping_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
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
                                if is_housekeeping_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
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
                            if is_housekeeping_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
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
                        if is_housekeeping_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
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
                        if is_housekeeping_record(record["Description"]) and (record.get("Days", 0) > 7 or record.get("Days_From_Today", 0) > 7) and record.get("Discipline") == "HSE":
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
        except Exception as e:
            logger.error(f"Unexpected error in generate_ncr_Housekeeping_report: {str(e)}")
            st.error(f"❌ Unexpected Error: {str(e)}")
            return {"error": f"Unexpected Error: {str(e)}"}, ""
    
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
    
    # Second attempt: Extract JSON from print(json.dumps(...)) output
    json_match = re.search(r'print$$ json\.dumps\((.*?),\s*indent=2 $$\)', cleaned_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extracted JSON: {str(e)} - Extracted JSON: {json_str}")
    
    logger.error(f"JSONDecodeError: Unable to parse response - Cleaned response: {cleaned_text}")
    return None

@st.cache_data
def generate_ncr_Safety_report_for_eligo(df, report_type, start_date=None, end_date=None, until_date=None, debug_bypass_api=False):
    """Generate Safety NCR report for Open or Closed records."""
    with st.spinner(f"Generating {report_type} Safety NCR Report with WatsonX..."):
        try:
            today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
            closed_start = pd.to_datetime(start_date) if start_date else None
            closed_end = pd.to_datetime(end_date) if end_date else None
            open_until = pd.to_datetime(until_date) if until_date else None

            # Define safety keywords
            safety_keywords = [
                            'safety precautions', 'temporary electricity', 'safety norms', 'safety belt', 'helmet',
                            'lifeline', 'guard rails', 'fall protection', 'PPE', 'electrical hazard', 'unsafe platform',
                            'catch net', 'edge protection', 'TPI', 'scaffold', 'lifting equipment', 'dust suppression',
                            'debris chute', 'spill control', 'crane operator', 'halogen lamps', 'fall catch net',
                            'environmental contamination', 'fire hazard',
                            # New additions based on your description
                            'working at height', 'PPE kit', 'HSE norms', 'negligence in supervision', 'violation of HSE',
                            'tower h', 'non-tower area', 'nta'
                        ]

            def is_safety_record(description):
                if description is None or not isinstance(description, str):
                    logger.debug(f"Invalid description encountered: {description}")
                    return False
                description_lower = description.lower()
                return any(keyword in description_lower for keyword in safety_keywords)

            # Preprocess date columns
            df['Created Date (WET)'] = pd.to_datetime(df['Created Date (WET)'], errors='coerce').astype(str)
            df['Expected Close Date (WET)'] = pd.to_datetime(df['Expected Close Date (WET)'], errors='coerce').astype(str)

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
            else:  # Open - FIXED LOGIC
                # DEBUG: Add logging to see filtering steps
                logger.debug(f"Initial DataFrame size: {len(df)}")
                
                # Step 1: Filter by Status = 'Open'
                open_records = df[df['Status'] == 'Open'].copy()
                logger.debug(f"Records with Status='Open': {len(open_records)}")
                
                # Step 2: Filter by valid Created Date
                open_records = open_records[pd.to_datetime(open_records['Created Date (WET)']).notna()].copy()
                logger.debug(f"Records with valid Created Date: {len(open_records)}")
                
                # Step 3: Calculate days from today
                open_records.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(open_records['Created Date (WET)'])).dt.days
                logger.debug(f"Days_From_Today calculated for {len(open_records)} records")
                
                # Step 4: Filter by Days_From_Today > 7
                open_records = open_records[open_records['Days_From_Today'] > 7].copy()
                logger.debug(f"Records with Days_From_Today > 7: {len(open_records)}")
                
                # Step 5: Apply date filter if specified
                if open_until:
                    open_records = open_records[
                        (pd.to_datetime(open_records['Created Date (WET)']) <= open_until)
                    ].copy()
                    logger.debug(f"Records after date filter (until {open_until}): {len(open_records)}")
                
                # Step 6: CORRECTED LOGIC - Filter by HSE discipline AND safety keywords in description
                # For open records, we want HSE records that ALSO have safety keywords (same as closed)
                filtered_df = open_records[
                    (open_records['Discipline'] == 'HSE') &
                    (open_records['Description'].apply(is_safety_record))
                ].copy()
                
                logger.debug(f"Final filtered records (HSE AND safety keywords): {len(filtered_df)}")

            if filtered_df.empty:
                logger.info("No safety records found after filtering")
                return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

            # DEBUG: Log sample of filtered data
            logger.debug(f"Sample of filtered data:\n{filtered_df[['Description', 'Discipline', 'Status', 'Created Date (WET)']].head()}")

            processed_data = filtered_df.to_dict(orient="records")
            
            cleaned_data = []
            seen_descriptions = set()
            for record in processed_data:
                description = str(record.get("Description", "")).strip()
                if description and description not in seen_descriptions:
                    seen_descriptions.add(description)
                    days = record.get("Days", 0)
                    days_from_today = record.get("Days_From_Today", 0)
                    
                    # DEBUG: Log why records might be skipped
                    if (report_type == "Closed" and days <= 7):
                        logger.debug(f"Skipping closed record due to insufficient days: {description}, Days={days}")
                        continue
                    elif (report_type == "Open" and days_from_today <= 7):
                        logger.debug(f"Skipping open record due to insufficient days from today: {description}, Days_From_Today={days_from_today}")
                        continue
                    
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
                    cleaned_record["Tower"] = f"Eden-Tower{tower_match.group(2).upper().zfill(2)}" if tower_match else "Common_Area"
                    logger.debug(f"Tower set to {cleaned_record['Tower']} for description: {description[:50]}...")

                    cleaned_data.append(cleaned_record)

            st.write(f"Total {report_type} records to process: {len(cleaned_data)}")
            logger.debug(f"Cleaned data count: {len(cleaned_data)}")
            logger.debug(f"Sample cleaned record: {cleaned_data[0] if cleaned_data else 'No records'}")

            if not cleaned_data:
                logger.info("No safety records after deduplication and processing")
                return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

            result = {"Safety": {"Sites": {}, "Grand_Total": 0}}

            if debug_bypass_api:
                logger.info("Bypassing WatsonX API for debugging")
                for record in cleaned_data:
                    # CORRECTED: For both open and closed records, check HSE discipline AND safety keywords
                    is_hse = record.get("Discipline") == "HSE"
                    has_safety_keywords = is_safety_record(record["Description"])
                    days_check = record.get("Days", 0) > 7 if report_type == "Closed" else record.get("Days_From_Today", 0) > 7
                    
                    logger.debug(f"Record check - HSE: {is_hse}, Safety keywords: {has_safety_keywords}, Days check: {days_check}")
                    
                    # CORRECTED: Use AND logic for both open and closed records
                    if is_hse and has_safety_keywords and days_check:
                        site = record["Tower"]
                        if site not in result["Safety"]["Sites"]:
                            result["Safety"]["Sites"][site] = {
                                "Count": 0,
                                "Descriptions": [],
                                "Created Date (WET)": [],
                                "Expected Close Date (WET)": [],
                                "Discipline": record.get("Discipline", ""),
                                "Status": []
                            }
                        result["Safety"]["Sites"][site]["Descriptions"].append(record["Description"])
                        result["Safety"]["Sites"][site]["Created Date (WET)"].append(record["Created Date (WET)"])
                        result["Safety"]["Sites"][site]["Expected Close Date (WET)"].append(record["Expected Close Date (WET)"])
                        result["Safety"]["Sites"][site]["Status"].append(record["Status"])
                        result["Safety"]["Sites"][site]["Count"] += 1
                        result["Safety"]["Grand_Total"] += 1
                        logger.debug(f"Added record to {site}: {record['Description'][:50]}...")
                    else:
                        logger.debug(f"Skipped record: HSE={is_hse}, Safety={has_safety_keywords}, Days={days_check}")
                        
                logger.debug(f"Debug result: {json.dumps(result, indent=2)}")
                return result, json.dumps(result)

            # Rest of the function remains the same...
            access_token = get_access_token(API_KEY)
            if not access_token:
                logger.error("Failed to obtain access token")
                st.error("Failed to obtain access token")
                return {"error": "Failed to obtain access token"}, ""

            chunk_size = 10
            total_chunks = (len(cleaned_data) + chunk_size - 1) // chunk_size

            # Enhanced session configuration with longer timeouts and better retry strategy
            session = requests.Session()
            retry_strategy = Retry(
                total=5,  # Increased retries
                backoff_factor=3,  # Longer backoff
                status_forcelist=[500, 502, 503, 504, 429, 408, 524],  # Added 524 timeout error
                allowed_methods=["POST"],
                raise_on_redirect=True,
                raise_on_status=False  # Don't raise on status to handle manually
            )
            adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
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
                logger.debug(f"Processing chunk {current_chunk}/{total_chunks}: {json.dumps(chunk, indent=2)}")

                prompt = (
                    "Generate ONE JSON object in the exact format below. Do not include code, explanations, multiple objects, or repeat input data. Count Safety NCRs by 'Tower' where 'Discipline' is 'HSE' AND description contains safety keywords. For open records, use 'Days_From_Today' > 7; for closed, use 'Days' > 7. Descriptions must contain these keywords (case-insensitive): "
                    "'safety precautions', 'temporary electricity', 'safety norms', 'safety belt', 'helmet', 'lifeline', 'guard rails', 'fall protection', 'PPE', 'electrical hazard', 'unsafe platform', "
                    "'catch net', 'edge protection', 'TPI', 'scaffold', 'lifting equipment', 'dust suppression', 'debris chute', 'spill control', 'crane operator', 'halogen lamps', 'fall catch net', 'PPE'," 
                    "'working at height', 'PPE kit', 'HSE norms', 'negligence in supervision', 'violation of HSE','tower h', 'non-tower area', 'nta'"
                    "'environmental contamination', 'fire hazard'. Group by 'Tower' (e.g., 'Eden-Tower06'). Include all input sites, even with count 0. Collect 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status' in arrays.\n\n"
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
                    f"Input Data: {json.dumps(chunk)}\n"
                )

                payload = {
                    "input": prompt,
                    "parameters": {
                        "decoding_method": "greedy",
                        "max_new_tokens": 1500,  # Reduced to avoid timeout
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
                    logger.debug(f"Sending WatsonX payload for chunk {current_chunk}: {json.dumps(payload, indent=2)}")
                    # Increased timeout to 60 seconds
                    response = session.post(
                        WATSONX_API_URL, 
                        headers=headers, 
                        json=payload, 
                        verify=certifi.where(), 
                        timeout=60  # Extended timeout
                    )
                    logger.info(f"WatsonX API response status for chunk {current_chunk}: {response.status_code}")

                    if response.status_code == 200:
                        api_result = response.json()
                        generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                        logger.debug(f"Generated text for chunk {current_chunk}: {generated_text}")

                        # Improved JSON extraction
                        json_str = None
                        try:
                            # Look for the first valid JSON object in the response
                            start_idx = generated_text.find('{')
                            end_idx = generated_text.rfind('}') + 1
                            if start_idx != -1 and end_idx != 0:
                                json_str = generated_text[start_idx:end_idx]
                                parsed_json = json.loads(json_str)
                                logger.debug(f"Successfully parsed JSON for chunk {current_chunk}: {json_str}")
                            else:
                                logger.warning(f"No JSON object found in generated text for chunk {current_chunk}: {generated_text}")
                                raise json.JSONDecodeError("No valid JSON object found", generated_text, 0)
                        except json.JSONDecodeError as je:
                            logger.error(f"JSONDecodeError for chunk {current_chunk}: {str(je)} - Generated text: {generated_text}")
                            error_placeholder.error(f"Failed to parse JSON for chunk {current_chunk}: Invalid JSON format")
                            # Fallback to manual processing
                            for record in chunk:
                                is_hse = record.get("Discipline") == "HSE"
                                has_safety_keywords = is_safety_record(record["Description"])
                                days_check = record.get("Days", 0) > 7 if report_type == "Closed" else record.get("Days_From_Today", 0) > 7
                                
                                # CORRECTED: Use AND logic consistently
                                if is_hse and has_safety_keywords and days_check:
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
                            continue

                        if "Safety" in parsed_json:
                            chunk_result = parsed_json.get("Safety", {})
                            chunk_sites = chunk_result.get("Sites", {})
                            chunk_grand_total = chunk_result.get("Grand_Total", 0)

                            for site, values in chunk_sites.items():
                                if not isinstance(values, dict):
                                    logger.warning(f"Invalid site data for {site} in chunk {current_chunk}: {values}")
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
                        else:
                            logger.warning(f"Unexpected JSON format for chunk {current_chunk}: {json_str}")
                            error_placeholder.error(f"Unexpected JSON format for chunk {current_chunk}")
                            # Fallback to manual processing - CORRECTED
                            for record in chunk:
                                is_hse = record.get("Discipline") == "HSE"
                                has_safety_keywords = is_safety_record(record["Description"])
                                days_check = record.get("Days", 0) > 7 if report_type == "Closed" else record.get("Days_From_Today", 0) > 7
                                
                                if is_hse and has_safety_keywords and days_check:
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
                        error_placeholder.error(f"WatsonX API error for chunk {current_chunk}: {response.status_code} - {response.text}")
                        # Fallback to manual processing - CORRECTED
                        for record in chunk:
                            is_hse = record.get("Discipline") == "HSE"
                            has_safety_keywords = is_safety_record(record["Description"])
                            days_check = record.get("Days", 0) > 7 if report_type == "Closed" else record.get("Days_From_Today", 0) > 7
                            
                            if is_hse and has_safety_keywords and days_check:
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
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
                    logger.error(f"Timeout error for chunk {current_chunk}: {str(e)}")
                    st.warning(f"⚠️ Timeout for chunk {current_chunk}. Using fallback processing...")
                    # Fallback to manual processing for this chunk - CORRECTED
                    for record in chunk:
                        is_hse = record.get("Discipline") == "HSE"
                        has_safety_keywords = is_safety_record(record["Description"])
                        days_check = record.get("Days", 0) > 7 if report_type == "Closed" else record.get("Days_From_Today", 0) > 7
                        
                        if is_hse and has_safety_keywords and days_check:
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
                    st.warning(f"⚠️ Connection error for chunk {current_chunk}. Using fallback processing...")
                    # Fallback to manual processing - CORRECTED
                    for record in chunk:
                        is_hse = record.get("Discipline") == "HSE"
                        has_safety_keywords = is_safety_record(record["Description"])
                        days_check = record.get("Days", 0) > 7 if report_type == "Closed" else record.get("Days_From_Today", 0) > 7
                        
                        if is_hse and has_safety_keywords and days_check:
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

            # Validate WatsonX result against cleaned_data - CORRECTED
            if result["Safety"]["Grand_Total"] == 0 and cleaned_data:
                logger.warning("WatsonX returned zero count despite valid records; using fallback counting")
                for record in cleaned_data:
                    is_hse = record.get("Discipline") == "HSE"
                    has_safety_keywords = is_safety_record(record["Description"])
                    days_check = record.get("Days", 0) > 7 if report_type == "Closed" else record.get("Days_From_Today", 0) > 7
                    
                    if is_hse and has_safety_keywords and days_check:
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

            # Deduplication and fix data types for PyArrow compatibility
            for site in result["Safety"]["Sites"]:
                # Ensure all values are strings and deduplicate
                result["Safety"]["Sites"][site]["Descriptions"] = list(set(
                    [str(desc) for desc in result["Safety"]["Sites"][site]["Descriptions"] if desc]
                ))
                result["Safety"]["Sites"][site]["Created Date (WET)"] = list(set(
                    [str(date) for date in result["Safety"]["Sites"][site]["Created Date (WET)"] if date]
                ))
                result["Safety"]["Sites"][site]["Expected Close Date (WET)"] = list(set(
                    [str(date) for date in result["Safety"]["Sites"][site]["Expected Close Date (WET)"] if date]
                ))
                result["Safety"]["Sites"][site]["Status"] = list(set(
                    [str(status) for status in result["Safety"]["Sites"][site]["Status"] if status]
                ))
                # Update count to match actual unique items
                result["Safety"]["Sites"][site]["Count"] = len(result["Safety"]["Sites"][site]["Descriptions"])
            
            # Recalculate grand total after deduplication
            result["Safety"]["Grand_Total"] = sum(
                site_data["Count"] for site_data in result["Safety"]["Sites"].values()
            )
            
            logger.debug(f"Final result after deduplication: {json.dumps(result, indent=2)}")
            
            # Clear progress displays
            progress_placeholder.empty()
            status_placeholder.empty()
            error_placeholder.empty()
            
            return result, json.dumps(result)
            
        except Exception as e:
            logger.error(f"Unexpected error in generate_ncr_Safety_report: {str(e)}")
            st.error(f"❌ Unexpected Error: {str(e)}")
            return {"error": f"Unexpected Error: {str(e)}"}, ""
    

@st.cache_data
def generate_consolidated_ncr_OpenClose_excel_for_eligo(combined_result, report_title="NCR"):
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
            "Tower-F": '#F9E79F',  # Soft yellow
            "Tower-G": '#A3CFFA',  # Blue
            "Tower-H": '#F5C3C2',  # Pink
            "Common_Area": '#B5EAD7',    # Mint
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
        
        # Extract data with better error handling
        print("DEBUG: combined_result structure:", combined_result)
        
        resolved_data = combined_result.get("NCR resolved beyond 21 days", {})
        open_data = combined_result.get("NCR open beyond 21 days", {})
        
        print("DEBUG: resolved_data type and structure:", type(resolved_data))
        print("DEBUG: resolved_data keys:", list(resolved_data.keys()) if isinstance(resolved_data, dict) else "Not a dict")
        print("DEBUG: open_data type and structure:", type(open_data))
        print("DEBUG: open_data keys:", list(open_data.keys()) if isinstance(open_data, dict) else "Not a dict")
        
        # Handle different possible data structures
        if isinstance(resolved_data, dict) and "Sites" in resolved_data:
            resolved_sites = resolved_data["Sites"]
        elif isinstance(resolved_data, dict):
            resolved_sites = resolved_data
        else:
            resolved_sites = {}
            
        if isinstance(open_data, dict) and "Sites" in open_data:
            open_sites = open_data["Sites"]
        elif isinstance(open_data, dict):
            open_sites = open_data
        else:
            open_sites = {}
        
        print("DEBUG: resolved_sites keys:", list(resolved_sites.keys()))
        print("DEBUG: resolved_sites sample data:", {k: type(v) for k, v in list(resolved_sites.items())[:2]})
        print("DEBUG: open_sites keys:", list(open_sites.keys()))
        print("DEBUG: open_sites sample data:", {k: type(v) for k, v in list(open_sites.items())[:2]})
        
        # Print first resolved site structure in detail
        if resolved_sites:
            first_key = list(resolved_sites.keys())[0]
            print(f"DEBUG: First resolved site '{first_key}' structure:", resolved_sites[first_key])
        
        # Print first open site structure in detail  
        if open_sites:
            first_key = list(open_sites.keys())[0]
            print(f"DEBUG: First open site '{first_key}' structure:", open_sites[first_key])
        
        # Define standard sites
        standard_sites = [
            "Eligo-Tower-F", "Eligo-Tower-G", "Eligo-Tower-H", "Common_Area"
        ]
        
        # Define module levels
        module_levels = {
            "Tower-F": ["F1", "F2", "Common Description"],
            "Tower-F": ["G1", "G2", "G3", "Common Description"],
            "Tower-F": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "Common Description"]
        }
        
        # Write header
        worksheet.merge_range('A1:H1', f"{report_title} {date_part}", title_format)
        row = 1
        worksheet.write(row, 0, 'Site/WET', header_format)
        worksheet.merge_range(row, 1, row, 3, 'NCR resolved beyond 45 days', header_format)
        worksheet.merge_range(row, 4, row, 6, 'NCR open beyond 45 days', header_format)
        worksheet.write(row, 7, 'Total', header_format)
        
        row = 2
        categories = ['Finishing', 'Works', 'MEP']
        worksheet.write(row, 0, '', header_format)
        for i, cat in enumerate(categories):
            worksheet.write(row, i+1, cat, subheader_format)
        for i, cat in enumerate(categories):
            worksheet.write(row, i+4, cat, subheader_format)
        worksheet.write(row, 7, '', header_format)
        
        # Updated category mapping to match more variations
        category_map = {
            'Finishing': ['FW', 'Civil Finishing', 'Finishing', 'finishing'],
            'Works': ['SW', 'Structure Works', 'Works', 'Structure', 'works'],
            'MEP': ['MEP', 'mep']
        }
        
        def get_category_count(site_data, category_key):
            """Helper function to get count for a category with flexible key matching"""
            count = 0
            possible_keys = category_map.get(category_key, [category_key])
            
            for key in possible_keys:
                if key in site_data:
                    value = site_data[key]
                    if isinstance(value, (int, float)):
                        count += int(value)
                    elif isinstance(value, list):
                        count += len(value)
            return count
        
        def process_module_data(site_data, tower_letter):
            """Process module-based data structure"""
            module_counts = {level: {'Finishing': 0, 'Works': 0, 'MEP': 0} 
                           for level in module_levels.get(f"Tower-{tower_letter}", [])}
            
            if not site_data:
                return module_counts
                
            disciplines = site_data.get("Discipline", [])
            modules = site_data.get("Modules", [])
            
            print(f"DEBUG: Processing {tower_letter} - disciplines: {disciplines}, modules: {modules}")
            
            for i, discipline in enumerate(disciplines):
                if i < len(modules):
                    module_list = modules[i] if isinstance(modules[i], list) else [modules[i]]
                    
                    # Map discipline to category
                    if discipline in ['FW', 'Civil Finishing', 'Finishing']:
                        cat = 'Finishing'
                    elif discipline in ['SW', 'Structure Works', 'Works']:
                        cat = 'Works'
                    elif discipline in ['MEP']:
                        cat = 'MEP'
                    else:
                        continue
                    
                    for module in module_list:
                        if module in module_levels.get(f"Tower-{tower_letter}", []):
                            module_counts[module][cat] += 1
            
            return module_counts
        
        row = 3
        site_totals = {}
        
        for site in standard_sites:
            print(f"DEBUG: Processing site: {site}")
            
            # Find matching keys in the data
            resolved_site_data = {}
            open_site_data = {}
            
            # Look for exact matches or similar patterns
            for key in resolved_sites.keys():
                if site.lower().replace("-", "").replace("_", "") in key.lower().replace("-", "").replace("_", ""):
                    resolved_site_data = resolved_sites[key]
                    print(f"DEBUG: Found resolved data for {site}: {key}")
                    break
                    
            for key in open_sites.keys():
                if site.lower().replace("-", "").replace("_", "") in key.lower().replace("-", "").replace("_", ""):
                    open_site_data = open_sites[key]
                    print(f"DEBUG: Found open data for {site}: {key}")
                    break
            
            # Get tower formatting
            formats = tower_formats.get(site, {})
            tower_total_format = formats.get('tower_total', workbook.add_format({
                'bold': True, 'align': 'left', 'valign': 'vcenter', 'border': 1, 'fg_color': '#D3D3D3'
            }))
            site_format = formats.get('site', default_site_format)
            cell_format = formats.get('cell', default_cell_format)
            
            resolved_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
            open_counts = {'Finishing': 0, 'Works': 0, 'MEP': 0}
            
            tower_letter = site.split("-")[-1] if "Tower" in site else None
            
            if "Tower" in site:
                # Process tower data with modules
                resolved_module_counts = process_module_data(resolved_site_data, tower_letter)
                open_module_counts = process_module_data(open_site_data, tower_letter)
                
                # Aggregate counts
                for cat in categories:
                    resolved_counts[cat] = sum(resolved_module_counts[level][cat] 
                                            for level in module_levels.get(f"Tower-{tower_letter}", []))
                    open_counts[cat] = sum(open_module_counts[level][cat] 
                                         for level in module_levels.get(f"Tower-{tower_letter}", []))
                
                print(f"DEBUG: {site} resolved_counts: {resolved_counts}")
                print(f"DEBUG: {site} open_counts: {open_counts}")
                
            else:
                # Process Common Area data
                for cat in categories:
                    resolved_counts[cat] = get_category_count(resolved_site_data, cat)
                    open_counts[cat] = get_category_count(open_site_data, cat)
            
            site_total = sum(resolved_counts.values()) + sum(open_counts.values())
            
            # Write tower row
            display_site = f"Tower {tower_letter}" if tower_letter in ['F', 'G', 'H'] else site.replace("_", " ")
            worksheet.write(row, 0, display_site, tower_total_format)
            for i, cat in enumerate(categories):
                worksheet.write(row, i+1, resolved_counts[cat], cell_format)
            for i, cat in enumerate(categories):
                worksheet.write(row, i+4, open_counts[cat], cell_format)
            worksheet.write(row, 7, site_total, cell_format)
            site_totals[site] = site_total
            row += 1
            
            # Add module rows for towers
            if "Tower" in site:
                for module in module_levels.get(f"Tower-{tower_letter}", []):
                    resolved_module_count = resolved_module_counts.get(module, {'Finishing': 0, 'Works': 0, 'MEP': 0})
                    open_module_count = open_module_counts.get(module, {'Finishing': 0, 'Works': 0, 'MEP': 0})
                    
                    module_total = sum(resolved_module_count.values()) + sum(open_module_count.values())
                    
                    worksheet.write(row, 0, module, site_format)
                    for i, cat in enumerate(categories):
                        worksheet.write(row, i+1, resolved_module_count[cat], cell_format)
                    for i, cat in enumerate(categories):
                        worksheet.write(row, i+4, open_module_count[cat], cell_format)
                    worksheet.write(row, 7, module_total, cell_format)
                    row += 1

        def write_detail_sheet(sheet_name, data, title):
            truncated_sheet_name = truncate_sheet_name(f"{sheet_name} {date_part}")
            detail_worksheet = workbook.add_worksheet(truncated_sheet_name)
            detail_worksheet.set_column('A:A', 20)
            detail_worksheet.set_column('B:B', 60)
            detail_worksheet.set_column('C:D', 20)
            detail_worksheet.set_column('E:E', 15)
            detail_worksheet.set_column('F:G', 15)
            detail_worksheet.merge_range('A1:G1', f"{title} {date_part}", title_format)
            headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline', 'Modules']
            for col, header in enumerate(headers):
                detail_worksheet.write(1, col, header, header_format)
            row = 2
            
            if not data or not isinstance(data, dict):
                # Write a "No data available" row if data is empty or invalid
                detail_worksheet.write(row, 0, "No data available", default_site_format)
                for col in range(1, 7):
                    detail_worksheet.write(row, col, "", default_cell_format)
                return
            
            for site, site_data in data.items():
                print(f"DEBUG: Processing detail sheet for site: {site}")
                print(f"DEBUG: site_data type: {type(site_data)}")
                print(f"DEBUG: site_data content: {site_data}")
                
                # Handle case where site_data is not a dictionary
                if not isinstance(site_data, dict):
                    print(f"DEBUG: site_data is not a dict: {site_data}")
                    detail_worksheet.write(row, 0, str(site), default_site_format)
                    detail_worksheet.write(row, 1, "Data format error - not a dictionary", default_cell_format)
                    for col in range(2, 7):
                        detail_worksheet.write(row, col, "", default_cell_format)
                    row += 1
                    continue
                
                # Check if this is a count-based structure (like {"FW": 8, "SW": 2})
                # vs a list-based structure (like {"Descriptions": [...], "Discipline": [...]})
                has_list_structure = any(key in site_data for key in ["Descriptions", "Description", "Created Date (WET)", "Status", "Discipline", "Modules"])
                has_count_structure = any(key in site_data for key in ["FW", "SW", "MEP", "Civil Finishing", "Structure Works"])
                
                print(f"DEBUG: has_list_structure: {has_list_structure}, has_count_structure: {has_count_structure}")
                
                if has_count_structure and not has_list_structure:
                    # This is a count-based structure, convert to display format
                    print(f"DEBUG: Converting count-based structure for {site}")
                    detail_worksheet.write(row, 0, str(site), default_site_format)
                    detail_worksheet.write(row, 1, "Summary counts only - no detailed records", default_cell_format)
                    
                    # Show the counts in the description
                    count_info = []
                    for key, value in site_data.items():
                        if isinstance(value, (int, float)) and value > 0:
                            count_info.append(f"{key}: {value}")
                        elif isinstance(value, list) and len(value) > 0:
                            count_info.append(f"{key}: {len(value)} items")
                    
                    detail_worksheet.write(row, 1, "; ".join(count_info) if count_info else "No counts available", default_cell_format)
                    for col in range(2, 7):
                        detail_worksheet.write(row, col, "", default_cell_format)
                    row += 1
                    continue
                
                # Handle list-based structure
                try:
                    # Try different possible key names for descriptions
                    descriptions = []
                    for desc_key in ["Descriptions", "Description", "descriptions", "description"]:
                        if desc_key in site_data:
                            desc_data = site_data[desc_key]
                            if isinstance(desc_data, list):
                                descriptions = desc_data
                            elif isinstance(desc_data, str):
                                descriptions = [desc_data]
                            break
                    
                    # Similarly for other fields
                    created_dates = []
                    for date_key in ["Created Date (WET)", "Created Date", "created_date"]:
                        if date_key in site_data:
                            date_data = site_data[date_key]
                            if isinstance(date_data, list):
                                created_dates = date_data
                            elif isinstance(date_data, str):
                                created_dates = [date_data]
                            break
                    
                    close_dates = []
                    for close_key in ["Expected Close Date (WET)", "Expected Close Date", "close_date"]:
                        if close_key in site_data:
                            close_data = site_data[close_key]
                            if isinstance(close_data, list):
                                close_dates = close_data
                            elif isinstance(close_data, str):
                                close_dates = [close_data]
                            break
                    
                    statuses = []
                    for status_key in ["Status", "status"]:
                        if status_key in site_data:
                            status_data = site_data[status_key]
                            if isinstance(status_data, list):
                                statuses = status_data
                            elif isinstance(status_data, str):
                                statuses = [status_data]
                            break
                    
                    disciplines = []
                    for disc_key in ["Discipline", "discipline"]:
                        if disc_key in site_data:
                            disc_data = site_data[disc_key]
                            if isinstance(disc_data, list):
                                disciplines = disc_data
                            elif isinstance(disc_data, str):
                                disciplines = [disc_data]
                            break
                    
                    modules = []
                    for mod_key in ["Modules", "modules"]:
                        if mod_key in site_data:
                            mod_data = site_data[mod_key]
                            if isinstance(mod_data, list):
                                modules = mod_data
                            elif isinstance(mod_data, str):
                                modules = [mod_data]
                            break
                    
                    print(f"DEBUG: Extracted data - descriptions: {len(descriptions)}, disciplines: {len(disciplines)}, modules: {len(modules)}")
                    
                    # Handle empty lists
                    if not any([descriptions, created_dates, close_dates, statuses, disciplines, modules]):
                        # If no list data is available, show available keys
                        detail_worksheet.write(row, 0, str(site), default_site_format)
                        available_keys = ", ".join(site_data.keys())
                        detail_worksheet.write(row, 1, f"No standard fields found. Available keys: {available_keys}", default_cell_format)
                        for col in range(2, 7):
                            detail_worksheet.write(row, col, "", default_cell_format)
                        row += 1
                        continue
                    
                    # Get maximum length safely
                    max_length = max(
                        len(descriptions) if descriptions else 0,
                        len(created_dates) if created_dates else 0,
                        len(close_dates) if close_dates else 0,
                        len(statuses) if statuses else 0,
                        len(disciplines) if disciplines else 0,
                        len(modules) if modules else 0,
                        1  # Minimum 1 row
                    )
                    
                    for i in range(max_length):
                        detail_worksheet.write(row, 0, str(site), default_site_format)
                        detail_worksheet.write(row, 1, str(descriptions[i]) if i < len(descriptions) else "", default_cell_format)
                        detail_worksheet.write(row, 2, str(created_dates[i]) if i < len(created_dates) else "", default_cell_format)
                        detail_worksheet.write(row, 3, str(close_dates[i]) if i < len(close_dates) else "", default_cell_format)
                        detail_worksheet.write(row, 4, str(statuses[i]) if i < len(statuses) else "", default_cell_format)
                        detail_worksheet.write(row, 5, str(disciplines[i]) if i < len(disciplines) else "", default_cell_format)
                        
                        # Handle modules safely
                        if i < len(modules) and modules[i]:
                            if isinstance(modules[i], list):
                                modules_str = ', '.join(str(m) for m in modules[i])
                            else:
                                modules_str = str(modules[i])
                        else:
                            modules_str = ""
                        detail_worksheet.write(row, 6, modules_str, default_cell_format)
                        row += 1
                        
                except Exception as e:
                    print(f"DEBUG: Error processing site {site}: {e}")
                    detail_worksheet.write(row, 0, str(site), default_site_format)
                    detail_worksheet.write(row, 1, f"Error: {str(e)}", default_cell_format)
                    for col in range(2, 7):
                        detail_worksheet.write(row, col, "", default_cell_format)
                    row += 1

        # Create detail sheets with error handling
        try:
            if resolved_sites and isinstance(resolved_sites, dict):
                write_detail_sheet("Closed NCR Details", resolved_sites, "Closed NCR Details")
            else:
                print("DEBUG: No valid resolved_sites data for detail sheet")
        except Exception as e:
            print(f"DEBUG: Error creating Closed NCR Details sheet: {e}")
        
        try:
            if open_sites and isinstance(open_sites, dict):
                write_detail_sheet("Open NCR Details", open_sites, "Open NCR Details")
            else:
                print("DEBUG: No valid open_sites data for detail sheet")
        except Exception as e:
            print(f"DEBUG: Error creating Open NCR Details sheet: {e}")

        output.seek(0)
        return output

@st.cache_data
def generate_consolidated_ncr_Housekeeping_excel_for_eligo(combined_result, report_title="Housekeeping: Current Month"):
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
        now = datetime.now()
        day = now.strftime("%d")
        month_name = now.strftime("%B")
        year = now.strftime("%Y")
        date_part = f"{day}_{month_name}_{year}"
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
        
        # Get the data from combined_result
        data = combined_result.get("Housekeeping", {}).get("Sites", {})
        
        # Debug: Print the actual data structure
        print("DEBUG: Housekeeping data structure:")
        print(f"Available sites in data: {list(data.keys())}")
        for site_key, site_data in data.items():
            print(f"Site: {site_key}, Count: {site_data.get('Count', 'N/A')}")
        
        # Create a mapping from actual data keys to display names
        # Based on your data generation function, the actual keys are:
        # "Eligo-Tower-F", "Eligo-Tower-G", "Eligo-Tower-H", "Eligo-Club", "External Development"
        site_key_mapping = {}
        
        for data_key in data.keys():
            # Map based on patterns from your data generation function
            if "eligo-tower-f" in data_key.lower() or data_key.lower() == "eligo-tower-f":
                site_key_mapping[data_key] = "Tower F"
            elif "eligo-tower-g" in data_key.lower() or data_key.lower() == "eligo-tower-g":
                site_key_mapping[data_key] = "Tower G"
            elif "eligo-tower-h" in data_key.lower() or data_key.lower() == "eligo-tower-h":
                site_key_mapping[data_key] = "Tower H"
            elif "eligo-club" in data_key.lower() or data_key.lower() == "eligo-club":
                site_key_mapping[data_key] = "Common Area"
            elif "external development" in data_key.lower():
                site_key_mapping[data_key] = "External Development"
            # Fallback pattern matching for other tower variations
            elif re.search(r'tower[- ]?f', data_key, re.IGNORECASE):
                site_key_mapping[data_key] = "Tower F"
            elif re.search(r'tower[- ]?g', data_key, re.IGNORECASE):
                site_key_mapping[data_key] = "Tower G"
            elif re.search(r'tower[- ]?h', data_key, re.IGNORECASE):
                site_key_mapping[data_key] = "Tower H"
            elif re.search(r'common[- ]?area', data_key, re.IGNORECASE):
                site_key_mapping[data_key] = "Common Area"
            else:
                # Keep original name if no match found
                site_key_mapping[data_key] = data_key
        
        print(f"DEBUG: Site key mapping: {site_key_mapping}")
        
        # Create reverse mapping for lookup
        display_to_data_key = {v: k for k, v in site_key_mapping.items()}
        
        # Define the order of sites to display (including External Development if needed)
        ordered_display_sites = ["Tower F", "Tower G", "Tower H", "Common Area"]
        
        # Add any unmapped sites to the display list
        for display_name in site_key_mapping.values():
            if display_name not in ordered_display_sites:
                ordered_display_sites.append(display_name)
        
        # Write summary sheet
        worksheet_summary.merge_range('A1:B1', report_title, title_format)
        row = 1
        worksheet_summary.write(row, 0, 'Site', header_format)
        worksheet_summary.write(row, 1, 'No. of Housekeeping NCRs beyond 7 days', header_format)
        
        row = 2
        for display_site in ordered_display_sites:
            worksheet_summary.write(row, 0, display_site, site_format)
            
            # Find the corresponding data key
            data_key = display_to_data_key.get(display_site)
            
            if data_key and data_key in data:
                count_value = data[data_key].get("Count", 0)
                print(f"DEBUG: {display_site} -> {data_key} -> Count: {count_value}")
            else:
                count_value = 0
                print(f"DEBUG: {display_site} -> No data found")
            
            worksheet_summary.write(row, 1, count_value, cell_format)
            row += 1
        
        # Write details sheet
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
        for display_site in ordered_display_sites:
            data_key = display_to_data_key.get(display_site)
            
            if data_key and data_key in data:
                site_data = data[data_key]
                descriptions = site_data.get("Descriptions", [])
                created_dates = site_data.get("Created Date (WET)", [])
                close_dates = site_data.get("Expected Close Date (WET)", [])
                statuses = site_data.get("Status", [])
                
                max_length = max(len(descriptions), len(created_dates), len(close_dates), len(statuses))
                
                if max_length > 0:
                    for i in range(max_length):
                        worksheet_details.write(row, 0, display_site, site_format)
                        worksheet_details.write(row, 1, descriptions[i] if i < len(descriptions) else "", description_format)
                        worksheet_details.write(row, 2, created_dates[i] if i < len(created_dates) else "", cell_format)
                        worksheet_details.write(row, 3, close_dates[i] if i < len(close_dates) else "", cell_format)
                        worksheet_details.write(row, 4, statuses[i] if i < len(statuses) else "", cell_format)
                        worksheet_details.write(row, 5, "HSE", cell_format)
                        row += 1
        
        output.seek(0)
        return output

@st.cache_data        
def generate_combined_excel_report_for_eligo(all_reports, filename_prefix="All_Reports"):
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
            "Eligo-Tower-F": '#C4E4B7',  # Green
            "Eligo-Tower-G": '#A3CFFA',  # Blue
            "Eligo-Tower-H": '#F5C3C2',  # Pink
            "Common_Area": '#C7CEEA'     # Periwinkle
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
        now = datetime.now()
        day = now.strftime("%d")
        month_name = now.strftime("%B")
        year = now.strftime("%Y")
        date_part = f"{day}_{month_name}_{year}"
        
        def truncate_sheet_name(base_name, max_length=31):
            if len(base_name) > max_length:
                return base_name[:max_length - 3] + "..."
            return base_name

        # Define module levels
        module_levels = {
            "Eligo-Tower-F": ["F1", "F2", "Common Description"],
            "Eligo-Tower-G": ["G1", "G2", "G3", "Common Description"],
            "Eligo-Tower-H": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "Common Description"]
        }

        # FIXED: Updated normalize_site_name function to handle Eden-Tower patterns
        def normalize_site_name(site):
            standard_sites = ["Eligo-Tower-F", "Eligo-Tower-G", "Eligo-Tower-H", "Common_Area"]
            
            if site in standard_sites:
                return site
            
            # Handle Eden-Tower0* patterns (e.g., Eden-Tower0G -> Eligo-Tower-G)
            eden_match = re.search(r'eden[- ]?tower[- ]?0([fgh])', site, re.IGNORECASE)
            if eden_match:
                letter = eden_match.group(1).upper()
                return f"Eligo-Tower-{letter}"
            
            # Handle Eden-Tower-* patterns (e.g., Eden-Tower-G -> Eligo-Tower-G)
            eden_match = re.search(r'eden[- ]?tower[- ]?([fgh])', site, re.IGNORECASE)
            if eden_match:
                letter = eden_match.group(1).upper()
                return f"Eligo-Tower-{letter}"
            
            # Handle Eligo-Tower-* patterns
            eligo_match = re.search(r'eligo[- ]?tower[- ]?([fgh])', site, re.IGNORECASE)
            if eligo_match:
                letter = eligo_match.group(1).upper()
                return f"Eligo-Tower-{letter}"
            
            # Handle Tower-* patterns
            tower_match = re.search(r'(?:^|[^a-z])tower[- ]?([fgh])', site, re.IGNORECASE)
            if tower_match:
                letter = tower_match.group(1).upper()
                return f"Eligo-Tower-{letter}"
            
            # Handle single letters F, G, H
            letter_match = re.search(r'^([fgh])$', site, re.IGNORECASE)
            if letter_match:
                letter = letter_match.group(1).upper()
                return f"Eligo-Tower-{letter}"
            
            # Default to Common_Area for anything else
            return "Common_Area"

        

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
        
        standard_sites = ["Eligo-Tower-F", "Eligo-Tower-G", "Eligo-Tower-H", "Common_Area"]
        
        site_mapping = {k: normalize_site_name(k) for k in (set(resolved_sites.keys()) | set(open_sites.keys()))}
        sorted_sites = sorted(standard_sites, key=lambda x: (x != "Common_Area", x))
        
        worksheet.merge_range('A1:H1', report_title_ncr, title_format)
        row = 1
        worksheet.write(row, 0, 'Site', header_format)
        worksheet.merge_range(row, 1, row, 3, 'NCR resolved beyond 21 days', header_format)
        worksheet.merge_range(row, 4, row, 6, 'NCR open beyond 21 days', header_format)
        worksheet.write(row, 7, 'Total', header_format)
        
        row = 2
        categories = ['Civil Finishing', 'Structure Works', 'MEP']
        worksheet.write(row, 0, '', header_format)
        for i, cat in enumerate(categories):
            worksheet.write(row, i+1, cat, subheader_format)
        for i, cat in enumerate(categories):
            worksheet.write(row, i+4, cat, subheader_format)
        worksheet.write(row, 7, '', header_format)
        
        category_map = {
            'Civil Finishing': ['FW', 'Civil Finishing', 'Finishing', 'finishing'],
            'Structure Works': ['SW', 'Structure Works', 'Works', 'Structure', 'works'],
            'MEP': ['MEP', 'mep', 'EL', 'HSE']
        }
        
        def process_module_data(site_data, tower_name):
            """Process module-based data structure"""
            module_counts = {level: {'Civil Finishing': 0, 'Structure Works': 0, 'MEP': 0} 
                           for level in module_levels.get(tower_name, [])}
            
            if not site_data or not isinstance(site_data, dict):
                return module_counts
                
            disciplines = site_data.get("Discipline", [])
            modules = site_data.get("Modules", [])
            
            for i, discipline in enumerate(disciplines):
                if i < len(modules):
                    module_list = modules[i] if isinstance(modules[i], list) else [modules[i]]
                    
                    # Map discipline to category
                    cat = None
                    if any(d in discipline for d in category_map['Civil Finishing']):
                        cat = 'Civil Finishing'
                    elif any(d in discipline for d in category_map['Structure Works']):
                        cat = 'Structure Works'
                    elif any(d in discipline for d in category_map['MEP']):
                        cat = 'MEP'
                    if not cat:
                        continue
                    
                    for module in module_list:
                        normalized_module = module
                        if module not in module_levels.get(tower_name, []):
                            if "common" in module.lower():
                                normalized_module = "Common Description"
                            else:
                                continue
                        module_counts[normalized_module][cat] += 1
            
            return module_counts
        
        row = 3
        site_totals = {}
        
        for site in sorted_sites:
            original_resolved_key = next((k for k, v in site_mapping.items() if v == site), None)
            original_open_key = next((k for k, v in site_mapping.items() if v == site), None)
            
            formats = tower_formats.get(site, {})
            tower_total_format = formats.get('tower_total', workbook.add_format({
                'bold': True, 'align': 'left', 'valign': 'vcenter', 'border': 1, 'fg_color': '#D3D3D3'
            }))
            site_format = formats.get('site', default_site_format)
            cell_format = formats.get('cell', default_cell_format)
            
            resolved_counts = {'Civil Finishing': 0, 'Structure Works': 0, 'MEP': 0}
            open_counts = {'Civil Finishing': 0, 'Structure Works': 0, 'MEP': 0}
            
            if "Tower" in site and site != "Common_Area":
                resolved_module_counts = process_module_data(
                    resolved_sites.get(original_resolved_key, {}) if original_resolved_key else {},
                    site
                )
                open_module_counts = process_module_data(
                    open_sites.get(original_open_key, {}) if original_open_key else {},
                    site
                )
                
                for cat in categories:
                    resolved_counts[cat] = sum(resolved_module_counts[level][cat] for level in module_levels.get(site, []))
                    open_counts[cat] = sum(open_module_counts[level][cat] for level in module_levels.get(site, []))
            
            elif site == "Common_Area":
                resolved_common = resolved_sites.get(original_resolved_key, {}) if original_resolved_key else {}
                open_common = open_sites.get(original_open_key, {}) if original_open_key else {}
                
                disciplines_resolved = resolved_common.get("Discipline", [])
                disciplines_open = open_common.get("Discipline", [])
                
                for discipline in disciplines_resolved:
                    if any(d in discipline for d in category_map['Civil Finishing']):
                        resolved_counts['Civil Finishing'] += 1
                    elif any(d in discipline for d in category_map['Structure Works']):
                        resolved_counts['Structure Works'] += 1
                    elif any(d in discipline for d in category_map['MEP']):
                        resolved_counts['MEP'] += 1
                
                for discipline in disciplines_open:
                    if any(d in discipline for d in category_map['Civil Finishing']):
                        open_counts['Civil Finishing'] += 1
                    elif any(d in discipline for d in category_map['Structure Works']):
                        open_counts['Structure Works'] += 1
                    elif any(d in discipline for d in category_map['MEP']):
                        open_counts['MEP'] += 1
            
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
                for level in module_levels.get(site, []):
                    level_total = sum(resolved_module_counts.get(level, {}).values()) + sum(open_module_counts.get(level, {}).values())
                    worksheet.write(row, 0, level, site_format)
                    for i, display_cat in enumerate(categories):
                        worksheet.write(row, i+1, resolved_module_counts.get(level, {}).get(display_cat, 0), cell_format)
                    for i, display_cat in enumerate(categories):
                        worksheet.write(row, i+4, open_module_counts.get(level, {}).get(display_cat, 0), cell_format)
                    worksheet.write(row, 7, level_total, cell_format)
                    row += 1

        # Combined NCR Detail Sheets
        def write_detail_sheet(sheet_name, data, title):
            truncated_sheet_name = truncate_sheet_name(f"{sheet_name} {date_part}")
            detail_worksheet = workbook.add_worksheet(truncated_sheet_name)
            detail_worksheet.set_column('A:A', 20)
            detail_worksheet.set_column('B:B', 60)
            detail_worksheet.set_column('C:D', 20)
            detail_worksheet.set_column('E:E', 15)
            detail_worksheet.set_column('F:G', 15)
            detail_worksheet.merge_range('A1:G1', f"{title} {date_part}", title_format)
            headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline', 'Modules']
            for col, header in enumerate(headers):
                detail_worksheet.write(1, col, header, header_format)
            row = 2
            
            if not data or not isinstance(data, dict):
                detail_worksheet.write(row, 0, "No data available", default_site_format)
                for col in range(1, 7):
                    detail_worksheet.write(row, col, "", default_cell_format)
                return
            
            for site, site_data in data.items():
                normalized_site = site_mapping.get(site, site)
                
                if not isinstance(site_data, dict):
                    detail_worksheet.write(row, 0, normalized_site, default_site_format)
                    detail_worksheet.write(row, 1, "Data format error - not a dictionary", default_cell_format)
                    for col in range(2, 7):
                        detail_worksheet.write(row, col, "", default_cell_format)
                    row += 1
                    continue
                
                descriptions = site_data.get("Descriptions", [])
                created_dates = site_data.get("Created Date (WET)", [])
                close_dates = site_data.get("Expected Close Date (WET)", [])
                statuses = site_data.get("Status", [])
                disciplines = site_data.get("Discipline", [])
                modules = site_data.get("Modules", [])
                
                max_length = max(
                    len(descriptions) if descriptions else 0,
                    len(created_dates) if created_dates else 0,
                    len(close_dates) if close_dates else 0,
                    len(statuses) if statuses else 0,
                    len(disciplines) if disciplines else 0,
                    len(modules) if modules else 0,
                    1  # Minimum 1 row
                )
                
                for i in range(max_length):
                    detail_worksheet.write(row, 0, normalized_site, default_site_format)
                    detail_worksheet.write(row, 1, str(descriptions[i]) if i < len(descriptions) else "", description_format)
                    detail_worksheet.write(row, 2, str(created_dates[i]) if i < len(created_dates) else "", default_cell_format)
                    detail_worksheet.write(row, 3, str(close_dates[i]) if i < len(close_dates) else "", default_cell_format)
                    detail_worksheet.write(row, 4, str(statuses[i]) if i < len(statuses) else "", default_cell_format)
                    detail_worksheet.write(row, 5, str(disciplines[i]) if i < len(disciplines) else "", default_cell_format)
                    modules_str = ', '.join(map(str, modules[i])) if i < len(modules) and modules[i] else ""
                    detail_worksheet.write(row, 6, modules_str, default_cell_format)
                    row += 1

        # Always create the detail sheets
        write_detail_sheet("Closed NCR Details", resolved_sites, "Closed NCR Details")
        write_detail_sheet("Open NCR Details", open_sites, "Open NCR Details")

        # 2. Safety and Housekeeping Reports
        def write_safety_housekeeping_report(report_type, data, report_title, sheet_type):
            # Summary Sheet
            worksheet = workbook.add_worksheet(truncate_sheet_name(f'{report_type} NCR {sheet_type} {date_part}'))
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:B', 15)
            worksheet.merge_range('A1:B1', f"{report_title} - {sheet_type}", title_format)
            row = 1
            worksheet.write(row, 0, 'Site', header_format)
            worksheet.write(row, 1, f'No. of {report_type} NCRs beyond 7 days', header_format)
            
            # FIXED: Updated to use the same normalize_site_name function
            sites_data = data.get(report_type, {}).get("Sites", {})
            site_mapping_safety = {k: normalize_site_name(k) for k in sites_data.keys()}
            row = 2
            site_counts = {site: 0 for site in sorted_sites}  # Initialize counts for all sites
            
            # FIXED: Properly aggregate counts for normalized sites
            for original_site, site_data in sites_data.items():
                normalized_site = normalize_site_name(original_site)
                if normalized_site in site_counts:
                    site_counts[normalized_site] += site_data.get("Count", 0)
            
            for site in sorted_sites:
                worksheet.write(row, 0, site, default_site_format)
                worksheet.write(row, 1, site_counts[site], default_cell_format)
                row += 1

            # Details Sheet
            worksheet_details = workbook.add_worksheet(truncate_sheet_name(f'{report_type} NCR {sheet_type} Details {date_part}'))
            worksheet_details.set_column('A:A', 20)
            worksheet_details.set_column('B:B', 60)
            worksheet_details.set_column('C:D', 20)
            worksheet_details.set_column('E:E', 15)
            worksheet_details.set_column('F:G', 15)
            worksheet_details.merge_range('A1:G1', f"{report_title} - {sheet_type} Details", title_format)
            headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline', 'Modules']
            row = 1
            for col, header in enumerate(headers):
                worksheet_details.write(row, col, header, header_format)
            row = 2

            # FIXED: Sort sites to match the summary sheet order and use normalized names
            sorted_site_keys = sorted(sites_data.keys(), key=lambda x: sorted_sites.index(normalize_site_name(x)) if normalize_site_name(x) in sorted_sites else len(sorted_sites))
            for original_site in sorted_site_keys:
                normalized_site = normalize_site_name(original_site)
                site_data = sites_data[original_site]
                descriptions = site_data.get("Descriptions", [])
                created_dates = site_data.get("Created Date (WET)", [])
                close_dates = site_data.get("Expected Close Date (WET)", [])
                statuses = site_data.get("Status", [])
                modules = site_data.get("Modules", [[] for _ in range(len(descriptions))])  # Default empty modules
                disciplines = site_data.get("Discipline", ["HSE"] * len(descriptions))  # Default to HSE for Safety NCRs
                max_length = max(len(descriptions), len(created_dates), len(close_dates), len(statuses), len(modules), len(disciplines))
                
                for i in range(max_length):
                    worksheet_details.write(row, 0, normalized_site, default_site_format)  # Use normalized site name
                    description = descriptions[i] if i < len(descriptions) else ""
                    worksheet_details.write(row, 1, description, description_format)
                    worksheet_details.write(row, 2, created_dates[i] if i < len(created_dates) else "", default_cell_format)
                    worksheet_details.write(row, 3, close_dates[i] if i < len(close_dates) else "", default_cell_format)
                    worksheet_details.write(row, 4, statuses[i] if i < len(statuses) else "", default_cell_format)
                    worksheet_details.write(row, 5, disciplines[i] if i < len(disciplines) else "HSE", default_cell_format)
                    modules_str = ', '.join(map(str, modules[i])) if i < len(modules) and modules[i] else ""
                    worksheet_details.write(row, 6, modules_str, default_cell_format)
                    row += 1
            
            if row == 2:  # No data was written
                worksheet_details.write(row, 0, "No data available", default_site_format)
                for col in range(1, 7):
                    worksheet_details.write(row, col, "", default_cell_format)

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

@st.cache_data
def generate_consolidated_ncr_Safety_excel(combined_result, report_title=None):
    """Generate an Excel file for Safety NCR report."""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Define cell formats
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

            # Generate report title with current date
            now = datetime.now()
            day = now.strftime("%d")
            month_name = now.strftime("%B")
            year = now.strftime("%Y")
            date_part = f"{month_name} {day}, {year}"
            if report_title is None:
                report_title = f"Safety: {date_part} - Current Month"
            else:
                report_title = f"{date_part}: Safety"

            # Truncate sheet names to fit Excel's 31-character limit
            def truncate_sheet_name(base_name, max_length=31):
                base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
                if len(base_name) > max_length:
                    return base_name[:max_length - 3] + "..."
                return base_name

            summary_sheet_name = truncate_sheet_name(f'Safety NCR Report {date_part}')
            details_sheet_name = truncate_sheet_name(f'Safety NCR Details {date_part}')

            # Summary Worksheet
            worksheet_summary = workbook.add_worksheet(summary_sheet_name)
            worksheet_summary.set_column('A:A', 20)
            worksheet_summary.set_column('B:B', 15)

            data = combined_result.get("Safety", {}).get("Sites", {})
            all_sites = list(data.keys())  # Use all available site keys from the data

            # Write summary sheet
            worksheet_summary.merge_range('A1:B1', report_title, title_format)
            row = 1
            worksheet_summary.write(row, 0, 'Site', header_format)
            worksheet_summary.write(row, 1, 'No. of Safety NCRs beyond 7 days', header_format)

            row = 2
            for site in sorted(all_sites):  # Sort all sites from the data
                worksheet_summary.write(row, 0, site, site_format)
                value = data.get(site, {}).get("Count", 0)
                worksheet_summary.write(row, 1, value, cell_format)
                row += 1

            # Details Worksheet
            worksheet_details = workbook.add_worksheet(details_sheet_name)
            worksheet_details.set_column('A:A', 20)
            worksheet_details.set_column('B:B', 60)
            worksheet_details.set_column('C:D', 20)
            worksheet_details.set_column('E:E', 15)
            worksheet_details.set_column('F:G', 15)

            worksheet_details.merge_range('A1:G1', f"{report_title} - Details", title_format)

            headers = ['Site', 'Description', 'Created Date (WET)', 'Expected Close Date (WET)', 'Status', 'Discipline', 'Modules']
            row = 1
            for col, header in enumerate(headers):
                worksheet_details.write(row, col, header, header_format)

            row = 2
            for site in sorted(all_sites):
                if site in data:
                    site_data = data[site]
                    descriptions = site_data.get("Descriptions", [])
                    created_dates = site_data.get("Created Date (WET)", [])
                    close_dates = site_data.get("Expected Close Date (WET)", [])
                    statuses = site_data.get("Status", [])
                    modules = site_data.get("Modules", [])  # Handle missing Modules
                    max_length = max(len(lst) for lst in [descriptions, created_dates, close_dates, statuses, modules] if lst) or 1
                    for i in range(max_length):
                        worksheet_details.write(row, 0, site, site_format)
                        worksheet_details.write(row, 1, descriptions[i] if i < len(descriptions) else "", description_format)
                        worksheet_details.write(row, 2, created_dates[i] if i < len(created_dates) else "", cell_format)
                        worksheet_details.write(row, 3, close_dates[i] if i < len(close_dates) else "", cell_format)
                        worksheet_details.write(row, 4, statuses[i] if i < len(statuses) else "", cell_format)
                        worksheet_details.write(row, 5, "HSE", cell_format)
                        module_value = ', '.join(modules[i]) if i < len(modules) and isinstance(modules[i], list) else modules[i] if i < len(modules) else ""
                        worksheet_details.write(row, 6, module_value, cell_format)
                        row += 1

        output.seek(0)
        return output.getvalue()

    except Exception as e:
        st.error(f"❌ Error generating Excel: {str(e)}")
        return None



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



