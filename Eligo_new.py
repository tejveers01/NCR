"""
NCR Report Generation System - Streamlit Application
Generates Non-Conformance Reports from Asite project data using WatsonX AI
"""

import io
import json
import logging
import os
import re
import urllib.parse
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

import certifi
import pandas as pd
import requests
import streamlit as st
import urllib3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from urllib3.util.retry import Retry

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Environment Variables
WATSONX_API_URL = os.getenv("WATSONX_API_URL")
MODEL_ID = os.getenv("MODEL_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")

# API Endpoints
LOGIN_URL = "https://dms.asite.com/apilogin/"
SEARCH_URL = "https://adoddleak.asite.com/commonapi/formsearchapi/search"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Validate Configuration
if not all([API_KEY, WATSONX_API_URL, MODEL_ID, PROJECT_ID]):
    st.error("❌ Required environment variables missing!")
    st.stop()

# ============================================================================
# AUTHENTICATION
# ============================================================================

def get_access_token(api_key: str) -> Optional[str]:
    """Generate WatsonX API access token"""
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    try:
        response = requests.post(
            IAM_TOKEN_URL,
            headers=headers,
            data=data,
            verify=certifi.where(),
            timeout=50
        )
        if response.status_code == 200:
            logger.info("Access token generated successfully")
            return response.json()['access_token']
        else:
            logger.error(f"Token generation failed: {response.status_code}")
            st.error(f"❌ Failed to get access token: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception getting access token: {str(e)}")
        st.error(f"❌ Error getting access token: {str(e)}")
        return None

def login_to_asite(email: str, password: str) -> Optional[str]:
    """Login to Asite and return session ID"""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {"emailId": email, "password": password}
    try:
        response = requests.post(
            LOGIN_URL,
            headers=headers,
            data=payload,
            verify=certifi.where(),
            timeout=50
        )
        if response.status_code == 200:
            session_id = response.json().get("UserProfile", {}).get("Sessionid")
            logger.info(f"Login successful, Session ID: {session_id}")
            return session_id
        else:
            logger.error(f"Login failed: {response.status_code}")
            st.error(f"❌ Login failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Login exception: {str(e)}")
        st.error(f"❌ Login error: {str(e)}")
        return None

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_project_data(
    session_id: str,
    project_name: str,
    form_name: str,
    record_limit: int = 1000
) -> Tuple[Dict, List[Dict], str]:
    """Fetch project data from Asite API with pagination"""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "Cookie": f"ASessionID={session_id}"
    }
    
    all_data = []
    start_record = 1
    total_records = None
    start_time = datetime.now()
    
    st.write(f"🔄 Fetching data started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Fetching data for project '{project_name}', form '{form_name}'")
    
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
            
            encoded_payload = f"searchCriteria={urllib.parse.quote(json.dumps(search_criteria))}"
            
            try:
                response = requests.post(
                    SEARCH_URL,
                    headers=headers,
                    data=encoded_payload,
                    verify=certifi.where(),
                    timeout=50
                )
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
    duration = (end_time - start_time).total_seconds()
    st.write(f"🔄 Fetching completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ({duration:.0f}s)")
    logger.info(f"Finished fetching data ({duration:.0f}s)")
    
    return {"responseHeader": {"results": len(all_data), "total": total_records}}, all_data, encoded_payload

# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_json_data(json_data: List[Dict]) -> pd.DataFrame:
    """Convert Asite JSON data to DataFrame"""
    data = []
    
    for item in json_data:
        form_details = item.get('FormDetails', {})
        custom_fields = form_details.get('CustomFields', {}).get('CustomField', [])
        
        created_date = form_details.get('FormCreationDate')
        expected_close_date = form_details.get('UpdateDate')
        form_status = form_details.get('FormStatus')
        
        # Extract custom fields
        discipline = None
        description = None
        for field in custom_fields:
            if field.get('FieldName') == 'CFID_DD_DISC':
                discipline = field.get('FieldValue')
            elif field.get('FieldName') == 'CFID_RTA_DES':
                description = BeautifulSoup(
                    field.get('FieldValue', '') or '',
                    "html.parser"
                ).get_text()
        
        # Calculate days difference
        days_diff = None
        if created_date and expected_close_date:
            try:
                created_obj = datetime.strptime(
                    created_date.split('#')[0],
                    "%d-%b-%Y"
                )
                close_obj = datetime.strptime(
                    expected_close_date.split('#')[0],
                    "%d-%b-%Y"
                )
                days_diff = (close_obj - created_obj).days
            except Exception as e:
                logger.error(f"Error calculating days: {str(e)}")
        
        data.append([
            days_diff, created_date, expected_close_date,
            description, form_status, discipline
        ])
    
    df = pd.DataFrame(
        data,
        columns=[
            'Days', 'Created Date (WET)', 'Expected Close Date (WET)',
            'Description', 'Status', 'Discipline'
        ]
    )
    
    # Convert date columns
    df['Created Date (WET)'] = pd.to_datetime(
        df['Created Date (WET)'].str.split('#').str[0],
        format="%d-%b-%Y",
        errors='coerce'
    )
    df['Expected Close Date (WET)'] = pd.to_datetime(
        df['Expected Close Date (WET)'].str.split('#').str[0],
        format="%d-%b-%Y",
        errors='coerce'
    )
    
    if df.empty:
        logger.warning("DataFrame is empty after processing")
        st.warning("⚠️ No data processed")
    
    return df

def clean_and_parse_json(text: str) -> Optional[Dict]:
    """Extract and parse JSON from text response"""
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
            return json.loads(text[start_idx:end_idx+1])
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON")
    
    logger.error(f"Could not extract valid JSON")
    return None

# ============================================================================
# MODULE & TOWER EXTRACTION
# ============================================================================

def extract_modules_from_description(description: str) -> List[str]:
    """Extract module numbers from description text"""
    if not description or not isinstance(description, str):
        return ["Common"]
    
    description_lower = description.lower()
    modules = set()
    
    # Pattern 1: Ranges (Module 1 to 3)
    range_patterns = r"(?:module|mod|m)[-\s]*(\d+)\s*(?:to|-|–)\s*(\d+)"
    for start_str, end_str in re.findall(range_patterns, description_lower):
        try:
            start, end = int(start_str), int(end_str)
            if 0 < start <= end <= 50:
                modules.update(f"Module {i}" for i in range(start, end + 1))
        except ValueError:
            continue
    
    # Pattern 2: Individual modules (Module 3)
    individual_patterns = r"(?:module|mod|m)[-\s]*(\d{1,2})"
    for num_str in re.findall(individual_patterns, description_lower):
        try:
            num = int(num_str)
            if 0 < num <= 50:
                modules.add(f"Module {num}")
        except ValueError:
            continue
    
    return sorted(list(modules)) if modules else ["Common"]

def determine_tower_assignment(description: str) -> str:
    """Determine tower from description"""
    if not description or not isinstance(description, str):
        return "Common_Area"
    
    description_lower = description.lower()
    
    # Check for specific tower indicators
    tower_match = re.search(r"\b(?:tower|t)\s*[-\s(]*([fgh])\b", description_lower, re.IGNORECASE)
    if tower_match:
        tower_letter = tower_match.group(1).upper()
        return f"Eligo-Tower-{tower_letter}"
    
    # Check for common areas
    common_indicators = ["steel yard", "qc lab", "corridor", "staircase"]
    if any(ind in description_lower for ind in common_indicators):
        return "Common_Area"
    
    return "Common_Area"

# ============================================================================
# REPORT GENERATION
# ============================================================================

@st.cache_data
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((requests.RequestException, ValueError))
)
def generate_ncr_report(
    df: pd.DataFrame,
    report_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    until_date: Optional[str] = None
) -> Tuple[Dict[str, Any], str]:
    """Generate NCR report with WatsonX AI"""
    
    if df is None or df.empty:
        st.error("❌ Input DataFrame is empty")
        return {"error": "Empty DataFrame"}, ""
    
    if report_type not in ["Open", "Closed"]:
        st.error(f"❌ Invalid report type: {report_type}")
        return {"error": "Invalid report type"}, ""
    
    df = df.copy()
    df = df[df['Created Date (WET)'].notna()]
    
    # Filter based on report type
    if report_type == "Closed":
        df = df[df['Expected Close Date (WET)'].notna()]
        df['Days'] = (df['Expected Close Date (WET)'] - df['Created Date (WET)']).dt.days
        filtered_df = df[(df['Status'] == 'Closed') & (df['Days'] > 21)].copy()
    else:  # Open
        if until_date is None:
            st.error("❌ Until Date required for Open reports")
            return {"error": "Until Date required"}, ""
        
        today = pd.to_datetime(until_date)
        filtered_df = df[df['Status'] == 'Open'].copy()
        filtered_df['Days_From_Today'] = (today - filtered_df['Created Date (WET)']).dt.days
        filtered_df = filtered_df[filtered_df['Days_From_Today'] > 21].copy()
    
    if filtered_df.empty:
        st.warning(f"No {report_type} NCRs found with duration > 21 days")
        return {"error": "No records found"}, ""
    
    # Convert to string for API
    filtered_df['Created Date (WET)'] = filtered_df['Created Date (WET)'].astype(str)
    if 'Expected Close Date (WET)' in filtered_df.columns:
        filtered_df['Expected Close Date (WET)'] = filtered_df['Expected Close Date (WET)'].astype(str)
    
    processed_data = filtered_df.to_dict(orient="records")
    st.write(f"Processing {len(processed_data)} records")
    logger.info(f"Processing {len(processed_data)} records for {report_type} report")
    
    # Get API token
    access_token = get_access_token(API_KEY)
    if not access_token:
        return {"error": "Failed to obtain access token"}, ""
    
    # Initialize results
    all_results = {report_type: {"Sites": {}, "Grand_Total": 0}}
    chunk_size = int(os.getenv("CHUNK_SIZE", 15))
    total_chunks = (len(processed_data) + chunk_size - 1) // chunk_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    # Process chunks
    for i in range(0, len(processed_data), chunk_size):
        chunk = processed_data[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        progress = min((chunk_num / total_chunks) * 100, 100)
        progress_bar.progress(int(progress))
        status_text.write(f"Processing chunk {chunk_num}/{total_chunks}")
        
        prompt = f"""Generate ONE JSON object for NCR analysis.
Report Type: {report_type}
Total Records: {len(chunk)}

Output Format:
{{
  "{report_type}": {{
    "Sites": {{
      "Tower_Name": {{
        "Descriptions": [],
        "Created_Dates": [],
        "Status": [],
        "Total": 0
      }}
    }},
    "Grand_Total": 0
  }}
}}

Data:
{json.dumps(chunk)}

Return ONLY valid JSON."""
        
        payload = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 2000,
                "temperature": 0.001
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
            response = session.post(
                WATSONX_API_URL,
                headers=headers,
                json=payload,
                verify=certifi.where(),
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("results", [{}])[0].get("generated_text", "")
                parsed_json = clean_and_parse_json(generated_text)
                
                if parsed_json and report_type in parsed_json:
                    chunk_result = parsed_json[report_type]
                    # Merge results
                    for site, data in chunk_result.get("Sites", {}).items():
                        if site not in all_results[report_type]["Sites"]:
                            all_results[report_type]["Sites"][site] = {
                                "Descriptions": [],
                                "Created_Dates": [],
                                "Status": [],
                                "Total": 0
                            }
                        site_data = all_results[report_type]["Sites"][site]
                        site_data["Descriptions"].extend(data.get("Descriptions", []))
                        site_data["Created_Dates"].extend(data.get("Created_Dates", []))
                        site_data["Status"].extend(data.get("Status", []))
                        site_data["Total"] += data.get("Total", 0)
                    
                    all_results[report_type]["Grand_Total"] += chunk_result.get("Grand_Total", 0)
                    st.success(f"✅ Chunk {chunk_num} processed")
        
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
            st.warning(f"⚠️ Error in chunk {chunk_num}")
    
    progress_bar.progress(100)
    status_text.empty()
    
    return all_results, json.dumps(all_results, indent=2)

# ============================================================================
# EXCEL GENERATION
# ============================================================================

def generate_excel_report(results: Dict, report_title: str = "NCR Report") -> io.BytesIO:
    """Generate Excel file from results"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Styles
        title_format = workbook.add_format({
            'bold': True, 'align': 'center', 'bg_color': 'yellow', 'border': 1
        })
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'border': 1, 'text_wrap': True
        })
        cell_format = workbook.add_format({
            'align': 'center', 'border': 1
        })
        
        # Summary sheet
        ws = workbook.add_worksheet('Summary')
        ws.merge_range('A1:D1', report_title, title_format)
        ws.write('A2', 'Site', header_format)
        ws.write('B2', 'Total', header_format)
        
        row = 2
        for site, data in results.get("Sites", {}).items():
            ws.write(row, 0, site)
            ws.write(row, 1, data.get("Total", 0), cell_format)
            row += 1
    
    output.seek(0)
    return output

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="NCR Report Generator", layout="wide")
    st.title("🏗️ NCR Report Generation System")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        email = st.text_input("Asite Email")
        password = st.text_input("Asite Password", type="password")
        project_name = st.text_input("Project Name", value="Eligo")
        form_name = st.text_input("Form Name", value="RTA")
        
        report_type = st.selectbox("Report Type", ["Open", "Closed"])
        until_date = st.date_input("Until Date") if report_type == "Open" else None
        
        if st.button("Generate Report"):
            with st.spinner("Authenticating..."):
                session_id = login_to_asite(email, password)
            
            if session_id:
                with st.spinner("Fetching data..."):
                    _, data, _ = fetch_project_data(session_id, project_name, form_name)
                    df = process_json_data(data)
                
                if not df.empty:
                    with st.spinner("Generating report..."):
                        results, json_str = generate_ncr_report(
                            df,
                            report_type,
                            until_date=str(until_date) if until_date else None
                        )
                    
                    if "error" not in results:
                        st.success("✅ Report generated successfully!")
                        st.json(results)
                        
                        # Download button
                        excel_file = generate_excel_report(results[report_type])
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_file,
                            file_name=f"NCR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()
