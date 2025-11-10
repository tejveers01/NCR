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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Tuple, Dict, Any, List

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

if not all([API_KEY, WATSONX_API_URL, MODEL_ID, PROJECT_ID]):
    st.error("❌ Required environment variables missing!")
    logger.error("Missing required environment variables")
    st.stop()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API Endpoints
LOGIN_URL = "https://dms.asite.com/apilogin/"
SEARCH_URL = "https://adoddleak.asite.com/commonapi/formsearchapi/search"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"


# ============================================================================
# CORE FUNCTIONS - KEEP ORIGINAL LOGIC WITH IMPROVEMENTS
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
            logger.error(f"Failed to get access token: {response.status_code}")
            st.error(f"❌ Failed to get access token: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception getting access token: {str(e)}")
        st.error(f"❌ Error getting access token: {str(e)}")
        return None


def login_to_asite(email, password):
    """Login to Asite and retrieve session ID."""
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"emailId": email, "password": password}
    try:
        response = requests.post(LOGIN_URL, headers=headers, data=payload, verify=certifi.where(), timeout=50)
        if response.status_code == 200:
            session_id = response.json().get("UserProfile", {}).get("Sessionid")
            logger.info(f"Login successful")
            return session_id
        else:
            logger.error(f"Login failed: {response.status_code}")
            st.error(f"❌ Login failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Login exception: {str(e)}")
        st.error(f"❌ Login error: {str(e)}")
        return None


def fetch_project_data(session_id, project_name, form_name, record_limit=1000):
    """Fetch project data from Asite with pagination - preserves ALL records."""
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
    st.write(f"🔄 Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration:.1f}s)")
    logger.info(f"Fetching complete. Total records: {len(all_data)}")

    return {"responseHeader": {"results": len(all_data), "total_results": total_records}}, all_data, encoded_payload


def process_json_data(json_data: List[Dict]) -> pd.DataFrame:
    """Process JSON data from Asite into a DataFrame - NO DEDUPLICATION."""
    data = []
    for idx, item in enumerate(json_data):
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
                logger.error(f"Error calculating days: {str(e)}")
                days_diff = None

        # Add record_id to track original record position
        data.append([idx, days_diff, created_date, expected_close_date, description, form_status, discipline])

    df = pd.DataFrame(data, columns=['Record_ID', 'Days', 'Created Date (WET)', 'Expected Close Date (WET)', 
                                     'Description', 'Status', 'Discipline'])
    df['Created Date (WET)'] = pd.to_datetime(df['Created Date (WET)'].str.split('#').str[0], 
                                              format="%d-%b-%Y", errors='coerce')
    df['Expected Close Date (WET)'] = pd.to_datetime(df['Expected Close Date (WET)'].str.split('#').str[0], 
                                                      format="%d-%b-%Y", errors='coerce')
    
    if df.empty:
        logger.warning("DataFrame is empty after processing")
        st.warning("⚠️ No data processed")
    
    return df


def clean_and_parse_json(text: str) -> Dict:
    """Extract and parse JSON from text response."""
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
            logger.error(f"Failed to parse JSON")
            
    return None


# ============================================================================
# HOUSEKEEPING & SAFETY REPORTS - IMPROVED VERSION
# ============================================================================

@st.cache_data
def generate_ncr_Housekeeping_report_for_eligo(df, report_type, start_date=None, end_date=None, until_date=None):
    """Generate Housekeeping NCR report - PRESERVES ALL OCCURRENCES."""
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
                if not description or not isinstance(description, str):
                    return False
                return any(keyword in description.lower() for keyword in housekeeping_keywords)

            # Filter data
            if report_type == "Closed":
                filtered_df = df[
                    (df['Discipline'] == 'HSE') &
                    (df['Status'] == 'Closed') &
                    (df['Days'].notnull()) &
                    (df['Days'] > 7) &
                    (df['Description'].apply(is_housekeeping_record))
                ].copy()
            else:
                open_records = df[(df['Status'] == 'Open') & 
                                 (pd.to_datetime(df['Created Date (WET)']).notna())].copy()
                open_records.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(
                    open_records['Created Date (WET)'])).dt.days
                filtered_df = open_records[
                    (open_records['Days_From_Today'] > 7) &
                    (open_records['Discipline'] == 'HSE') &
                    (open_records['Description'].apply(is_housekeeping_record))
                ].copy()

            if filtered_df.empty:
                return {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}, ""

            result = {"Housekeeping": {"Sites": {}, "Grand_Total": 0}}
            
            # Process ALL records without deduplication
            for idx, row in filtered_df.iterrows():
                description = str(row.get("Description", "")).strip()
                site = "Common_Area"
                
                tower_match = re.search(r"(tower|t)\s*-?\s*([A-Za-z])", description, re.IGNORECASE)
                if tower_match:
                    site = f"Eligo-Tower-{tower_match.group(2).upper()}"
                
                if site not in result["Housekeeping"]["Sites"]:
                    result["Housekeeping"]["Sites"][site] = {
                        "Count": 0,
                        "Descriptions": [],
                        "Created Date (WET)": [],
                        "Expected Close Date (WET)": [],
                        "Status": [],
                        "Record_IDs": []
                    }
                
                result["Housekeeping"]["Sites"][site]["Descriptions"].append(description)
                result["Housekeeping"]["Sites"][site]["Created Date (WET)"].append(str(row.get("Created Date (WET)", "")))
                result["Housekeeping"]["Sites"][site]["Expected Close Date (WET)"].append(str(row.get("Expected Close Date (WET)", "")))
                result["Housekeeping"]["Sites"][site]["Status"].append(str(row.get("Status", "")))
                result["Housekeeping"]["Sites"][site]["Record_IDs"].append(row.get("Record_ID", idx))
                result["Housekeeping"]["Sites"][site]["Count"] += 1
                result["Housekeeping"]["Grand_Total"] += 1

            st.write(f"Total {report_type} records processed: {result['Housekeeping']['Grand_Total']} (All duplicates preserved)")
            logger.info(f"Processed {result['Housekeeping']['Grand_Total']} records with duplicates preserved")
            
            return result, json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Error in Housekeeping report: {str(e)}")
            st.error(f"❌ Error: {str(e)}")
            return {"error": str(e)}, ""


@st.cache_data
def generate_ncr_Safety_report_for_eligo(df, report_type, start_date=None, end_date=None, until_date=None):
    """Generate Safety NCR report - PRESERVES ALL OCCURRENCES."""
    with st.spinner(f"Generating {report_type} Safety NCR Report..."):
        try:
            today = pd.to_datetime(datetime.today().strftime('%Y/%m/%d'))
            
            safety_keywords = [
                'safety precautions', 'temporary electricity', 'safety norms', 'safety belt', 'helmet',
                'lifeline', 'guard rails', 'fall protection', 'PPE', 'electrical hazard', 'unsafe platform',
                'catch net', 'edge protection', 'TPI', 'scaffold', 'lifting equipment', 'dust suppression',
                'debris chute', 'spill control', 'crane operator', 'halogen lamps', 'fall catch net',
                'environmental contamination', 'fire hazard', 'working at height', 'PPE kit', 'HSE norms',
                'negligence in supervision', 'violation of HSE', 'tower h', 'non-tower area', 'nta'
            ]

            def is_safety_record(description):
                if not description or not isinstance(description, str):
                    return False
                return any(keyword in description.lower() for keyword in safety_keywords)

            # Filter data
            if report_type == "Closed":
                filtered_df = df[
                    (df['Discipline'] == 'HSE') &
                    (df['Status'] == 'Closed') &
                    (df['Days'].notnull()) &
                    (df['Days'] > 7) &
                    (df['Description'].apply(is_safety_record))
                ].copy()
            else:
                open_records = df[(df['Status'] == 'Open') & 
                                 (pd.to_datetime(df['Created Date (WET)']).notna())].copy()
                open_records.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(
                    open_records['Created Date (WET)'])).dt.days
                filtered_df = open_records[
                    (open_records['Days_From_Today'] > 7) &
                    (open_records['Discipline'] == 'HSE') &
                    (open_records['Description'].apply(is_safety_record))
                ].copy()

            if filtered_df.empty:
                return {"Safety": {"Sites": {}, "Grand_Total": 0}}, ""

            result = {"Safety": {"Sites": {}, "Grand_Total": 0}}
            
            # Process ALL records without deduplication
            for idx, row in filtered_df.iterrows():
                description = str(row.get("Description", "")).strip()
                site = "Common_Area"
                
                tower_match = re.search(r"(tower|t)\s*-?\s*([A-Za-z])", description, re.IGNORECASE)
                if tower_match:
                    site = f"Eligo-Tower-{tower_match.group(2).upper()}"
                
                if site not in result["Safety"]["Sites"]:
                    result["Safety"]["Sites"][site] = {
                        "Count": 0,
                        "Descriptions": [],
                        "Created Date (WET)": [],
                        "Expected Close Date (WET)": [],
                        "Status": [],
                        "Record_IDs": []
                    }
                
                result["Safety"]["Sites"][site]["Descriptions"].append(description)
                result["Safety"]["Sites"][site]["Created Date (WET)"].append(str(row.get("Created Date (WET)", "")))
                result["Safety"]["Sites"][site]["Expected Close Date (WET)"].append(str(row.get("Expected Close Date (WET)", "")))
                result["Safety"]["Sites"][site]["Status"].append(str(row.get("Status", "")))
                result["Safety"]["Sites"][site]["Record_IDs"].append(row.get("Record_ID", idx))
                result["Safety"]["Sites"][site]["Count"] += 1
                result["Safety"]["Grand_Total"] += 1

            st.write(f"Total {report_type} records processed: {result['Safety']['Grand_Total']} (All duplicates preserved)")
            logger.info(f"Processed {result['Safety']['Grand_Total']} records with duplicates preserved")
            
            return result, json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Error in Safety report: {str(e)}")
            st.error(f"❌ Error: {str(e)}")
            return {"error": str(e)}, ""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_report_title(prefix):
    """Generate timestamped report title."""
    now = datetime.now()
    return f"{prefix}: {now.strftime('%d_%B_%Y')}"


def show_duplicate_analysis(result_data: Dict, report_type: str):
    """Display duplicate analysis in Streamlit UI."""
    if not result_data or "error" in result_data:
        return
    
    sites = result_data.get(report_type, {}).get("Sites", {})
    
    analysis_data = []
    for site, site_data in sites.items():
        total_count = site_data.get("Count", 0)
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


# ============================================================================
# EXPORT FUNCTION
# ============================================================================

def export_results_to_excel(result_data: Dict, report_name: str) -> BytesIO:
    """Export results to Excel with duplicate tracking."""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formatting
        title_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': 'yellow', 
            'border': 1, 'font_size': 12
        })
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        data_format = workbook.add_format({
            'align': 'left', 'valign': 'vcenter', 'border': 1, 'text_wrap': True
        })
        
        # Summary Sheet
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
                total = site_data.get("Count", 0)
                unique = len(set(site_data.get("Descriptions", [])))
                dupes = total - unique
                dup_pct = f"{(dupes/total*100):.1f}%" if total > 0 else "0%"
                
                ws_summary.write(row, 0, site, data_format)
                ws_summary.write(row, 1, total, data_format)
                ws_summary.write(row, 2, unique, data_format)
                ws_summary.write(row, 3, dupes, data_format)
                ws_summary.write(row, 4, dup_pct, data_format)
                row += 1
        
        # Detail Sheet
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
                
                for i in range(len(descriptions)):
                    ws_detail.write(row, 0, record_num, data_format)
                    ws_detail.write(row, 1, site, data_format)
                    ws_detail.write(row, 2, descriptions[i], data_format)
                    ws_detail.write(row, 3, created[i] if i < len(created) else "", data_format)
                    ws_detail.write(row, 4, close_date[i] if i < len(close_date) else "", data_format)
                    ws_detail.write(row, 5, status[i] if i < len(status) else "", data_format)
                    ws_detail.write(row, 6, record_ids[i] if i < len(record_ids) else "", data_format)
                    row += 1
                    record_num += 1
    
    output.seek(0)
    return output
