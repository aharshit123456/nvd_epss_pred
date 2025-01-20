import json
import os
import requests
from typing import List, Tuple

def fetch_and_process_cve_data(years: List[int], save_dir: str = "./cve_data") -> Tuple[List[str], List[float], List[float]]:
    """
    Fetch, download, and process CVE data for given years, removing duplicates.

    Args:
        years (List[int]): List of years to fetch CVE data for.
        save_dir (str): Directory to save the downloaded files. Default is "./cve_data".

    Returns:
        Tuple[List[str], List[float], List[float]]:
            - description: List of CVE descriptions.
            - cvss_scores: List of CVSS base scores.
            - targets: List of exploitability scores.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize output lists
    descriptions = []
    cvss_scores = []
    targets = []
    seen_cve_ids = set()  # To filter duplicates
    
    for year in years:
        # Generate download link
        url = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
        local_file = os.path.join(save_dir, f"nvdcve-1.1-{year}.json.gz")
        
        # Download and decompress the file if not already done
        if not os.path.exists(local_file):
            print(f"Downloading CVE data for {year}...")
            response = requests.get(url, stream=True)
            with open(local_file, "wb") as f:
                f.write(response.content)
        
        # Decompress and load the JSON data
        with gzip.open(local_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
        
        # Process each CVE entry
        for item in data["CVE_Items"]:
            cve_id = item["cve"]["CVE_data_meta"]["ID"]
            if cve_id not in seen_cve_ids and item.get("impact"):
                seen_cve_ids.add(cve_id)  # Add CVE ID to seen set
                
                try:
                    # Extract required fields
                    base_score = item["impact"]["baseMetricV3"]["cvssV3"]["baseScore"]
                    exploitability_score = item["impact"]["baseMetricV3"]["exploitabilityScore"]
                    description = item["cve"]["description"]["description_data"][0]["value"]
                    
                    # Append to lists
                    descriptions.append(description)
                    cvss_scores.append(base_score)
                    targets.append(exploitability_score)
                except KeyError:
                    # Skip incomplete CVE records
                    continue
    
    print(f"Processed {len(descriptions)} unique CVE entries.")
    return descriptions, cvss_scores, targets
