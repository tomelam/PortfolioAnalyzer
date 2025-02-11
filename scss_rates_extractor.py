#!/usr/bin/env python3
"""
Module: scss_rates_extractor.py
Extracts the SCSS interest rate series from the NSI India website.
"""

import requests
from bs4 import BeautifulSoup
import urllib3

# Suppress SSL certificate warnings.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_html(url, verify_ssl=False):
    """
    Fetch HTML content from the specified URL.
    """
    response = requests.get(url, verify=verify_ssl)
    response.raise_for_status()
    return response.text

def find_target_table(soup):
    """
    Look for a container (either a <table> or a <tbody>) whose first row has
    at least two cells with the first cell equal to "YEAR" and the second cell containing "INTEREST".
    """
    for container in soup.find_all(['table', 'tbody']):
        first_row = container.find('tr')
        if first_row:
            cells = first_row.find_all(['td', 'th'])
            if len(cells) >= 2:
                header1 = cells[0].get_text(strip=True).lower()
                header2 = cells[1].get_text(strip=True).lower()
                if header1 == "year" and "interest" in header2:
                    return container
    return None

def extract_rate_series(html):
    """
    Extracts the interest rate series from the identified table.
    Returns:
        A list of dictionaries (one per row) using the first row as header keys.
    """
    soup = BeautifulSoup(html, 'html.parser')
    target_table = find_target_table(soup)
    if not target_table:
        raise ValueError("Target table not found.")
    
    header_row = target_table.find('tr')
    headers = [cell.get_text(strip=True) for cell in header_row.find_all(['td', 'th'])]
    
    data = []
    for row in target_table.find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])
        cells_text = [cell.get_text(strip=True) for cell in cells]
        if len(cells_text) < len(headers):
            cells_text.extend([""] * (len(headers) - len(cells_text)))
        else:
            cells_text = cells_text[:len(headers)]
        data.append(dict(zip(headers, cells_text)))
    return data

def main():
    url = "https://www.nsiindia.gov.in/(S(2xgxs555qwdlfb2p4ub03n3n))/InternalPage.aspx?Id_Pk=181"
    try:
        html = fetch_html(url, verify_ssl=False)
        rate_series = extract_rate_series(html)
        for row in rate_series:
            print(row)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
