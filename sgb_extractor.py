#!/usr/bin/env python3
"""
Module: sgb_extractor
Description: Contains functions to extract Sovereign Gold Bond series data
             from the Wikipedia page using BeautifulSoup.
"""

import requests
from bs4 import BeautifulSoup


def fetch_wikipedia_page(url: str) -> str:
    """
    Fetches the HTML content from the given URL.
    
    Args:
        url (str): URL to fetch.
    
    Returns:
        str: HTML content of the page.
    
    Raises:
        requests.HTTPError: if the request fails.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_table(table) -> list:
    """
    Parses an HTML table into a list of dictionaries.
    
    Args:
        table (bs4.element.Tag): A BeautifulSoup Tag representing the table.
    
    Returns:
        list: A list of dictionaries, each representing a row.
    """
    headers = []
    header_row = table.find('tr')
    if header_row:
        for th in header_row.find_all('th'):
            headers.append(th.get_text(strip=True))
    
    rows = []
    for tr in table.find_all('tr')[1:]:
        cols = tr.find_all(['td', 'th'])
        if not cols:
            continue
        row_data = {}
        for i, col in enumerate(cols):
            header = headers[i] if i < len(headers) else f"Column{i+1}"
            row_data[header] = col.get_text(strip=True)
        rows.append(row_data)
    return rows


def find_sgb_series_table(soup: BeautifulSoup) -> list:
    """
    Searches for the wikitable that contains the SGB series data by checking
    for a header containing the word 'Tranche'.
    
    Args:
        soup (BeautifulSoup): Parsed HTML soup.
    
    Returns:
        list: Parsed table data as a list of dictionaries, or None if not found.
    """
    tables = soup.find_all('table', class_='wikitable')
    for table in tables:
        header_row = table.find('tr')
        if header_row and "Tranche" in header_row.get_text():
            return parse_table(table)
    return None


def extract_sgb_series(url: str = "https://en.wikipedia.org/wiki/Sovereign_Gold_Bond") -> list:
    """
    Extracts the Sovereign Gold Bond series data from the given Wikipedia page.
    
    Args:
        url (str): The URL of the Wikipedia page.
    
    Returns:
        list: List of dictionaries representing each row in the series table.
    """
    html = fetch_wikipedia_page(url)
    soup = BeautifulSoup(html, "html.parser")
    series_table = find_sgb_series_table(soup)
    return series_table


if __name__ == "__main__":
    # Test the module functionality.
    try:
        series = extract_sgb_series()
        if series:
            print("Extracted SGB Series Data:")
            for row in series:
                print(row)
        else:
            print("No SGB series table found on the page.")
    except Exception as e:
        print(f"An error occurred: {e}")
