"""
symbols_from_gdoc.py
--------------------
Reads a public Google Doc containing a 3-column table:
    | x-coord | symbol | y-coord |
    |---------|--------|---------|
    | 1       | ★      | 2       |
    | 3       | ●      | 5       |

Usage:
    python symbols_from_gdoc.py --doc_url YOUR_doc_url
    python symbols_from_gdoc.py --doc_url "https://docs.google.com/document/d/YOUR_ID/edit"
"""
import argparse
import sys
from typing import Any

import numpy as np
from bs4 import BeautifulSoup
import requests
from numpy import dtype, ndarray


def fetch_doc_as_html(doc_url: str) -> str:
    try:
        print("--- Fetching document as HTML ---")
        response = requests.get(doc_url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"✗ Failed to fetch doc: {e}")
        print("  Make sure the doc is shared as 'Anyone with the link can view'.")
        sys.exit(1)

def create_canvas_from_table(data: np.ndarray) -> list[list[str]]:

    # Convert coordinates as points(x,y,char)
    points = [
        (int(row[0]), int(row[2]), row[1])
        for row in data
    ]

    # Grid size
    max_x = max(x for x, y, c in points)
    max_y = max(y for x, y, c in points)

    # Create empty canvas
    canvas = [[' ' for _ in range(max_x + 1)]
              for _ in range(max_y + 1)]

    # Fill canvas
    for x, y, char in points:
        canvas[y][x] = char
    return canvas

def parse_table_in_html(html: str) -> ndarray[tuple[Any, ...], dtype[_ScalarT]] | None:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    data = []
    if table:
        print("--- Reading Table Rows ---")
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            values = [col.text for col in cols]
            data.append(values)
        #skip the header row
        # matrix = np.array(data[1:])
        return np.array(data[1:])
    
    else:
        print("Could not find any <table> element on this page.")
        sys.exit(1)


def print_canvas(canvas):
    # Print rows in reverse order because to represent cartesian coordinates x,y
    #where (0,0) is at the bottom of the output
    print("*** Start Canvas ***")
    for row in reversed(canvas):
        print(''.join(row))
    print("*** End Canvas ***")



def main():
    parser = argparse.ArgumentParser(description="Prints symbols from a public Google Doc table")
    parser.add_argument("--doc_url", help="Google Doc full URL")
    args = parser.parse_args()

    if args.doc_url:
        html = fetch_doc_as_html(args.doc_url)
        ndarray_table = parse_table_in_html(html)
        canvas = create_canvas_from_table(ndarray_table)
        print_canvas(canvas)
        print("--- Program executed successfully ---")
        sys.exit(0)
    else:
        print("No --doc_url provided. Using built-in fallback symbols.")
        sys.exit(1)


if __name__ == "__main__":
    main()