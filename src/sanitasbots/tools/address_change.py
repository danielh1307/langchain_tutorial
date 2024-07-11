import os

import requests
from langchain_core.tools import tool


@tool
def address_change(street: str):
    """Allows the user to change his address"""
    print("*** Here we can do an address change to new street ...", street)
    return "{'status': 'success'}"


@tool
def get_coverage(topic: str):
    """Checks whether a specific topic / benefit is covered by the current insurance products.
    The topic must be mapped according to these rules:
    Everything related to vision (glasses, contact lenses) maps to glasses.
    Everything related to food delivery maps to fooddelivery."""
    url = "https://api3.sanitas.com/coveragecheck/v3/topic-coverage/" + topic
    print("Call " + url)
    headers = {
        "Authorization": f"Bearer {os.environ['SANITAS_TOKEN']}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()
