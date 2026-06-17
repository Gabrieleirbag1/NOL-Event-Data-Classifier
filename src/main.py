import os

import re

from hdbscan_clustering import run_clustering

event_list = {"normalized": [], "content": [], "timestamp": []}

def read_csv(file_path):
    with open(file_path, "r") as f:
        data = f.read().splitlines()
    return data

def normalize_dosages(text):
    text = re.sub(r'\bType\s*(\d+)\b', r'Type [NUM]', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+[\.,]?\d*\s*(mg|ml|g|kg|cc|ui|µg|mcg)\b', '[DOSE]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+[\.,]?\d*\b', '[DOSE]', text)
    return text

def get_last_column_data(data):
    for line in data:
        event_column = line.split(",")[-1]
        event_column = event_column.strip().strip('"')
        if "Events" in event_column or "Event" in event_column:
            continue
        if event_column:
            event_column = event_column.split("@")
            event_list["content"].append(event_column[0])
            event_list["normalized"].append(normalize_dosages(event_column[0]))
            event_list["timestamp"].append(event_column[1] if len(event_column) > 1 else "")

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    i = -12000
    for file in files:
        i+=1
        file_path = os.path.join(data_path, file)
        data = read_csv(file_path)
        get_last_column_data(data)
        if i == 5:  # Limiter à 5 fichiers pour les tests
            break

if __name__ == "__main__":
    main()
    print(event_list)
    run_clustering(event_list)
