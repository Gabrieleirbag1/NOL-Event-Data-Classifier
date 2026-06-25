import os
from supervised_clustering import run_matching_for_all_models

def load_events_from_data_dir():
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "..","data")
    event_list = []

    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in lines:
            value = line.split(",")[-1].strip().strip('"')
            if "Events" in value or "Event" in value:
                continue
            if value:
                event_list.append(value)
    return event_list


if __name__ == "__main__":
    event_list_raw = load_events_from_data_dir()
    print(f"{len(event_list_raw)} events loaded")

    if not event_list_raw:
        print("No events loaded, exiting.")
        raise SystemExit(0)

    all_results = run_matching_for_all_models(event_list_raw)