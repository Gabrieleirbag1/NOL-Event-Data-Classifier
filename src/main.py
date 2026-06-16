import os

from hdbscan_clustering import run_clustering

event_list = []

def read_csv(file_path):
    with open(file_path, "r") as f:
        data = f.read().splitlines()
    return data

def get_last_column_data(data):
    for line in data:
        event_column = line.split(",")[-1]
        event_column = event_column.strip().strip('"')
        if event_column:
            event_list.append(event_column)

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    i = 0
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
