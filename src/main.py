import os

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

data_path = os.path.join(os.path.dirname(__file__), "..", "data")
files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
for file in files:
    file_path = os.path.join(data_path, file)
    data = read_csv(file_path)
    get_last_column_data(data)

print(event_list)

if __name__ == "__main__":
    pass