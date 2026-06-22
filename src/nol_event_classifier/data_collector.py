import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..","data")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..","output")

def load_events_from_data_dir():
    event_list = []

    files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    for file in files:
        file_path = os.path.join(DATA_PATH, file)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in lines:
            value = line.split(",")[-1].strip().strip('"')
            if "Events" in value or "Event" in value:
                continue
            if value:
                event_list.append(value)
    return event_list

def clean_event_list(event_list):
    #remove special caracters and numbers
    cleaned_list = []
    for event in event_list:
        cleaned_event = ''.join(c for c in event if c.isalpha() or c.isspace())
        cleaned_list.append(cleaned_event)
    return cleaned_list

def collect_unique_words_from_events(event_list):
    unique_words = set()
    for event in event_list:
        words = event.split()
        unique_words.update(words)
    return unique_words

def clean_unique_words(unique_words):
    cleaned_words = set()
    for word in unique_words:
        if "libreoffice" in word.lower() or len(word) <= 1:
            continue
        cleaned_words.add(word)
    return cleaned_words

def get_words_from_events(event_list):
    unique_words = collect_unique_words_from_events(event_list)
    cleaned_unique_words = clean_unique_words(unique_words)
    return cleaned_unique_words

def save_unique_words_to_json(unique_words, output_file):
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(list(unique_words), f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    event_list_raw = load_events_from_data_dir()
    print(f"{len(event_list_raw)} events loaded")
    
    event_list_cleaned = clean_event_list(event_list_raw)
    unique_words = get_words_from_events(event_list_cleaned)
    print(f"{len(unique_words)} unique words found")

    output_file = os.path.abspath(os.path.join(OUTPUT_PATH, "unique_words.json"))
    save_unique_words_to_json(unique_words, output_file)
    print(f"Unique words saved to {output_file}")