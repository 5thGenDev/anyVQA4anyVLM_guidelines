import sys
import json

def find_text_in_file(filename, search_text):
    """Find the exact text in the file and return the starting position."""
    try:
        with open(filename, 'r') as file:
            content = file.read()
            start_position = content.find(search_text)
            if start_position == -1:
                print(f"Text '{search_text}' not found in the file.")
                return None
            return start_position
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

def read_json_chunk(filename, start, stop):
    try:
        with open(filename, 'r') as file:
            file.seek(start)
            chunk = file.read(stop - start)
            # Try to load JSON, it might not be a complete structure, so we just print the chunk
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                data = chunk  # If not valid JSON, print as is
            return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

def process_json_data(data):
    if isinstance(data, dict):
        # Pretty print the JSON object if it's a valid dict
        print(json.dumps(data, indent=4))
    else:
        # Print as text if it's not a valid JSON
        print(data)

# python read_text_in_json.py <your json file> <start index> <stop index> =>> return entire .json (try about 20k range)
# python read_text_in_json.py <find> <keyword> =>> return index of what keyword
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 read_json.py <filename> <start or 'find'> <stop or 'text_to_find'>")
        sys.exit(1)

    filename = sys.argv[1]
    start_or_find = sys.argv[2]
    stop_or_text = sys.argv[3]

    if start_or_find.lower() == 'find':
        start_position = find_text_in_file(filename, stop_or_text)
        if start_position is not None:
            print(f"Text found at position: {start_position}")
    else:
        start = int(start_or_find)
        stop = int(stop_or_text)
        json_data = read_json_chunk(filename, start, stop)
        process_json_data(json_data)

if __name__ == "__main__":
    main()

