import requests
import config
import sys
import json
url, headers = config.url, config.headers

def get_record_by_id(slug):
    slug = f"{slug}"
    response = requests.get(url+slug, headers=headers)

    return response.json()

if __name__ == "__main__":
    id = sys.argv[1]
    content = get_record_by_id(id)
    
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w') as f:
            json.dump(content, f)
