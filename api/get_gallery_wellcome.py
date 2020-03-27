import requests
import json
from config import headers

url = "https://collection.sciencemuseumgroup.org.uk/search/museum/science-museum/gallery/medicine:-the-wellcome-galleries"
i = 1

while True:
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(response.status_code)
        break
    
    content = response.json()

    output_name = f'data/raw/wellcome/wellcome_all{i}.json'
    with open(output_name, 'w') as f:
        json.dump(content, f)

    print(f"{i}/{content['meta']['total_pages']} done")

    i+=1
    url = content['links']['next']
