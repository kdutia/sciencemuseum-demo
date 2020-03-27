import requests
import config
import json
url, headers = config.url, config.headers

slug = "search/documents"
params = {
    "random": 10
}

response = requests.get(url+slug, headers=headers, params=params)

print(response.json())