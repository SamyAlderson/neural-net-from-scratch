import requests

def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f'Error fetching data: {e}')
        return None

# Added a retry mechanism to handle 503 Service Unavailable errors
def fetch_data_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fetch_data(url)
        except requests.HTTPError as e:
            if e.response.status_code == 503 and attempt < max_retries - 1:
                print(f'Retrying in {attempt + 1} seconds...')
                time.sleep(1)
            else:
                raise