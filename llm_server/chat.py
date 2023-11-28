import requests
import json
import argparse

def send_request(data):
    """ Send a request to the API and print the response.

    Args:
        data (dict): The data to be sent to the API. See main.py for the expected keys.
    """
    url = "http://localhost:8080/process"
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    full_response_text = response.json().get('text', '')
    reqtime = response.json().get('request_time', 0)
    print(f"Response:\n{full_response_text}")
    print(f"Request time: {reqtime}s")

def main():
    parser = argparse.ArgumentParser(description="Send a prompt to the LLM API.")
    parser.add_argument('-p', '--prompt', help='The prompt to be sent to the API')
    parser.add_argument('-m', '--max_new_tokens', help='The max_new_tokens to be sent to the API')
    parser.add_argument('-t', '--temperature', help='Temperature for generation')
    args = parser.parse_args()
    data = {}
    if args.prompt:
        data['prompt'] = args.prompt
    if args.max_new_tokens:
        data['max_new_tokens'] = args.max_new_tokens
    if args.temperature:
        data['temperature'] = args.temperature
    if data:
        send_request(data)

if __name__ == "__main__":
    main()
