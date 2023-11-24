from openai import OpenAI
response = OpenAI().chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Translate: 'The lazy fox' into spanish."},
  ]
)
print(response)
