from openai import OpenAI
import csv
import ast
import time
OPENAI_API_KEY= "YOUR GPT API KEY"
train_path = "train.csv"
augmented_train_path = "augmented_train.csv"
client = OpenAI(api_key=OPENAI_API_KEY)


# Open the source CSV file for reading and the destination CSV file for writing
with open(train_path, mode='r', newline='', encoding='utf-8') as source_file, open(augmented_train_path, mode='w', newline='', encoding='utf-8') as destination_file:
    reader = csv.DictReader(source_file)
    writer = csv.DictWriter(destination_file, fieldnames=reader.fieldnames)
    writer.writeheader()
    start_index = 0
    end_index = 10000
    index = 0
    for row in reader:
        if index < start_index:
            index = index + 1
            continue
        if index > end_index:
            index = index + 1
            break
        index = index + 1

        utterance_data_1 = ast.literal_eval(row['utterance1'])
        utterance_data_2 = ast.literal_eval(row['utterance2'])
        print("Sample", index - 1)
        print("Original Data Text:", utterance_data_1['text'])
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are tasked with generating text that conveys the same meaning and maintains a similar language style to the given sample from the dataset. Please only respond with the generated text."},
                {"role": "user", "content": "Given sample: "+utterance_data_1['text']}
            ]
        )
        utterance_data_1['text'] = completion.choices[0].message.content
        print("Generated Data Text:", utterance_data_1['text'])

        print("Original Data Text:", utterance_data_2['text'])
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are tasked with generating text that conveys the same meaning and maintains a similar language style to the given sample from the dataset. Please only respond with the generated text."},
                {"role": "user", "content": "Given sample: "+utterance_data_2['text']}
            ]
        )
        utterance_data_2['text'] = completion.choices[0].message.content
        print("Generated Data Text:", utterance_data_2['text'])

        row['utterance1'] = str(utterance_data_1)
        row['utterance2'] = str(utterance_data_2)
        writer.writerow(row)
        time.sleep(0.01)

'''
completion = client.chat.completions.create(
    model="gpt-3.5-turbo", # Specify the model you want to use
    messages=[
        {"role": "system", "content": "You are tasked with generating text that conveys the same meaning and maintains a similar language style to the given sample from the dataset. Please only respond with the generated text."},
        {"role": "user", "content": "Given sample: How are you doing today?"}
    ]
)
print(completion.choices[0].message)
print(completion.choices[1].message)
'''
