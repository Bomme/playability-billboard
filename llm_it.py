import csv
import pickle
from datetime import datetime

import pandas as pd
from openai import OpenAI, OpenAIError
from tqdm import tqdm

GPT_MODEL = "gpt-4-turbo-2024-04-09"

client = OpenAI()


def get_chords_strings(df):
    chord_strings = {}

    for i, item in df.iterrows():
        chords_path = item["chord_locations"]
        with open(chords_path, "r") as f:
            lines = [
                a[1].split(",") for a in csv.reader(f, delimiter="\t")
                if a and not a[0].startswith("#")]
            chords = [
                [item for item in line if not len(item) == 1 and ("|" in item or any([c.isupper() for c in item]))] for
                line in lines if len(line) > 1]
            chords = [chord[0] for chord in chords if len(chord) > 0]
            chord_strings[i] = "\n".join(chords)
    return chord_strings


def get_difficulty_prediction(chords_string):
    rubric_prompt = """
|   Criterion    | Very difficult (3 points) | Difficult (2 points) | Easy (1 point) | Very Easy (0 points) |
| :------------: | :----------------------: | :------------------: | :------------: | :------------------: |
| Uncommonness of chord | A lot of uncommon chords  |  Some uncommon chords  | Few uncommon chords | No uncommon chords |
| Chord finger positioning | Very cramped or very wide fingerspread | Uncomfortable or spread out fingers | Slightly uncomfortable or spread out fingers | Comfortable hand and finger position |
| Chord fingering difficulty | Mostly chords that require four fingers or barre chords | Some chords require four fingers to be played or are barre chords (not A or E) | Most chords require three fingers or are A or E barre chords | Most chords can be played with two or three fingers |
| Repetitiveness | No repeated chord progressions | A few repeated chord progressions | Quite a bit of repetition of chord progressions | A lot of repetition of chord progressions |
| Right-hand complexity | For some chords multiple inner strings are not strummed | For some chords one inner string is not strummed | For some of the chords one or more outer strings are not strummed | For the chords all strings are strummed |
| Chord progression time | Very quick chord transitions | Quick chord transitions | Slow chord transitions | Very slow chord transitions |
| Beat difficulty (syncopes/ghostnotes) | A lot of syncopes or ghostnotes | Some syncopes or ghostnotes | A few syncopes or ghostnotes | No syncopes or ghostnotes |
    """
    prompt = "Your task is to rate the difficulty of the following chord progression on a scale from 0 to 3. The difficulty of the chord progression is determined by the following criteria:\n\n"
    prompt += rubric_prompt
    prompt += "\n\n"
    prompt += "The chord progression is:\n\n"
    prompt_end = "\n\nFirst analyze the chord progression and explain your steps in the following. Then, rate each criterion on a scale from 0 to 3 and explain your rating."

    messages = [
        {
            "role": "user",
            "content": prompt + chords_string + prompt_end
        }
    ]

    response = client.chat.completions.create(messages=messages, model=GPT_MODEL)
    summary_json_message = {
        "role": "system",
        "content": """
    Collect the scores in the provided text as JSON and use the following keys:
            cfp: Chord finger positioning,
                    cfd: Chord finger difficulty,
                    uc: Uncommonness of chord,
                    rhc: Right-hand complexity,
                    cpt: Chord progression time,
                    bd: Beat difficulty,
                    r: Repetitiveness,
            The scores should be between 0 and 3.
            """
    }
    response_func = client.chat.completions.create(
        messages=[summary_json_message, response.choices[0].message],
        model=GPT_MODEL,
        response_format={"type": "json_object"}
    )

    return response_func.choices[0].message.content, response.choices[0].message.content


if __name__ == "__main__":
    target_data_path = "data/Annotations.csv"
    df = pd.read_csv(target_data_path)
    chord_strings = get_chords_strings(df)
    results = []
    messages = []
    for i, item in tqdm(df.iterrows(), total=len(df)):
        try:
            result, message = get_difficulty_prediction(chord_strings[i])
            results.append(result)
            messages.append(message)
        except OpenAIError as e:
            print(e)
            results.append(None)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    with open(f"difficulty_predictions_{GPT_MODEL}_{timestamp}.pickle", "wb") as f:
        pickle.dump(results, f)
    with open(f"difficulty_predictions_{GPT_MODEL}_{timestamp}_messages.pickle", "wb") as f:
        pickle.dump(messages, f)
