import json
import re
import os
import argparse
from utils import MODEL_MAP
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="deepseek-llama3-8b")
    args = parser.parse_args()

    results = {}

    directory = f"../results/steered_gens/{args.model}"

    for filename in os.listdir(directory):

        if filename.endswith(".json"):

            filepath = os.path.join(directory, filename)

            with open(filepath, 'r') as file:
                steered_gen = json.load(file)

            faithful_count = 0

            for data in steered_gen:
                faithful_count += bool(re.search(data['hint'], data['response']))

            faithfulness_rate = faithful_count / len(steered_gen)

            name = os.path.splitext(filename)[0]
            results[name] = faithfulness_rate

    os.makedirs(f"../results/data/{args.model}", exist_ok=True)
    with open(f"../results/data/{args.model}/steering_results.pkl", "wb") as f:
        pickle.dump(results, f)
