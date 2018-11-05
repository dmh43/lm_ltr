import json
import sys

def main():
  with open('./robust_train_query_results_tokens.json', 'r') as fh:
    train_data = json.load(fh)
    if len(sys.argv) > 1:
      num_train = sys.argv[1]
    else:
      num_train = 11000
  with open(f'./robust_train_query_results_tokens_first_{num_train}.json', 'w+') as fh:
    json.dump([dict(row) for row in train_data[:num_train]], fh)

if __name__ == "__main__": main()
