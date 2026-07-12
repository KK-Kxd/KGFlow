import json
import os
class QADataset:
    def __init__(self, data, dir=None):
        if dir is None:
            # Get the directory of this file and construct path to dataset
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dir = os.path.join(current_dir, "dataset")
        self.data = data.lower().split("_")[0]
        benchmark = json.load(open(os.path.join(dir, "benchmark.json")))
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __process_data__(self, key):
        data = self.dataset[self.index[key]]
        question = data["question"]
        choices = [v for k, v in data["options"].items()]

        options = [" A: ", " B: ", " C: ", " D: "]

        text = question + "\n"
        for j in range(len(choices)):
            text += "{} {}\n".format(options[j], choices[j])

        answer = data["answer"].strip()
        label_index = ord(answer) - ord('A')
        answer_content = choices[label_index]

        return {"text": text, "answer": answer, "answer_index": label_index, "answer_content": answer_content}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")