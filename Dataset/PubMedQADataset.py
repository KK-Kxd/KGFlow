import json
import os
class PubMedQADataset:

    def __init__(self, file_path=None):
        """
        初始化数据集。

        参数:
            file_path (str): PubMedQA JSON 文件的路径。
        """
        if file_path is None:
            # Get the directory of this file and construct path to dataset
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "dataset", "pubmedqa.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件未找到: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        self.ids = sorted([
            pid for pid, data in self.dataset.items() 
            if data.get('final_decision') in ['yes', 'no', 'maybe']
        ])

    def __len__(self):

        return len(self.ids)

    def __getitem__(self, index):

        sample_id = self.ids[index]
        data = self.dataset[sample_id]
        

        question = data.get('QUESTION', '').strip()

        contexts_text = " ".join(data.get('CONTEXTS', [])).strip()
        

        options_text = "A. Yes\nB. No\nC. Maybe"
        
        # 3. 组合成最终的 text

        text = f"{contexts_text} {question}\n\nOptions:\n{options_text}"

        final_decision = data['final_decision']
        if final_decision == 'yes':
            answer_char = 'A'
            answer_index = 0
            answer_content = 'Yes'
        elif final_decision == 'no':
            answer_char = 'B'
            answer_index = 1
            answer_content = 'No'
        elif final_decision == 'maybe':
            answer_char = 'C'
            answer_index = 2
            answer_content = 'Maybe'
        else:
           pass
            
        return {
            "id": sample_id,
            "text": text,
            "answer": answer_char,
            "answer_index": answer_index,
            "answer_content": answer_content
        }    

