
entity_extract_prompt = """
Given the following multiple-choice query {query}, extract all relevant UMLS medical entities contained within the question stem.
Identify and extract all medical entities, such as diseases, proteins, genes, drugs, phenotypes, anatomical regions, treatments, or other relevant medical entities.
Ensure that the extracted entities are specific and medically relevant.
If no relevant medical entities are found in the question, please provide some UMLS entities that might be related to answering the question.
Only return the extracted entities in JSON format with the key "medical_terminologies" and the value is a list of extracted entities.

Example 1:
Query: "A previously healthy 37-year-old woman comes to the physician because of a 3-month history of episodes of severe anxiety, shortness of breath, palpitations, and numbness in her hands and feet. Her vital signs are within normal limits. Physical examination shows no abnormalities. Thyroid function studies and an ECG show no abnormalities. Which of the following is the most appropriate pharmacotherapy?"
Output: {{"medical_terminologies": ["anxiety", "shortness of breath", "palpitations", "numbness", "thyroid function", "ECG", "pharmacotherapy"]}}

Example 2:
Query: "The high cou has the power to stay the execution of a pregnant woman according to which section of Criminal Procedure Code?"
Output: {{"medical_terminologies": ["pregnant woman"]}}


Query: {query}
"""



score_relation_prompt = """Please score relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of all relations must equal 1).
Return the result in JSON format as follows:
{{
  "relations": [
    {{"relation": "relation_1", "score": 0.XX}},
    {{"relation": "relation_2", "score": 0.XX}},
  ]
}}

Do not answer with any other information or format.
Q: {}
entities: {}
Relation: {}
"""

score_entity_prompt = """
Please score the entities contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities must equal 1).
if the entity is not relevant to the question, please score it as 0.And if the entity score is 0, please do not include it in the result.

IMPORTANT: You must ONLY score entities from the provided entities list below. Do NOT create new entities or modify entity names. Use the exact entity names as they appear in the list.

Return the result in JSON format as follows:
{{
  "entities": [
    {{"entity": "entity_1", "score": 0.XX}},
    {{"entity": "entity_2", "score": 0.XX}},
  ]
}}


Q: {}
Relation: {}
Entities to score:
"""




prompt_evaluate="""Given a multiple-choice question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Example:
Example:
Q: What is a common medication for Type 2 Diabetes, and what is a primary risk factor for this condition?
Knowledge Triplets: (Type 2 Diabetes, has_treatment, Metformin)
A: {No}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide a common medication for Type 2 Diabetes, which is Metformin. To answer the second part of the question, additional knowledge about the risk factors for the disease is necessary.\\

Q: The drug Atorvastatin, used to treat hypercholesterolemia, belongs to what class of medications?
Knowledge Triplets:(Hypercholesterolemia, has_treatment, Atorvastatin)  
(Atorvastatin, is\_a, Statin)
A: {Yes}. Based on the given knowledge triplets, the drug used for hypercholesterolemia, Atorvastatin, is a Statin. Therefore, the information is sufficient to answer the question.
Question:
"""


answer_without_prompt = """Given a multiple-choice questions, you are asked to answer the question with  your knowledge.
Please provide the answer in the following format: "Answer: A/B/C/D".

Q: {}
"""




