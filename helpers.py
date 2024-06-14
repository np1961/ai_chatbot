import numpy as np
import pandas as pd
from warnings import filterwarnings
from openai import OpenAI
filterwarnings(action='ignore')

from config import constants



class StatusAssigner:
    @classmethod
    def bad_status(cls, message):
        return {
            'status': 'error',
            'error_reason': message
    }
    @classmethod
    def good_status(cls):
        return {
            'status': 'accepted',
    }

class LLM(StatusAssigner):
    def __init__(self):
        self.client=OpenAI(api_key=constants.API_KEY)



    def send_to_model(self,prompt,  file_contents, question, model="gpt-3.5-turbo"):
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": file_contents},
                {"role": "user", "content": question}
            ]
        )

        return completion.choices[0].message.content.splitlines()








class EMBEDDING(StatusAssigner):
    def __init__(self, file_path ='/home/np_1961/chatbot_bank/config/BankFAQs.csv'):

        self.file_path=file_path
        self.client = OpenAI(api_key=constants.API_KEY)
        self.embedding_model = "text-embedding-3-small"
        self.df = None

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _get_embedding(self, text, model):
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding

    def search_reviews(self, input_question, n=3):
       embedding = self._get_embedding(input_question, model='text-embedding-3-small')
       self.df['similarities'] = self.df['embedding'].apply(lambda x: self._cosine_similarity(x, embedding))

       res = self.good_status()
       res['data'] = self.df.sort_values('similarities', ascending=False).head(n)
       return res

    def load_embeddings(self, amount=10):

        self.df = pd.read_csv(self.file_path).head(amount)
        self.df["embedding"] = self.df['Question'].apply(lambda x: self._get_embedding(x, model=self.embedding_model))
        self.df.to_csv('embeddings.csv')
        res = self.good_status()
        return res

    def ask_question(self, input_question):
        if self.df is None:
            return self.bad_status(message= "first load the embeddings then call this request")

        res = self.good_status()
        res['answer'] = self.search_reviews(input_question)['Answer'].iloc[0]
        return res





class RAG:

    def __init__(self):
        self.embedding=EMBEDDING()
        self.llm = LLM()





