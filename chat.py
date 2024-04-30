from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from fastapi import FastAPI
from dialogo import Dialogo

import os
import uvicorn

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'aaaaaaaaaaaaaaaaaaaa'

app = FastAPI()


def resposta(question):
    template = """
        Pergunta: {question}
        Resposta:"""

    prompt = PromptTemplate(template=template, input_variables=['question'])
    model = HuggingFaceEndpoint(
        repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
        task='text-generation',
        max_new_tokens=1024,
        top_k=30,
        temperature=0.1,
        repetition_penalty=1.03
    )

    llm_chain = LLMChain(prompt=prompt, llm=model)

    return llm_chain.run(question)


def chat():
    while True:
        question = input('Humano: ')
        responses = resposta(question).split('        ')
        print(f'Maquina: {responses[0].replace('\n', '')}')


@app.get('/chat/')
async def endpoint(dialogo: Dialogo):
    responses = resposta(dialogo.request).split('        ')
    return {
        "request": dialogo.request,
        "response": responses[0].replace('\n', '')
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
