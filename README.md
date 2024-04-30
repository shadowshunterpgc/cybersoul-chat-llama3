# Cria chat com Python + LLama-3 + HuggingFace

## Descrição
A ideia deste código foi criar um chat utilizando Python, HuggingFace e o LLama-3 da Meta e expor o diálogo através de uma API

## Bibliotecas utilizadas
* langchain
* uvicorn
* fastapi
* pydantic

## Código

#### chat.py
```  
  from langchain.chains import LLMChain
  from langchain.prompts import PromptTemplate
  from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
  from fastapi import FastAPI
  from dialogo import Dialogo

  import os
  import uvicorn

  os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

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
  ```

#### dialogo.py
```
from pydantic import BaseModel
from typing import Optional


class Dialogo(BaseModel):
    request: str
    response: Optional[str] = None
```

## Acessando API
![img.png](img.png)