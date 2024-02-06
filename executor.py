# Wrap the StableLM LLM from Stability.AI 
# Serve it using Deployment
# one deployment serves one Executor , for multiple Executors combine them into a pipeline


from jina import Executor , requests
from docarray import DocList, BaseDoc

from transformers import pipeline

class Prompt(BaseDoc):
    text : str

class Generation(BaseDoc):
    prompt : str
    text : str 


class StableLM(Executor):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.generator = pipeline(
            'text-generation' , model = 'stabilityai/stablelm-base-alpha-3b'
        )


@requests 
def generate( self , docs: DocList[Prompt] , **kwargs) -> DocList[Generation]:
    generations = DocList[Generation]()
    prompts = docs.text
    result = self.generator(prompts)
    for prompt , output in zi(prompts , result):
        generations.append(Generation(propmt=prompt , text = output ))
    return generations