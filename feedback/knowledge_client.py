from xai_sdk import Client
from xai_sdk.chat import user, system

from pydantic import BaseModel, Field

class PayloadClassified(BaseModel):
    """
    create a model with atributes to organize the "label" and "explanation"

    args:
        BaseModel → define the object that organize and create fields with description

    output:
        None

    time complexity → o(1)
    """
    label: float = Field(description='0.0 (BENIGN) OR 1.0 (MALICIUS)') # payload label (0 OR 1)
    explanation: str = Field(description='Sentence of your reasoning (max length; 43 words)') # the reasoning of the model of his decission

class KnowledgeDistillerLLM:
    def __init__(self, model: str, api_key: str):
        """
        initialize the method with model, system prompts and LLM client

        args:
            model: str → receive the name of an LLM model

        output:
            None

        time complexity → o(1)
        """
        self.model: str = model # model name (id)

        # open the 'payload classification system prompt' in reading mode
        with open(r'feedback/payload.md', 'r', encoding='utf-8') as pcs:
            self.payload_classification_system = pcs.read()

        # initialize the LLM model client whit a timeout of 60 seconds
        self.client = Client(
            api_key=api_key,
            timeout=60
        )

    def inference_query(self, payload: str):
        """
        receive a n-gram, and return a vector with the unicode values

        args:
            payload: str → attack payload and explanation

        output:
            dict[str, float|str] → dictionary with label and LLM explanation

        time complexity → o(n*i)
        """
        chat = self.client.chat.create(self.model)

        chat.append(system(self.payload_classification_system))
        chat.append(user(payload))

        _, payload_classified = chat.parse(PayloadClassified) # change the `PayloadClassified` descriptions whit the LLM responses

        # build a JSON response for the LLM
        response = {
            "label": payload_classified.label,
            "explanation": payload_classified.explanation
        }

        return response

if __name__ == '__main__':
    """
    execute this block only when the script is run directly
    
    command (.../geometric_active_perceptron):
        python -B -m feedback.knowledge_client
    """

    api_key = 'n/A' # example api key (xai api key)

    knowledge_distiller = KnowledgeDistillerLLM("grok-4-fast-reasoning", api_key) # select the model and initialize the object

    # make inference to the LLM model whit a payload
    payload_classfied = knowledge_distiller.inference_query(payload='The actor made 120 request in 1 minute; <payload>Hello, my name is Maria and I like reading books.</payload>')

    print(payload_classfied)
