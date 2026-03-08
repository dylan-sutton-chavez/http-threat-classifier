# Large Language Data Destillation (Black Box Destillation)

_Knowledge distillation is a machine learning technique that aims to transfer the learnings of a large pre-trained model, the “teacher model,” to a smaller “student model.” It’s used in deep learning as a form of model compression and knowledge transfer, particularly for massive deep neural networks._ — (Dave Bergmann, IBM)

## 1. Knowledge Distiller Usage

The object `KnowledgeDistillerLLM` works as a wrapper of the xai sdk, where the model receive two arguments; `model: str`, `api_key: str`.

```python
api_key = 'n/A' # example api key (xai api key)
knowledge_distiller = KnowledgeDistillerLLM("grok-4-fast-reasoning", api_key) # select the model and initialize the object
```

### 1.1 Expected Prompt

This model has a pre-created system prompt, and the model expec an specific format for the user prompt;

```markdown
The user will provide an untrusted, potentially payload within <payload>...<payload> delimiters. And a explanation of the request (method, metrcis,...)
```

Basically you need to explain the data of the request and the payload write whit two delimiters; <payload>...<payload>

### 1.2 Knowledge Distiller Module

The `inference query` receive an argument; payload: str. Where receive the prompt of the user.

```python
# make inference to the LLM model whit a payload
payload_classfied = knowledge_distiller.inference_query(payload='The actor made 120 request in 1 minute; <payload>Hello, my name is Maria and I like reading books.</payload>')
print(payload_classfied)
```

Ouput (dictionary with the label and explanation);

```bash
{'label': 1.0, 'explanation': 'High request rate (120/min) → indicates potential DoS or flood attack vector, despite benign payload content.'}
```
