import ast
import json
from llama_cpp import Llama
from openai import OpenAI


class LLMInfer:

    def __init__(self):
        self.model = None
        self.n_gpu_layers = 20
        self.n_batch = 4

    def load_model(self):
        self.model = Llama(model_path="./model/mistral-7b-fed-auto-unsloth.Q8_0_original.gguf",
                           n_gpu_layers=self.n_gpu_layers,  # Uncomment to use GPU acceleration
                           # seed=1337, # Uncomment to set a specific seed
                           # n_ctx=n_batch, # Uncomment to increase the context window
                           )

    def infer(self, user_prompt):
        system_prompt = "### Instruction: Your task is to provide a json given a instruction by the user for " \
                        "federated learning task.Output should be in json format. Keys for json are algo - " \
                        "Classification/Regression, minibatch - size of minibatch (16), epoch- number of epochs(5), " \
                        "lr - learning rate(0.0001), scheduler - full/random/round_robin/latency_proportional(full), " \
                        "clientFraction- fraction of clients involved in federated learning (1),comRounds- number of " \
                        "communication rounds in federated learning(10), optimizer-pytorch optimizer(Adam), " \
                        "loss(CrossEntropyLoss), compress- No/quantize(No), dtype-img(img),dataset-dataset used for " \
                        "training(MNIST). Default values for each key is given inside the bracket.Seperated by / are " \
                        "possible values for the relavent key. Your task is to create a json with above keys and " \
                        "extract possible values from given human prompt as values for the Json.Respond only the " \
                        "json.  ### Input: {inp} ### Response: "

        instruction = system_prompt.format(inp=user_prompt)

        output = self.model(
            instruction,  # Prompt
            max_tokens=250,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["### Input"],  # Stop generating just before the model would generate a new question
            echo=True  # Echo the prompt back in the output
        )  # Generate a completion, can also call create_completion

        result = output['choices'][0]['text'].split("### Response:")[-1]
        print('result ' + str(result))
        result = result.replace("\'", "\"")
        llm_json = json.loads(result)

        return llm_json

    def infer_model(self, config):
        client = OpenAI()
        config = ast.literal_eval(config)

        system_prompt = """You are an AI that strictly conforms to responses in Python. Your responses consist of valid 
        python syntax, with no other comments, explanations, reasoning, or dialogue not consisting of valid Python. 
        Your task is to output pytorch Model code according to the given user prompt Only include pytorch imports, 
        init function and the forward pass function. Do not include code for initializing the model """

        prompt_template = "create a CNN architecture for the following task. Task is a {algo} task with {label_no} " \
                          "labels. image size is {data_shape} and have {data_no} data points altogether. Considering above " \
                          "information create a neural network architecture which could achieve good accuracy "

        user_prompt = prompt_template.format(algo=str(config['algo']), label_no=str(config['no_of_labels']),
                                             data_shape=str(config['data_shape']), data_no=str(config['no_of_data']))

        print(user_prompt)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": str(system_prompt)},
                {"role": "user",
                 "content": user_prompt},
            ]
        )

        model = response.choices[0].message.content
        return model

