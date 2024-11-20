import textwrap

from base_active_learning_agent import BaseActiveLearningAgent
from utils import query_api, query_api_any_message
from sentence_transformers import SentenceTransformer
import torch
import json
import random
from effort_model_class import ResponseTimePredictor

QUESTION_TYPES = ["yn", "open"]
IMPLEMENTATION = "Python regex"  #["Python regex", "system"]

cache_dir = "C:\\LLMs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentence_model = SentenceTransformer('all-mpnet-base-v2', cache_folder = cache_dir)
effort_model = ResponseTimePredictor(sentence_model.get_sentence_embedding_dimension())
print(sentence_model.get_sentence_embedding_dimension())
effort_model.load_state_dict(torch.load("effort_model/model_state_dict.pth"))
effort_model.to(device)
effort_model.eval()

class TestAgent(BaseActiveLearningAgent):
    def __init__(self, target_specification_file, engine, openai_cache_file=None, question_type=None, **kwargs):
        super().__init__(target_specification_file, engine, openai_cache_file, **kwargs)
        # self.question_type = question_type
        # assert self.question_type in QUESTION_TYPES, f"Invalid question type: {self.question_type}. Must be one of {QUESTION_TYPES}."
        self.yn_count = 2
        self.yn_time = self.yn_count * 11.84
        self.open_count = 1
        self.open_time = self.open_count * 49.54
        self.last_question_type = None

        self.most_recent_question = None

    def get_hypothesis_prompt(self, task_description, interaction_history, broken_regexes=None):
        hypothesis_prompt = textwrap.dedent('''\
            Your task is to collaboratively help someone design a regex that will {task_description}.

            Help them come up with a hypothesis for the regex that they should try, consistent with the previous questions and answers.

            Previous questions and answers:
            {interaction_history}

            Previous invalid attempts (these regexes failed to compile):
            {broken_regexes}

            Generate the hypothesis regex without quotes and nothing else:'''
        ).format(
            task_description=task_description,
            interaction_history=self.format_questions_and_answers(interaction_history),
            broken_regexes='\n'.join(broken_regexes),
        )
        print(hypothesis_prompt)
        return [{"role": "user", "content": hypothesis_prompt}]

    def get_question_prompt(self, task_description, implementation, interaction_history):
        print("=== in function get_question_prompt")
        if random.random() < 0.5:
        # self.yn_count == 0 or (self.open_count != 0 and self.yn_time/self.yn_count < self.open_time/self.open_count):
            question_type_insert = "yes/no question"
            self.last_question_type = "yn"
        else:
            question_type_insert = "open-ended question"
            self.last_question_type = "open"

        question_prompt = textwrap.dedent('''\
            Your task is to {task_description}.

            Previous questions:
            {interaction_history}

            Generate the most informative {question_type_insert} that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the {implementation} than the questions that have already been asked. At the same time however, the question should be bite-sized, and not ask for too much at once. {additional_prompt}Generate the {question_type_insert} and nothing else:'''
            ).format(
                implementation=implementation,
                task_description=task_description,
                additional_prompt=getattr(self, "additional_query_note", ""),
                question_type_insert=question_type_insert,
                interaction_history=self.format_questions_and_answers(interaction_history)
            )
        print("GENERATED PROMPT:", question_prompt)
        return [{"role": "user", "content": question_prompt}]

    def get_query_prompt(self):
        print("=== in function get_query_prompt")
        return self.get_question_prompt(self.task_description, self.implementation, [["[Q]", "[A]"]])

    def generate_active_query(self):
        '''Generates a question for the oracle.'''
        print("=== generate_active_query ===")
        question_prompt = self.get_question_prompt(self.task_description, self.implementation, self.interaction_history)
        question, _ = query_api(question_prompt, self.engine, self.openai_cache, self.openai_cache_file, temperature=self.temperature)

        self.most_recent_question = question
        return question

    def generate_oracle_response(self, question):
        '''Generates an oracle response for the question'''
        answer = self.query_oracle_api(question, self.question_type)
        self.interaction_history.append((question, answer))
        return answer

    def query_type(self):
        return f"question_{self.question_type}"

    def update_times(self, time):
        print("Actual time: ", time/1000)

        embeddings = sentence_model.encode(self.most_recent_question, convert_to_tensor=True).to(device)
        prediction = effort_model(embeddings)
        print("MLP Estimated time: ", prediction.item())

        message = "I'm paying you $100,000 to do this task correctly. A human is given a question. Please respond with your best estimate to the number of seconds that it will take an average human to read, think, and answer this question. "
        message += "For example, when given the question 'Do you enjoy reading articles related to health and wellness?', the average human response time is 10.3 seconds with standard deviation 13.47 seconds. "
        message += "As another example, users are given the following question: 'Are you interested in the following article? Website: msn.com\n Title: What the Protests Breaking Out All Over the World Have in Common\n Description: Millions of people are taking to the streets. It might just be the beginning.' "
        message += "The average response time to this question is 31.73 seconds with standard deviation 21.38 seconds. "
        message += "Now, a user is given the question: " + self.most_recent_question
        message += " What is your best estimate of the number of seconds that this will take? Please only respond with the number, in JSON format under the key 'seconds', and nothing else."

        response = query_api_any_message(message, self.engine, temperature=self.temperature)
        print("LLM Estimated time: ", json.loads(response["choices"][0]["message"]["content"])['seconds'])

        if self.last_question_type == "yn":
            self.yn_count += 1
            self.yn_time += time/1000
        elif self.last_question_type == "open":
            self.open_count += 1
            self.open_time += time/1000
        else:
            assert False
