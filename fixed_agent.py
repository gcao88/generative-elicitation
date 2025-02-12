import textwrap

from base_active_learning_agent import BaseActiveLearningAgent
from utils import query_api, query_api_any_message
import json
import random

QUESTION_TYPES = ["yn", "text"]
IMPLEMENTATION = "Python regex"  #["Python regex", "system"]
with open("fixed_questions.json", "r", encoding="utf-8") as file:
    question_type_to_questions = json.load(file)
yes_no_types = ['Yes/No', 'Slider', 'text']

class FixedAgent(BaseActiveLearningAgent):
    def __init__(self, target_specification_file, engine, openai_cache_file=None, question_type=None, **kwargs):
        super().__init__(target_specification_file, engine, openai_cache_file, **kwargs)
        question_type_order = random.sample(list(question_type_to_questions.keys()), len(question_type_to_questions))
        self.question_list = []
        for question_type in question_type_order:
            questions = question_type_to_questions[question_type]
            for question in questions:
                if question_type == 'yes_no':
                    self.question_list.append({'question': question, 'type': random.choice(yes_no_types)})
                elif question_type == 'comparison':
                    self.question_list.append({'question': question, 'type': "Options"})
                else:
                    self.question_list.append({'question': question, 'type': "text"})

        self.cur_question = 0

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

    def generate_active_query(self):
        '''Generates a question for the oracle.'''
        question = self.question_list[self.cur_question]
        self.cur_question += 1
        return question

    def generate_oracle_response(self, question):
        '''Generates an oracle response for the question'''
        answer = self.query_oracle_api(question, self.question_type)
        self.interaction_history.append((question, answer))
        return answer

    def query_type(self):
        return f"question_{self.question_type}"
