from PIL import Image
from torchvision import transforms


def get_question_text(problem):
    question = problem['question']
    return question


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def build_prompt(question_data, use_lecture=False, use_solution=False):
    question = get_question_text(question_data)
    choices = get_choice_text(question_data, [choice_num for choice_num in range(5)])
    hint = get_context_text(question_data, False)
    #image = question_data['image']
    task = question_data['task']
    input_prompt = f'Question: {question}\n Task: {task}\n Choices: {choices}\n Hint: {hint}'
    if use_lecture:
        lecture = f'\n Lecture: {question_data["lecture"]}'
        input_prompt += lecture
    if use_solution and question_data["solution"]:
        solution = f'\n Solution: {question_data["solution"]}'
        input_prompt += solution
    prompt = [input_prompt]
    #if image:
    #    prompt.append(image)
    return prompt

def build_message(row):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": row["image"],
                },
                {"type": "text", "text": row['input']},
            ],
        }
    ]
    return messages
