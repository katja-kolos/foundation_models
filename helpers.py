def build_prompt(question_data, use_lecture=False, use_solution=False):
    question = base_prompt.get_question_text(question_data)
    choices = base_prompt.get_choice_text(question_data, [choice_num for choice_num in range(5)])
    hint = base_prompt.get_context_text(question_data, False)
    image = question_data['image']
    task = question_data['task']
    input_prompt = f'Question: {question}\n Task: {task}\n Choices: {choices}\n Hint: {hint}'
    if use_lecture:
        lecture = f'\n Lecture: {question_data["lecture"]}'
        input_prompt += lecture
    if use_solution and question_data["solution"]:
        solution = f'\n Solution: {question_data["solution"]}'
        input_prompt += solution
    prompt = [input_prompt]
    if image:
        prompt.append(image)
    return prompt
