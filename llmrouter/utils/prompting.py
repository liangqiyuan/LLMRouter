"""
Prompt formatting utilities for LLMRouter scripts
"""

def format_mc_prompt(question, choices):
    """Format prompt for multiple choice tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    formatted_choices = ""
    options = ["A", "B", "C", "D"]

    for i, choice in enumerate(choices):
        formatted_choices += f"{options[i]}. {choice}\n"

    system_prompt = "Answer the following multiple-choice question by selecting the correct option (A, B, C, or D). You MUST put your final answer letter in a parenthesis."
    user_query = f"""## Question:
{question}

## Options:
{formatted_choices}"""

    return {"system": system_prompt, "user": user_query}

def format_gsm8k_prompt(query):
    """Format prompt for GSM8K math tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    system_prompt = "Answer the following math question step by step."
    user_query = f"Question: {query}"
    return {"system": system_prompt, "user": user_query}

def format_math_prompt(query):
    """Format prompt for MATH tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    system_prompt = "Answer the following math question. Make sure to put the answer (and only answer) inside \\boxed{}."
    user_query = f"Question: {query}"
    return {"system": system_prompt, "user": user_query}

def format_commonsense_qa_prompt(query, choices):
    """Format prompt for commonsense QA tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    label = choices["label"]
    text = choices["text"]
    choice_text = ""
    for i, j in zip(label, text):
        choice_text += "\n" + "(" + i + ")" + " " + j

    system_prompt = "Answer the following multiple-choice question by selecting the correct option. You MUST put your final answer letter in a parenthesis."
    user_query = f"Question: {query}\n{choice_text}"
    return {"system": system_prompt, "user": user_query}

def format_mbpp_prompt(text, tests):
    """Format prompt for MBPP code generation tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    tests_str = "\n".join(tests)
    system_prompt = "You are an expert Python programmer. Implement the function with no irrelevant words or comments. Put your code in this format: [BEGIN] <Your Code> [Done]"
    user_query = f"Task: {text}\n\nYour code should pass these tests:\n\n{tests_str}"
    return {"system": system_prompt, "user": user_query}

def format_humaneval_prompt(prompt):
    """Format prompt for HumanEval code generation tasks

    Returns:
        dict: {"system": system_prompt, "user": user_query}
    """
    system_prompt = "You are an expert Python programmer. Implement the function body only. Do not repeat the function signature or docstring. Put the code in this format: [BEGIN] <Your Code - function body only> [Done]"
    user_query = f"Complete the following function:\n\n{prompt}"
    return {"system": system_prompt, "user": user_query}

def generate_task_query(task_name, sample_data):
    """Generate query prompt based on task name and sample_data.

    Returns:
        dict: {"system": system_prompt, "user": user_query}
              For simple tasks without special formatting, system will be None.
    """
    if task_name in ["natural_qa", "trivia_qa"]:
        # No special system prompt for these tasks
        return {"system": None, "user": sample_data['query']}
    elif task_name in ["mmlu"]:
        return format_mc_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "gpqa":
        return format_mc_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "mbpp":
        return format_mbpp_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "human_eval":
        return format_humaneval_prompt(sample_data['query'])
    elif task_name == "gsm8k":
        return format_gsm8k_prompt(sample_data['query'])
    elif task_name == "commonsense_qa":
        return format_commonsense_qa_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "math":
        return format_math_prompt(sample_data['query'])
    elif task_name == "openbook_qa":
        return format_commonsense_qa_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "arc_challenge":
        return format_commonsense_qa_prompt(sample_data['query'], sample_data['choices'])
    else:
        raise ValueError(f"Unknown task name: {task_name}")








