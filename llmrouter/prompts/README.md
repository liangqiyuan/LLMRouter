# Prompt Templates Directory

This directory contains all prompt templates stored as YAML files. Each template is stored in its own YAML file and can be loaded using the `load_prompt_template()` function.

## Directory Structure

```
llmrouter/prompts/
├── __init__.py                    # Loader utility functions
├── task_mc.yaml                   # Multiple choice task system prompt
├── task_gsm8k.yaml                # GSM8K math task system prompt
├── task_math.yaml                 # MATH task system prompt
├── task_mbpp.yaml                 # MBPP code generation system prompt
├── task_humaneval.yaml            # HumanEval code completion system prompt
├── router_qwen.yaml               # Router_R1 prompt for Qwen models
├── router_llama.yaml              # Router_R1 prompt for LLaMA models
├── agent_prompt.yaml              # Agent prompt for multi-agent reasoning
├── agent_decomp_cot.yaml         # Chain-of-thought aggregation prompt
├── agent_decomp.yaml             # Simple decomposition prompt
├── agent_decomp_route.yaml       # Decomposition + routing template
├── data_conversion.yaml          # Data format conversion prompt
└── README.md                     # This file
```

## Usage

```python
from llmrouter.prompts import load_prompt_template

# Load a template
template = load_prompt_template("task_mc")
prompt = template.format(question="What is 2+2?")
```

## Template Files

### Task Prompts
- **task_mc.yaml**: System prompt for multiple choice questions (used by mmlu, gpqa, commonsense_qa, etc.)
- **task_gsm8k.yaml**: System prompt for GSM8K math word problems
- **task_math.yaml**: System prompt for MATH dataset problems
- **task_mbpp.yaml**: System prompt for MBPP code generation
- **task_humaneval.yaml**: System prompt for HumanEval code completion

### Router Prompts
- **router_qwen.yaml**: Router_R1 prompt template for Qwen model family
- **router_llama.yaml**: Router_R1 prompt template for LLaMA model family

### Agent Prompts
- **agent_prompt.yaml**: Instructions for specialized assistant models in multi-agent reasoning
- **agent_decomp_cot.yaml**: Chain-of-thought prompt for aggregating decomposed responses
- **agent_decomp.yaml**: Simple query decomposition prompt
- **agent_decomp_route.yaml**: Template for decomposition + routing (filled at runtime)

### Data Prompts
- **data_conversion.yaml**: Prompt template for data format conversion

## YAML Format

Each YAML file follows this structure:

```yaml
template: |
  Your prompt template string here.
  Can span multiple lines.
  Use {placeholder} for formatting.
```

## Loading Templates

All templates are loaded through the `load_prompt_template()` function in `__init__.py`, which:
1. Reads the YAML file from `llmrouter/prompts/`
2. Extracts the `template` key
3. Returns the template string

This ensures all prompt strings are centralized and easily editable without modifying Python code.
