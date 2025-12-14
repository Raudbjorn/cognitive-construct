# Cross-LLM

Multi-provider LLM routing wrapper.

## Usage

```bash
python cross_llm.py call "your prompt" --provider openai
python cross_llm.py call "your prompt" --provider anthropic --model claude-3-haiku-20240307
python cross_llm.py call "your prompt" --tag coding
python cross_llm.py providers
```

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `DEEPSEEK_API_KEY`: DeepSeek API key
- `GEMINI_API_KEY`: Google Gemini API key
- `XAI_API_KEY`: xAI (Grok) API key
- `MISTRAL_API_KEY`: Mistral API key

## Tag-Based Routing

The `--tag` option selects the best provider for different task types:

| Tag | Provider Priority |
|-----|-------------------|
| coding | deepseek → anthropic → openai |
| reasoning | deepseek → anthropic → openai |
| creative | openai → anthropic → gemini |
| business | openai → anthropic → mistral |
| general | openai → anthropic → deepseek |
| math | deepseek → openai → anthropic |

## Dependencies

```bash
pip install httpx
```
