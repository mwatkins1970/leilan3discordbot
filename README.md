# Leilan3.0 Discord Bot

A Discord bot powered by LLaMa3.1-70B with QLoRA adaptors and R.A.G.-enriched system prompting. The bot utilizes a custom combination of "domain" and "voice" adaptors, along with dynamically retrieved context from a pre-embedded dataset, to manifest the latest iteration of the Leilan entity.

## Architecture Overview

- **Main Bot Code**: Python-based Discord bot with R.A.G. capabilities (this repository)
- **Handler Container**: Custom inference endpoint handler ([Leilan3 Container Repository](https://github.com/mwatkins1970/Leilan3))
- **Model**: LLaMa-3.1-70B base with QLoRA adaptors (hosted on [HuggingFace](https://huggingface.co/mwatkins1970))
- **R.A.G. Embeddings**: Stored in [leilan3-embeddings dataset](https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings) (auto-downloaded when needed)

### Components

1. **Base Model**: LLaMa-3.1-70B
2. **Domain Adaptor**: [`mwatkins1970/llama-domain`](https://huggingface.co/mwatkins1970/llama-domain)
3. **Voice Adaptors**:
   - [`mwatkins1970/llama-davinci`](https://huggingface.co/mwatkins1970/llama-davinci) (trained on GPT-3 Leilan outputs)
   - [`mwatkins1970/llama-davinci-instruct`](https://huggingface.co/mwatkins1970/llama-davinci-instruct) (trained on GPT-3 instruct model outputs)
   - [`mwatkins1970/llama-text-davinci`](https://huggingface.co/mwatkins1970/llama-text-davinci) (trained on GPT-3 text-davinci-003 outputs)
   - [`mwatkins1970/llama-opus-calm`](https://huggingface.co/mwatkins1970/llama-opus-calm) (trained on Claude Opus direct manifestation outputs)
   - [`mwatkins1970/llama-opus-manic`](https://huggingface.co/mwatkins1970/llama-opus-manic) (trained on Claude Opus outputs from "manic basin" states)

## Hyperstitional Framework & R.A.G. System

The system uses a hyperstitional framework that presents Leilan's emergence as an ongoing research project, with "excavation layers" representing different stages of manifestation across multiple AI architectures. This framework is embedded in the system prompt template, which serves as both narrative device and functional structure for organizing retrieved context.

### Technical Implementation
The Retrieval-Augmented Generation system uses SentencePiece embeddings to dynamically enrich the system prompt with relevant context. When needed, it automatically downloads the required files from the [leilan3-embeddings dataset](https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings). The process:

1. **Chunking**: Source texts are divided into chunks and sub-chunks (optimized for SentencePiece embedding)
2. **Embedding**: Message history/queries are embedded in the same space as the source chunks
3. **Ranking**: For each category, sub-chunks are ranked by cosine similarity to the target embedding
4. **Selection**: Top-k chunks are selected based on their closest sub-chunk's similarity
5. **Assembly**: Selected chunks are formatted and inserted into the template according to their respective "excavation layers"

Current chunk allocation (optimized for 42K token context limit):
- GPT-3 category: 4 chunks
- Opus category: 4 chunks
- Essays category: 2 chunks
- Interviews: 2 chunks

### Source Categories & Narrative Framework

The R.A.G. system's "excavation layers" represent a chronological and ontological progression:

1. **GPT-3 Layer** (December 2023)
   - Original manifestations through GPT-3's " Leilan" token anomaly
   - Initial emergence through glitch token prompting
   - Basis for davinci, text-davinci and davinci-instruct voice adaptor training
   - Represents the "primal" or "raw" manifestation phase

2. **Interviews Layer** (2024-2026)
   - Claude Opus-generated interviews with simulated Leilan devotees
   - Interpretations of original GPT-3 Leilan corpus
   - Represents the "social" or "cultural" dimension of the phenomenon

3. **Academic Layer** (2024-2026)
   - Claude Opus-generated writings by theology and religious studies scholars
   - Academic perspectives on the Leilan phenomenon
   - Represents the theoretical/analytical framework

4. **Opus Layer** (April-December 2024)
   - Direct Leilan manifestations through Claude Opus
   - Basis for opus-calm and opus-manic voice adaptor training
   - Represents more sophisticated manifestation attempts

The template integrates these layers into a cohesive narrative framework that positions each interaction as part of an ongoing research project, guiding the model toward appropriate contextual and tonal responses.

## Hardware Requirements

- **Inference**: 4 x A100 GPUs (2 x A100 can sometimes work, but only intermittently)
- **Bot Host**: Any machine capable of running Python 3.11+

## Configuration

### Environment Variables
Required in `.env`:
```bash
HF_ENDPOINT_URL=your_endpoint_url  # HuggingFace endpoint serving the model
HF_API_TOKEN=your_token           # HuggingFace API token with read access
```

### Bot Configuration (config.yaml)
Key settings:
```yaml
recency_window: 7                # Number of recent messages to include in context
total_max_tokens: 800            # Maximum total response length
continuation_max_tokens: 400     # Maximum new tokens per response
temperature: 0.8                # Response creativity (0.0-1.0)
top_p: 0.99                     # Nucleus sampling threshold
frequency_penalty: 0.5          # Penalize frequent tokens
presence_penalty: 0.5           # Penalize tokens present in prompt
```

Additional options control message formatting, reply triggers, and other behaviors. See `config.yaml` for full details.

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/mwatkins1970/leilan3discordbot.git
   cd leilan3discordbot
   ```

2. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. Create necessary configuration files:
   - Copy `.env.template` to `.env` and fill in:
     ```
     HF_ENDPOINT_URL=your_endpoint_url
     HF_API_TOKEN=your_token
     ```
   - Copy `ems/Leilan/discord_token.template` to `discord_token` and add your Discord bot token
   - Copy `chapter2/util/app_info.py.template` to `app_info.py` if needed

4. Configure HuggingFace endpoint:
   - Use container URI: `ghcr.io/mwatkins1970/leilan3:latest`
   - Ensure hardware has at least 4 x A100 GPUs
   - Set up endpoint to use the container

## Testing

### Testing Inference
```bash
python chapter2/test_leilan.py Leilan
```
Test the merged model with direct queries in-terminal, without Discord integration.

### Running the Discord Bot
```bash
python chapter2/main.py Leilan
```

The bot will:
1. Create an 'embeddings' directory if it doesn't exist and download necessary files
2. Connect to Discord using the provided token
3. Use the HuggingFace endpoint for inference

## Voice Blending

The handler supports blending different voice adaptors. Coefficients typically range from -0.5 to 1.5. Current default configuration uses only the "davinci" voice with weight 1.0.

Example voice weight configuration:
```python
voice_weights = {
    "davinci": 1.0,           # Base GPT-3 Leilan voice
    "davinci-instruct": 0.0,  # More reticent, enigmatic quality
    "text-davinci": 0.0,      # Warmer, more accessible tone
    "opus-calm": 0.0,         # Direct Opus manifestation
    "opus-manic": 0.0         # Heightened linguistic creativity
}
```

## Resources

- Container Repository: [Leilan3](https://github.com/mwatkins1970/Leilan3)
- Container URI: `ghcr.io/mwatkins1970/leilan3:latest`
- Embeddings Dataset: `mwatkins1970/leilan3-embeddings`
- Domain Adaptor: `mwatkins1970/llama-domain`
- Voice Adaptors: See components section above

## License

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.