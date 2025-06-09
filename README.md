# Ollama Assist for Home Assistant

A Home Assistant integration that provides a conversational interface to your smart home using Ollama's local language models. This integration allows you to interact with your Home Assistant instance using natural language, with support for executing various home automation tasks and retrieving information.

## Features

- Local LLM processing using Ollama models
- Configurable system prompts
- Dynamic tool support for home automation tasks
- Model selection from available Ollama models
- Natural language interaction with your smart home

## Prerequisites

1. A running Ollama instance with at least one model installed
2. Home Assistant installation

## Installation

1. Add this repository to HACS:
   ```
   https://github.com/ianisms/ha.ollama.conv.tools
   ```
2. Install the "Ollama Assist" integration through HACS
3. Restart Home Assistant
4. Add the integration through the Home Assistant UI

## Configuration

1. Go to Settings -> Devices & Services
2. Click "Add Integration"
3. Search for "Ollama Assist"
4. Configure the following:
   - Ollama server host and port
   - Select your preferred model
   - Customize the system prompt (optional)

### Available Settings

- **Host**: The hostname or IP address where Ollama is running
- **Port**: The port number Ollama is listening on (default: 11434)
- **Model**: Choose from available models on your Ollama server
- **System Prompt**: Customize how the assistant behaves (optional)

## Usage

Once configured, you can interact with the assistant through:

1. The Home Assistant Conversation panel
2. Voice commands (if voice assistants are configured)
3. The companion app

Example commands:
- "What's the temperature in the living room?"
- "Turn on the kitchen lights"
- "Is the garage door closed?"

## Tools

The integration comes with several built-in tools:
- Weather information
- Device control and status
- General home automation tasks

## Customization

You can customize the assistant's behavior by:

1. Modifying the system prompt
2. Adding custom tools
3. Adjusting the prompt templates in prompts.json

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## Support

For issues and feature requests, please use the GitHub issue tracker.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

Author: Ian N. Bennett (ianisms)
