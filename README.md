OpenAI API compatible TTS model hosted on localhost:5000

### Features 
- Nice GUI
- Custom Voices / Voice Cloning (from only seconds of audio)
- Microphone recording
- Lazy loading / Auto Unload (5m)

![webui](images/webui.png)

### How To Run
- clone the repo
- run `python server.py`
- go to `http://localhost:5000`

### Adding voices
- place your voice in `voices/` folder
- rerun server.py and refresh the page
- select your voice in the dropdown menu

### Using in Open WebUI
- `Admin Panel > Settings > Audio`
- `TTS engine: OpenAI`
- `API Base URL: http://localhost:5000/v1`
- `API Key: unused (0)`
- `TTS Voice: elise`
- `TTS Model: unused (chatterbox)`
- Save