# Solar Power Plant Acoustic Monitoring — Real-Time Prototype

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Select **Live microphone**, allow browser microphone access, and press Start. The dashboard refreshes approximately once per second and compares the current one-second audio window with the immediately preceding one-second window. You can alternatively upload a WAV file; the current summary compares the final two seconds of the file.

## Optional LLM
Set `OPENAI_API_KEY` before starting Streamlit. The LLM is called only when a feature change crosses the configured trigger threshold and calls are throttled to the configured minimum interval. This is to control latency/cost. If the threshold is not crossed, no LLM call is made.

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
streamlit run app.py
```

## Deployment notes
- Live mic uses WebRTC. Localhost usually works directly; hosted deployment needs HTTPS and working WebRTC connectivity (STUN/TURN may be needed behind restrictive networks).
- For WAV upload, the dashboard advances through the file as a whole but this basic version summarizes the final two seconds; it does not simulate playback from the beginning.
- LLM outputs are not calibrated probabilities. A threshold is used as a low-cost trigger for when to ask the LLM, not as the final fault diagnosis.
- This prototype is not a validated fault detector; audio changes can be caused by operating load, noise, or mic position.
