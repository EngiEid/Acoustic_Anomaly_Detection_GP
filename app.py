import io, os, json, time, threading
from dotenv import load_dotenv

# Load .env for local development only. On Railway, set GEMINI_API_KEY in Variables.
load_dotenv()
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from scipy.io import wavfile
from scipy.signal import spectrogram
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

st.set_page_config(page_title="Solar Acoustic Monitor", page_icon="☀️", layout="wide")
st.title("☀️ Solar Power Plant — Real-Time Acoustic Monitor")
st.caption("Initial baseline • Continuous synchronized visuals • Estimated change onset • Gemini fallback")

class AudioBuffer(AudioProcessorBase):
    def __init__(self):
        self.lock=threading.Lock(); self.chunks=[]; self.sample_rate=48000
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        a=frame.to_ndarray()
        if a.ndim==2: a=a.mean(axis=0) if a.shape[0]<=8 else a.mean(axis=1)
        a=np.asarray(a,dtype=np.float32).reshape(-1); sr=frame.sample_rate or 48000
        with self.lock:
            self.sample_rate=sr; self.chunks.append(a.copy())
            while self.chunks and sum(len(c) for c in self.chunks)>sr*600: self.chunks.pop(0)
        return frame
    def snapshot(self):
        with self.lock:
            return self.sample_rate, np.concatenate(self.chunks).astype(float) if self.chunks else np.array([],float)

def read_wav(data):
    sr,x=wavfile.read(io.BytesIO(data))
    if x.ndim==2: x=x.astype(float).mean(axis=1)
    else: x=x.astype(float)
    if np.issubdtype(x.dtype,np.integer): x=x/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
    elif x.size and np.max(np.abs(x))>1: x=x/np.max(np.abs(x))
    return sr,np.asarray(x,float)

def features(x,sr):
    x=np.asarray(x,float)
    if len(x)<max(16,int(sr*.1)): return None
    x=x-x.mean(); rms=float(np.sqrt(np.mean(x*x))); peak=float(np.max(np.abs(x)))
    spec=np.abs(np.fft.rfft(x*np.hanning(len(x)))); freq=np.fft.rfftfreq(len(x),1/sr)
    if len(spec): spec[0]=0
    total=float(spec.sum()); power=spec**2; ptotal=float(power.sum())+1e-12
    return {"dominant_hz":float(freq[np.argmax(spec)]) if total else 0.,
            "spectral_centroid_hz":float((freq*spec).sum()/total) if total else 0.,
            "rms":rms,"peak":peak,
            "low_0_250Hz":float(power[(freq>=0)&(freq<250)].sum()/ptotal),
            "mid_250_1000Hz":float(power[(freq>=250)&(freq<1000)].sum()/ptotal),
            "high_1000Hz_plus":float(power[(freq>=1000)&(freq<sr/2)].sum()/ptotal)}

def deviations(base,cur):
    out={}
    for k,a in base.items():
        b=cur[k]
        out[k]=abs(b-a)*100 if k in ("low_0_250Hz","mid_250_1000Hz","high_1000Hz_plus") else abs(b-a)/max(abs(a),1e-6)*100
    return out

def get_gemini_api_key():
    """Read the API key from environment variables or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        key = ""
    return key or None

def gemini_assess(key,equipment,baseline,current,changes,onset):
    from google import genai
    client=genai.Client(api_key=key)
    prompt = f"""You are a professional acoustic condition-monitoring analyst specializing in electrical equipment at solar power plants.

Analyze the provided acoustic signal features and assess whether the detected change indicates a potentially abnormal operating condition.

Return exactly one valid JSON object. Do not include Markdown, code fences, or text outside the JSON. All human-readable text must be written in professional technical English.

Use exactly this JSON structure:

{{
  "status": "normal",
  "confidence": "medium",
  "explanation_en": "A concise, professional technical assessment.",
  "evidence": [
    "A specific observation supported by the measured signal features."
  ],
  "possible_non_fault_causes": [
    "A brief, relevant alternative explanation, only when justified by the evidence."
  ]
}}

Assessment guidelines:
1. Base the assessment strictly on the supplied acoustic features, baseline deviations, and estimated change onset.
2. Distinguish measured observations from potential fault interpretations.
3. Use "suspicious" when deviations suggest a potentially abnormal acoustic condition, "normal" when evidence supports expected variation, and "uncertain" when evidence is insufficient.
4. Never claim that a mechanical or electrical fault is confirmed from acoustic features alone.
5. Keep the explanation concise, objective, and technically precise.
6. Include only 1–3 concise evidence items based on actual measurements.
7. Include at most two non-fault causes, only when relevant. Do not list generic causes or repeat caveats.
8. Set confidence according to the strength and consistency of the evidence.
9. Do not treat a detected change point as proof of a fault.
10. All human-readable output must be in English. Return valid JSON with exactly the specified keys.

Equipment: {equipment}

Initial baseline features: {json.dumps(baseline)}

Current signal features: {json.dumps(current)}

Deviation from baseline (%; spectral band values are percentage points): {json.dumps(changes)}

Estimated change onset (seconds): {onset}"""

    errors=[]
    for model in ["gemini-3.7-flash","gemini-3.6-flash","gemini-3.5-flash"]:
        try:
            r=client.models.generate_content(model=model,contents=prompt)
            txt=(r.text or "").strip()
            if txt.startswith("```"):
                txt=txt.split("\n",1)[-1]
                if txt.endswith("```"): txt=txt[:-3]
            result=json.loads(txt); result["_model_used"]=model
            return result,errors
        except Exception as e: errors.append(f"{model}: {str(e)[:220]}")
    return None,errors

with st.sidebar:
    st.header("Monitoring setup")
    equipment=st.selectbox("Equipment",["PV Inverter","Transformer","Cooling Fan","Switchgear / Breaker"])
    source=st.radio("Audio source",["Live microphone","Audio file (real-time playback)"])
    baseline_secs=st.selectbox("Initial baseline duration (seconds)",[1,2],index=1)
    shown=st.slider("Waveform history shown (seconds)",5,30,15)
    threshold=st.slider("Baseline deviation trigger (%)",10,200,40,5)
    st.caption("Threshold is used to estimate a candidate change point; Gemini interprets the candidate.")
sr=48000; audio=np.array([],float)
if source=="Live microphone":
    st.subheader("Live microphone")
    ctx=webrtc_streamer(key="solar-live-v4",mode=WebRtcMode.SENDONLY,audio_processor_factory=AudioBuffer,
      media_stream_constraints={"video":False,"audio":True},async_processing=True)
    if ctx.audio_processor: sr,audio=ctx.audio_processor.snapshot()
    if ctx.state.playing:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=1000,key="live-refresh-v4")
        except Exception: st.warning("Install streamlit-autorefresh for automatic refresh.")
else:
    file=st.file_uploader("Upload WAV file",type=["wav"])
    if file:
        try:
            sr,full=read_wav(file.getvalue()); st.audio(file.getvalue())
            # Keep the completed audio in session state so the charts/results remain visible.
            a,b=st.columns(2)
            if a.button("▶ Start / restart real-time playback",type="primary"):
                st.session_state.file_started=time.time()
                st.session_state.file_playing=True
                st.session_state.file_finished=False
                st.session_state.file_last_audio=np.array([],float)
            if b.button("⏹ Stop playback"):
                st.session_state.file_playing=False
                st.session_state.file_finished=False
            if st.session_state.get("file_playing",False):
                elapsed=min(time.time()-st.session_state.file_started,len(full)/sr)
                audio=full[:int(elapsed*sr)]
                st.session_state.file_last_audio=audio
                st.caption(f"Playback simulation: {elapsed:.1f} / {len(full)/sr:.1f} seconds. Charts update as audio progresses.")
                if elapsed < len(full)/sr:
                    try:
                        from streamlit_autorefresh import st_autorefresh
                        st_autorefresh(interval=500,key="file-refresh-v5")
                    except Exception:
                        st.info("Automatic refresh unavailable; install streamlit-autorefresh.")
                else:
                    st.session_state.file_playing=False
                    st.session_state.file_finished=True
                    audio=full
            elif st.session_state.get("file_finished",False):
                audio=full
                st.success("Audio finished. Analysis and charts are retained below; restart to replay.")
            elif len(st.session_state.get("file_last_audio",np.array([],float))):
                audio=st.session_state.file_last_audio
                st.caption("Playback stopped. Analysis of the audio captured so far remains visible.")
            else:
                st.caption("Press Start to simulate real-time progression through the file.")
        except Exception as e: st.error(f"Could not read WAV: {e}")

duration=len(audio)/sr
if duration<baseline_secs:
    st.info(f"Collecting initial baseline: {duration:.1f}/{baseline_secs} seconds.")
else:
    feature_start=time.perf_counter()
    frames=[]
    for sec in range(int(duration)):
        f=features(audio[sec*sr:(sec+1)*sr],sr)
        if f: frames.append({"second":sec,"features":f})
    feature_elapsed=time.perf_counter()-feature_start
    st.caption(f"Local signal analysis completed in {feature_elapsed:.2f}s • Audio length: {duration:.1f}s • Gemini is a separate step and runs only if a change point is detected.")
    if len(frames)<baseline_secs: st.info("Collecting complete one-second windows.")
    else:
        baseline={k:float(np.mean([z["features"][k] for z in frames[:baseline_secs]])) for k in frames[0]["features"]}
        records=[]; onset=None
        for z in frames:
            d=deviations(baseline,z["features"]); triggered=[k for k,v in d.items() if v>=threshold]
            if z["second"]>=baseline_secs and triggered and onset is None: onset=z["second"]
            records.append({**z,"deviations":d,"triggered":triggered})
        latest=records[-1]; cur=latest["features"]; dev=latest["deviations"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Equipment",equipment); c2.metric("Elapsed audio",f"{duration:.1f} s")
        c3.metric("Estimated change onset",f"{onset} s" if onset is not None else "Not detected")
        c4.metric("Current window","Potential change" if latest["triggered"] else "No trigger")
        left=max(0,duration-shown); ix=int(left*sr); visible=audio[ix:]; tx=np.arange(len(visible))/sr+left
        fig=go.Figure(go.Scatter(x=tx,y=visible,mode="lines",name="Audio waveform",line=dict(width=1)))
        fig.add_vrect(x0=0,x1=baseline_secs,fillcolor="green",opacity=.08,line_width=0)
        if onset is not None: fig.add_vline(x=onset,line_dash="dash",line_color="red",annotation_text=f"Change starts ~{onset}s")
        fig.update_layout(title="Time Domain — continuous waveform on audio timeline",xaxis_title="Elapsed time (seconds)",yaxis_title="Amplitude",height=340)
        fig.update_xaxes(range=[left,duration]); st.plotly_chart(fig,use_container_width=True)
        nper=min(2048,len(audio)); nover=min(int(nper*.75),max(0,nper-1))
        fq,tt,S=spectrogram(audio,fs=sr,nperseg=nper,noverlap=nover)
        fig=go.Figure(go.Heatmap(x=tt,y=fq,z=10*np.log10(S+1e-12),colorbar=dict(title="dB")))
        if onset is not None: fig.add_vline(x=onset,line_dash="dash",line_color="red",annotation_text=f"Change ~{onset}s")
        fig.update_layout(title="Frequency Domain over Time — Spectrogram",xaxis_title="Elapsed time (seconds)",yaxis_title="Frequency (Hz)",height=370)
        fig.update_yaxes(range=[0,min(sr/2,5000)]); fig.update_xaxes(range=[left,duration]); st.plotly_chart(fig,use_container_width=True)
        labels={"dominant_hz":"Dominant frequency","spectral_centroid_hz":"Spectral centroid","rms":"RMS amplitude","peak":"Peak amplitude"}
        fig=go.Figure()
        for k,label in labels.items(): fig.add_trace(go.Scatter(x=[z["second"] for z in records],y=[z["deviations"][k] for z in records],mode="lines+markers",name=label))
        fig.add_hline(y=threshold,line_dash="dash",annotation_text="Candidate trigger")
        if onset is not None: fig.add_vline(x=onset,line_dash="dash",line_color="red",annotation_text=f"Onset ~{onset}s")
        fig.update_layout(title="Deviation from initial baseline",xaxis_title="Elapsed time (seconds)",yaxis_title="Deviation (%)",height=350)
        fig.update_xaxes(range=[left,duration]); st.plotly_chart(fig,use_container_width=True)
        st.subheader("Change timeline")
        st.dataframe(pd.DataFrame([{"Second":z["second"],"Dominant Hz":round(z["features"]["dominant_hz"],2),"Centroid Hz":round(z["features"]["spectral_centroid_hz"],2),"RMS":round(z["features"]["rms"],5),"Max deviation (%)":round(max(z["deviations"].values()),1),"Triggered features":", ".join(z["triggered"]) or "—"} for z in records]),use_container_width=True,hide_index=True)
        st.subheader("Gemini assessment")
        key=get_gemini_api_key()
        if onset is None: st.info("No change point crossed the configured baseline-deviation trigger yet.")
        elif not key: st.warning("Change point found, but GEMINI_API_KEY is missing. For local runs, add it to .env. For Railway, add it under your service → Variables, then redeploy.")
        else:
            cache=f"gemini_assess_{equipment}_{onset}"
            if cache not in st.session_state:
                gemini_start=time.perf_counter()
                with st.status("Gemini is assessing the detected change. The app is waiting for the API response…", expanded=True) as status:
                    st.write("Trying models in order: gemini-3.7-flash → gemini-3.6-flash → gemini-3.5-flash")
                    result,errs=gemini_assess(key,equipment,baseline,cur,dev,onset)
                    gemini_elapsed=time.perf_counter()-gemini_start
                    if result:
                        status.update(label=f"Gemini assessment completed in {gemini_elapsed:.1f}s",state="complete",expanded=False)
                    else:
                        status.update(label=f"Gemini attempts finished in {gemini_elapsed:.1f}s; all attempts failed",state="error",expanded=True)
                st.session_state[cache]={"result":result,"errors":errs,"elapsed":gemini_elapsed}
            saved=st.session_state[cache]; result=saved["result"]
            if result:
                st.caption(f"Model used: {result.get('_model_used')} • Gemini response time: {saved.get('elapsed', 0):.1f}s")
                status=result.get("status","uncertain")
                if status=="suspicious": st.warning("Gemini assessment: suspicious acoustic change")
                elif status=="normal": st.success("Gemini assessment: likely normal variation")
                else: st.info("Gemini assessment: uncertain")
                st.write(result.get("explanation_ar",""))
                if result.get("evidence"): st.write("**Evidence:** "+"; ".join(result["evidence"]))
                if result.get("possible_non_fault_causes"): st.write("**Possible non-fault causes:** "+"; ".join(result["possible_non_fault_causes"]))
                with st.expander("View raw JSON response", expanded=False):
                    st.json(result)
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(result, ensure_ascii=False, indent=2),
                        file_name=f"acoustic_assessment_{onset}s.json",
                        mime="application/json",
                    )
            else:
                st.warning("All requested Gemini model attempts failed; onset detection is still available.")
                for err in saved["errors"]: st.caption(err)
        st.caption("Change onset is the first whole-second window crossing the configured deviation trigger, not a confirmed fault time. Audio changes can also come from load, noise, mic position, or gain.")
