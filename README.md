# Supertonic 3 on SiMa Modalix

Hybrid text-to-speech for a SiMa Modalix DevKit. The duration predictor and
text encoder run once per utterance in persistent ONNX Runtime sessions. The
eight-step vector field and fixed-profile vocoder use persistent PyNeat model
runners on MLA. The host performs classifier-free guidance, Euler updates,
and WAV serialization.

The upstream model is pinned to revision
`724fb5abbf5502583fb520898d45929e62f02c0b`.

## Repository layout

```text
compilation/
  reference_cases.py      deterministic end-to-end validation fixtures
  vector_estimator/
    graph_surgery.py       public surgery pipeline
    transforms.py          private transformation implementation
    contract.py            fixed source and compiled tensor contracts
    compile.py             vector-estimator MPK packaging
  vocoder/
    graph_surgery.py       complete vocoder surgery pipeline
    compile.py             vocoder MPK packaging
app/
  supertonic_sima/        persistent hybrid runtime
  examples/
    simple.py             WAV-producing CLI with latency and RTF
    gui.py                browser voice/language playground
    speech_server.py      POST /v1/speech
scripts/
  setup_devkit.sh
requirements-devkit.txt
```

Generated models, compiler output, reports, and audio belong under ignored
workspace paths such as `build/`, `models/`, or the configured application root.

## Fresh DevKit setup

Run this repository on the DevKit. The script creates an isolated virtual
environment under `${SUPERTONIC_APP_ROOT:-$HOME/supertonic-tts}`, downloads the
matching PyNeat wheel with `sima-cli neat install core -t pyneat`, installs ONNX
Runtime and PyNeat into that environment, and downloads the upstream CPU models
plus compiled MLA models. An existing application environment is recreated to
keep the install isolated and reproducible.

```bash
export SUPERTONIC_APP_ROOT="$HOME/supertonic-tts"
bash scripts/setup_devkit.sh
source "$SUPERTONIC_APP_ROOT/.venv/bin/activate"
```

The scripts and Python runtime honor the same path configuration:

- `SUPERTONIC_APP_ROOT` defaults to `$HOME/supertonic-tts`; a fresh setup puts
  its virtual environment there.
- `SUPERTONIC_MODEL_ROOT` defaults to `$SUPERTONIC_APP_ROOT/models`.
- `SUPERTONIC_OUTPUT_ROOT` defaults to `$SUPERTONIC_APP_ROOT/output`.
- `SUPERTONIC_CACHE_ROOT` defaults to
  `${XDG_CACHE_HOME:-$HOME/.cache}/supertonic-tts` for setup and downloads.
- `SUPERTONIC_REPO_ROOT` defaults to the repository containing the setup script.
- `SUPERTONIC_HF_CLI` optionally selects an existing `hf` executable.

Set paths to absolute values before setup and before launching an example. CLI
path arguments still take precedence over the environment-derived defaults.
The DevKit must have `sima-cli` configured only when performing the fresh setup
that downloads PyNeat.

### Reuse installed Neat and PyNeat

When the board already has Neat and PyNeat, do not run `setup_devkit.sh`; that
script deliberately recreates an isolated environment. Download the models
directly instead. `download_models.sh` reuses `hf` from `PATH` (or
`SUPERTONIC_HF_CLI`) and does not install or modify Neat or PyNeat.

For the shared workspace on Modalix:

```bash
export SUPERTONIC_APP_ROOT=/workspace/GitHub/supertonic-sima
export SUPERTONIC_MODEL_ROOT=/workspace/llima/models
export SUPERTONIC_OUTPUT_ROOT="$HOME/supertonic-output"
export SUPERTONIC_CACHE_ROOT="$HOME/.cache/supertonic-tts"
cd "$SUPERTONIC_APP_ROOT"
bash scripts/download_models.sh
```

Use the existing PyNeat interpreter for the application. If ONNX Runtime is
not already present, install it once without reinstalling PyNeat:

```bash
source /home/sima/pyneat/bin/activate
python -m pip install onnxruntime==1.22.1
```

## Examples

Generate a WAV. One warmup runs after constructing the sessions/runners; all
measured runs reuse those objects.

```bash
python app/examples/simple.py \
  --text "Hello from Modalix." --voice M1 --lang en \
  --output "${SUPERTONIC_OUTPUT_ROOT:-$SUPERTONIC_APP_ROOT/output}/hello.wav"
```

The CLI prints audio length, generation time, real-time factor, latent length,
and per-stage latency. The default is eight denoising steps; use `--steps 12`
to compare the higher-quality schedule, `--runs N` for repeated warm
measurements, or `--vocoder-backend cpu` to compare the upstream CPU vocoder.

Start the browser UI:

```bash
python app/examples/gui.py --host 0.0.0.0 --port 8080
```

The browser fills the 192-character processed-text envelope and splits long
text at `.`, `!`, `?`, `:`, and `;` boundaries, including their Japanese/CJK
forms such as `。`, `！`, `？`, and `…`. An oversized segment falls back to
commas, then whitespace or a hard boundary. It synthesizes chunks sequentially
and starts playback as soon as the first chunk is ready while subsequent chunks
are synthesized into the continuous Web Audio queue. Inter-chunk outputs have
their generated trailing silence removed with a short retained tail and fade;
the final chunk remains unchanged.

If a chunk still exceeds the 192-frame latent-duration profile, the browser
automatically splits only that chunk to a safer size and retries it before
continuing the playback queue.

The server logs each chunk's complete JSON-escaped text together with its
position and split boundary, raw/processed lengths, synthesis settings, latent
frames, audio duration, generation time, and RTF. The browser console
additionally reports the complete chunk plan and per-chunk trimming measurements.

Start the speech endpoint:

```bash
python app/examples/speech_server.py --host 0.0.0.0 --port 8000
```

```bash
curl -sS http://127.0.0.1:8000/v1/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Guten Tag aus dem Modalix DevKit.","voice":"F1","language":"de"}' \
  --output speech.wav
```

Only WAV output is supported. Response headers include
`X-Audio-Length-Seconds`, `X-Generation-Length-Seconds`,
`X-Real-Time-Factor`, and `X-Latent-Length`. Requests are serialized around
the persistent MLA runners.

### Listen to Jarvic from a host browser

Keep the Jarvic terminal client on Modalix and open
`http://MODALIX_IP:8080/listen` on your laptop. Click **Enable audio** once and
leave the page open. Then run on Modalix, from the Jarvic repository:

```bash
./scripts/devkit/start_jarvic_chat_client.sh --tts \
  --tts-url http://127.0.0.1:8080 --tts-playback browser
```

Restart the Supertonic GUI after updating this repository. The same listener page
is available on `speech_server.py`; use port 8000 in both URLs for that server.
The server must listen on `0.0.0.0` (the examples' default). No Python environment,
audio player installation, or Jarvic client is needed on the laptop.

Choose **Voice** (F1–F5 or M1–M5), **Language**, and **Speed** (0.7×–2.0×) on the
listener page for the next Jarvic response. Language controls speech pronunciation;
it does not translate Jarvic's text. All chunks of a response keep the same settings,
even if you change them while it is speaking. **Use Jarvic voice**, **Use Jarvic
language**, and clearing the speed field restore the corresponding defaults from
Jarvic's YAML or CLI flags. Settings are shared across listeners, survive refreshing
the page, and reset when Supertonic restarts. Direct speech requests and the original
GUI keep using their own settings.

Audio is delivered live to every browser that has enabled listening. **Stop**
disconnects that browser and clears its audio queue. Joining or reconnecting does
not replay earlier speech. Browsers require the initial click to enable sound.
The browser schedules WAV chunks in order while later chunks arrive. Slow
listeners are disconnected when their bounded queues fill and can click
Enable audio to resume with new speech.

When Jarvic starts a new assistant response, it interrupts the previous speech
on the first text delta. Queued and currently playing browser audio is flushed
without disconnecting the listener. Any old chunk still being decoded or
synthesized is discarded; the new response plays automatically. An inference
call already running on Modalix finishes normally, so the next synthesis may
wait for that call even though playback has already stopped.

For integrations, `POST /v1/speech/broadcast` accepts the same text/voice/language
payload as `/v1/speech`, synthesizes once, sends it to connected listeners, and
returns `{"listeners": N}`. Without listeners it returns HTTP 409 before running
inference. Model profile errors remain HTTP 400 so clients can split and retry.
`GET /listen/events` carries audio as base64 WAV in server-sent events; `/health`
includes `browser_listeners`. Ordinary `/v1/speech` requests continue returning
WAV directly and are not broadcast.

To replace a response, call `POST /v1/speech/interrupt` with a unique client
`stream_id` and an increasing integer `sequence`, starting at 1. It returns a
`generation` to include in every `/v1/speech/broadcast` chunk of that response.
An `interrupt` event flushes each listener's queue. Audio events carry their
generation, and stale requests/results return HTTP 202 with
`{"superseded": true}` instead of emitting audio. Repeating the same current
control request is idempotent; delayed older sequence numbers are ignored.

`GET /listen/settings` returns the available voices, languages, speed bounds, and
shared settings. `POST /listen/settings` accepts any non-empty subset of
`{"voice": "F2", "language": "en", "speed": 1.2}`; omitted settings stay unchanged.
Set any value to `null` to restore that client-provided setting. Invalid updates
return HTTP 400 without changing any settings. Settings are captured when a
response begins with `/v1/speech/interrupt`; updates reach connected pages as
`settings` events and do not interrupt current speech.

### Run Supertonic and Jarvic on separate DevKits

Supertonic can run on one Modalix DevKit while the entire Jarvic stack, including
the terminal chat client, runs on another. Your laptop plays the audio through
Supertonic's browser listener. Voice, language, speed, and interruption work the
same way as when both applications run on one DevKit.

Replace `TTS_DEVKIT_IP` with the Supertonic DevKit's IP address and
`JARVIC_DEVKIT_IP` with the Jarvic DevKit's IP address.

1. On the **Supertonic DevKit**, activate its existing Python environment and
   retain the model path settings described above, then start:

   ```bash
   cd /workspace/GitHub/supertonic-sima
   python app/examples/speech_server.py --host 0.0.0.0 --port 8000
   ```

   Wait until startup prints the `listening=` and `listener=` URLs.

2. On the **Jarvic DevKit**, check that Supertonic is reachable, then start chat:

   ```bash
   curl -fsS http://TTS_DEVKIT_IP:8000/health

   cd /workspace/GitHub/jarvic-framework-feature-sima_with_fpf
   ./scripts/devkit/start_jarvic_chat_client.sh \
     --tts --tts-url http://TTS_DEVKIT_IP:8000 \
     --tts-playback browser
   ```

3. On your **laptop**, open `http://TTS_DEVKIT_IP:8000/listen`, click
   **Enable audio**, and keep the page open while chatting.

Use the port that Supertonic actually listens on in both the Jarvic URL and
the browser URL. `speech_server.py` defaults to 8000; `gui.py` defaults to 8080.
Binding to `0.0.0.0` accepts connections through both the DevKit's network address
and localhost. Binding to a specific IP only accepts connections through that
address.

#### Relay through the laptop when the DevKits cannot reach each other

Both DevKits being accessible from your laptop does not guarantee that they can
reach each other. If they are on separate networks and the health check from the
Jarvic DevKit fails, first confirm that your laptop can reach
`http://TTS_DEVKIT_IP:8000/health`. Then run this SSH tunnel **on your laptop**:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:18000:TTS_DEVKIT_IP:8000 \
  sima@JARVIC_DEVKIT_IP
```

This opens port 18000 on the Jarvic DevKit's loopback interface. Requests to that
port travel through SSH to your laptop, which connects to the Supertonic DevKit.
Keep the laptop and SSH session running while using the relay.

On the **Jarvic DevKit**, check the tunnel and use its local URL for TTS:

```bash
curl -fsS http://127.0.0.1:18000/health

./scripts/devkit/start_jarvic_chat_client.sh \
  --tts --tts-url http://127.0.0.1:18000 \
  --tts-playback browser
```

The laptop's browser still opens `http://TTS_DEVKIT_IP:8000/listen` directly.

## Compiled contracts

The vector estimator accepts eight contiguous FP32 tensors in this order:

```text
noisy_latent       [1,144,1,192]
text_emb           [1,256,1,192]
style_ttl          [1,256,1,50]
style_key          [1,256,1,50]
latent_mask        [1,1,1,192]
text_mask          [1,192,1,1]
time_sinusoidal    [1,64,1,1]
rope_tables        [1,128,1,192]
```

It emits FP32 `velocity [1,144,1,192]`. The vocoder accepts FP32
`latent [1,144,1,192]` and emits FP32 `wav_frames [1,512,1,1152]`; public HWC
order is already time-major, so the runtime crops a flat view to the natural
predicted length.

The host creates the denoising timestep embeddings in memory for the selected
`--steps` value. Runtime data stores only the RoPE bank and CFG/style constants;
the app remains compatible with older archives that also contain a timestep
table.

Processed text and predicted latent length must both be at most 192. The
latent profile represents about 13.37 seconds at 44.1 kHz.

## Graph surgery and compilation

Keep the graph and released-package environments separate, then generate the
deterministic upstream reference cases:

```bash
python3 -m venv .venv-graph
.venv-graph/bin/pip install -r compilation/requirements.txt
python3 -m venv .venv-reference
.venv-reference/bin/pip install supertonic==1.3.1
.venv-reference/bin/python compilation/reference_cases.py \
  --model-dir models/supertonic-3 \
  --output-dir build/reference-cases \
  --summary build/reference-cases.json
```

The vector transformation validates those NPZs and produces the externalized
runtime table:

```bash
.venv-graph/bin/python compilation/vector_estimator/graph_surgery.py \
  --source models/supertonic-3/onnx/vector_estimator.onnx \
  --reference-dir build/reference-cases \
  --output-dir build/vector-surgery
```

The vocoder transformation accepts repeatable reference cases for optional
waveform equivalence validation:

```bash
.venv-graph/bin/python compilation/vocoder/graph_surgery.py \
  --input models/supertonic-3/onnx/vocoder.onnx \
  --output build/vocoder-surgery/supertonic_vocoder_sima.onnx \
  --report build/vocoder-surgery/report.json \
  --reference-case build/reference-cases/en_m1_short.npz
```

After quantizing each graph with the SiMa model compiler, package the saved
single-MLA network with the component command:

```bash
python compilation/vector_estimator/compile.py \
  --network-directory build/vector-quantized/supertonic_vector_field_sima \
  --output-directory build/vector-compiled \
  --compiler-debug-directory build/vector-compiler-debug

python compilation/vocoder/compile.py \
  --network-directory build/vocoder-quantized/supertonic_vocoder_sima \
  --output-directory build/vocoder-compiled \
  --compiler-debug-directory build/vocoder-compiler-debug
```

The vector profile uses BF16 activations and symmetric per-channel INT8
weights. The vocoder must use BF16 activations and BF16 weights; INT8 weights
damage its quantization-sensitive output head.
