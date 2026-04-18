# Aura Pi

Installazione audio/video per Raspberry Pi 5 con AI HAT+ 26 TOPS, camera IR fisheye USB e registrazione locale.

## Obiettivo

Questo progetto costruisce una pipeline ibrida tra videosorveglianza performativa e visual installation:

- acquisizione video da camera IR/fisheye USB
- rilevamento dei musicisti sul palco
- tracking persistente dei performer
- generazione di un'aura visiva guidata da movimento e audio
- preview/proiezione live via HDMI
- registrazione del compositing finale tramite `ffmpeg`
- export del file finale per upload su YouTube o Vimeo

## Architettura

La pipeline e' divisa in moduli:

1. `vision.py`
   Acquisizione frame da camera USB via OpenCV oppure da Picamera2 come opzione secondaria.
2. `detectors/`
   Detector dei performer. Il progetto include `motion_person` per prototipazione locale e uno scheletro `hailo_person` per Raspberry Pi 5 + AI HAT+.
3. `pipeline.py`
   Tracking delle persone, sincronizzazione con audio, orchestrazione del rendering.
4. `audio.py`
   Analisi realtime di RMS, energia e centro spettrale del segnale.
5. `renderer.py`
   Costruzione dell'aura per ciascun performer e HUD minimale.
6. `recorder.py`
   Invio del compositing a `ffmpeg` per salvataggio H.264/H.265 con audio sincronizzato.

## Perche' questa base

Su Raspberry Pi 5 + AI HAT+ 26 TOPS conviene separare:

- detection/tracking realtime
- rendering creativo
- encoding finale con `ffmpeg`

Questo permette di sostituire facilmente il detector baseline con un backend Hailo quando avrai definito il modello finale di recognition.

## Setup consigliato su Raspberry Pi

- Raspberry Pi 5
- Raspberry Pi AI HAT+ 26 TOPS
- Raspberry Pi OS aggiornato
- camera IR fisheye USB compatibile UVC
- microfono USB o interfaccia audio USB
- videoproiettore collegato via HDMI

Pacchetti di sistema tipici:

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv ffmpeg v4l-utils libatlas-base-dev
```

Ambiente Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Avvio

Configura il file YAML di esempio:

```bash
cp config/pi5.example.yaml config/local.yaml
```

Poi avvia:

```bash
PYTHONPATH=src python3 -m aura_pi.main --config config/local.yaml
```

Per trovare l'indice corretto della camera USB su Raspberry Pi:

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```

Poi imposta `video.device_index` nel file YAML.

## Strategia detection/reco

La baseline inclusa usa `motion_person`, una motion segmentation con filtro geometrico sui blob, utile per:

- prove in sala
- stage con pochi performer
- ambiente IR a basso rumore

Per arrivare a una vera recognition dei singoli musicisti, il passo successivo e':

- detector `person`
- pose estimation opzionale
- embedding/re-identification per mantenere lo stesso ID
- mappatura performer -> palette/effetto aura dedicato

## Prova Su MacBook

Si', puoi usare il MacBook come ambiente di preview per verificare la resa visiva:

- imposta `video.source: opencv`
- lascia `video.device_index: 0` per la webcam integrata
- puoi anche disattivare il recording mettendo `recording.enabled: false`
- il detector `motion_person` funziona anche sul Mac per simulare la scena

Esempio rapido:

```bash
cp config/pi5.example.yaml config/macbook.yaml
PYTHONPATH=src python3 -m aura_pi.main --config config/macbook.yaml
```

Sul Mac la pipeline usa il backend video di default di OpenCV, quindi non resta vincolata a V4L2/Linux.

Se lasci il recording attivo:

- su macOS `ffmpeg` usa di default `avfoundation` con `:0` come input audio
- su Raspberry/Linux `ffmpeg` usa di default `alsa` con device `default`
- puoi forzare il device nel file YAML se vuoi una specifica scheda audio USB

## Integrazione Hailo

Il progetto e' preparato per sostituire il detector baseline con un detector accelerato Hailo. Anche con camera USB, conviene mantenere la camera su OpenCV/V4L2 e usare Hailo solo per inferenza sui frame.

In pratica:

- usa questa base per preview, audio-reactive rendering e recording
- attiva `detector.type: hailo_person`
- punta `detector.model_path` a un modello `.hef` compatibile
- opzionalmente imposta `detector.labels_path` a un file class labels
- mantieni invariati tracking, projection e recorder

### Passaggio a Hailo sul Pi

Quando ti arriva il Raspberry Pi 5 con AI HAT+ 26 TOPS:

1. installa Raspberry Pi OS 64-bit aggiornato
2. installa lo stack AI ufficiale Raspberry Pi / Hailo
3. copia un modello person detection compatibile sul Pi
4. usa il file `config/pi5.hailo.example.yaml` come base
5. completa l'adapter in `src/aura_pi/detectors/hailo_person.py` in base al runtime Hailo installato

Lo scheletro `HailoPersonDetector` e' gia' pronto per:

- filtrare solo la label `person`
- convertire bbox normalizzate o pixel
- mantenere invariati tracker, aura e recorder

In altre parole, il punto di innesto AI e' tutto concentrato li'.

## Note USB camera

Per una camera IR fisheye USB su Pi 5, questa e' la strategia piu' semplice e robusta:

- cattura video da `/dev/video*` tramite OpenCV o backend V4L2
- exposure e gain regolati lato camera, se disponibili
- eventuale dewarping fisheye in uno step successivo del renderer
- Hailo usato solo come acceleratore inference, non come interfaccia camera

## Audio Recording

Il file video finale puo' includere anche l'audio:

- su Raspberry puoi puntare a una scheda specifica impostando `recording.audio_input_device`
- sul MacBook puoi usare il microfono interno lasciando `audio_input_format: auto` e `audio_input_device: auto`

Per scoprire i device audio disponibili:

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

su macOS, oppure:

```bash
arecord -l
```

su Raspberry Pi / Linux.

## Output

Il file prodotto da `ffmpeg` viene salvato nel percorso configurato in YAML, ad esempio:

```text
output/session_%Y%m%d_%H%M%S.mp4
```

Quel file e' gia' pensato come master locale per upload successivo su YouTube o Vimeo.
