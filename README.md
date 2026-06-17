# Sarvam TTS Dataset Curation — Bilingual Pipeline

**Indian English (en-IN) · Malayalam (ml-IN)**

A high-quality bilingual speech dataset pipeline built for Text-to-Speech model training, developed as part of the Sarvam AI assignment. The emphasis throughout is on data quality over pipeline complexity: every design decision was made in service of producing clean, well-annotated audio segments.

**HuggingFace dataset:** [gouri005/Sarvam_Assignment](https://huggingface.co/datasets/gouri005/Sarvam_Assignment)

---

## Repository Structure

```
├── English.ipynb                  # Full English pipeline (Google Colab)
├── Malayalam__1_.ipynb            # Full Malayalam pipeline (Google Colab)
├── metadata_english.jsonl         # Per-clip metadata for all 83 English clips
├── rejected_log_eng.jsonl         # 56 rejected English clips with rejection reasons
├── english_accepted_segments.csv  # Flat CSV version of English accepted metadata


```

---

## Dataset Summary

| Language | Clips | Duration | Avg SNR | Rejection Rate |
|---|---|---|---|---|
| Indian English (en-IN) | 83 | 20.6 min | 34.1 dB | 40.3% |
| Malayalam (ml-IN) | 160 | 29.8 min | ~34 dB | 0.0% |
| **Total** | **243** | **50.4 min** | **~34 dB** | — |

---

## Source Material

**English** — A single 50-minute geopolitical news panel debate ([en_v4](https://youtu.be/x3TvDatWtqE)) with 5 speakers (moderator + 4 international panelists). The 5-second outro was trimmed, leaving 29 minutes of effective audio.

**Malayalam** — Six single-speaker YouTube videos (ml_v1–ml_v6): news readings and lectures. Per-video intros and outros were trimmed. Three additional videos were added in a second pass after the initial three yielded only 24.9 minutes (below the 30-minute target).

---

## Pipeline Architecture

### English: Diarization-First

Silence-based segmentation was attempted and abandoned — it produced meaningless speaker labels because each clip's diarization started fresh. The final pipeline diarizes the entire 29-minute audio in a single Sarvam Batch STT job before doing any slicing, ensuring speaker IDs are consistent across the full recording.

```
Download (yt-dlp, 16kHz mono WAV)
  → Trim (5s outro, librosa)
  → Full-file diarization (Sarvam Batch STT, num_speakers=5)
  → Turn slicing (turns >28s split and re-transcribed)
  → SNR gate (>15 dB)
  → Duration gate (>5s)
  → Contamination check (re-diarize each clip; reject if >1 speaker detected)
  → LLM emotion tagging (sarvam-105b → emotion, style, register, confidence)
  → Export (JSONL + CSV)
```

The contamination check is the most expensive step (55 min, 6 API batch jobs) and the most important: a clip with two voices is worse than no clip at all. It removed 30 clips (26.5% of accepted candidates).

### Malayalam: Silence-Based Segmentation + Saaras ASR

Single-speaker content makes silence-based segmentation reliable. No diarization or contamination check is needed.

```
Download (yt-dlp, 16kHz mono WAV, per video)
  → Trim (per-video intro/outro, librosa)
  → Silence-based segmentation (gap >0.4s, energy <-35 dB)
  → SNR gate (>15 dB)
  → Duration gate (5–28s)
  → ASR transcription (Sarvam Saaras v3, ml-IN)
  → Empty transcript filter (<3 characters discarded)
  → LLM emotion tagging (sarvam-105b, reasoning_effort=None)
  → Export (JSONL + CSV)
```

`reasoning_effort=None` was used for Malayalam LLM tagging — single-speaker news content has a narrow emotional range, and disabling extended reasoning reduced latency and API cost without affecting tag quality.

---

## Metadata Schema

Each accepted clip carries the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique clip identifier (e.g. `eng_0001`) |
| `file_path` | string | Path to the WAV file |
| `language` | string | BCP-47 code (`en-IN` or `ml-IN`) |
| `speaker_id` | string | Speaker label (e.g. `en_v4_spk4`) |
| `duration_sec` | float | Clip duration in seconds |
| `transcript` | string | ASR transcript |
| `emotion` | string | `neutral` / `happy` / `sad` / `excited` / `angry` / `frustrated` |
| `style` | string | `conversational` / `narrative` / `formal` / `instructional` |
| `register` | string | `colloquial` / `formal` / `code-mixed` |
| `tag_confidence` | string | `high` / `medium` / `low` |
| `snr_db` | float | Signal-to-noise ratio in dB |
| `source_url` | string | Original YouTube URL |
| `curated_at` | string | ISO 8601 timestamp |

Rejected clips in `rejected_log_eng.jsonl` carry `id`, `duration`, and `reason` (`too_short`, `overlap_contamination`, `empty_transcript`, `low_snr`).

---

## Key Design Decisions

**`num_speakers=5` (English)** — The speaker count was known in advance. Auto-detection is unreliable for 5-voice debates and produced inconsistent turn assignments across runs; setting it explicitly stabilized the output.

**Re-diarization contamination check** — Every accepted English clip was resubmitted as a standalone file for diarization. Clips where a second speaker was detected were rejected. The 26.5% contamination rate is expected for a debate with frequent interjections.

**LLM emotion tagging** — Transcripts are passed to `sarvam-105b` with a structured prompt requesting JSON output. Tags are auditable (traceable to specific transcript text) and consistent (same prompt, stable output). The model defaults to neutral with low/medium confidence when the transcript is ambiguous.

**Code-mixed register detection** — Two Malayalam clips were tagged `register: code-mixed`, identifying mid-sentence switches to English. This is linguistically significant for Malayalam TTS and allows downstream users to include or exclude these clips explicitly.

---

## Errors Encountered

| Error | Root Cause | Fix |
|---|---|---|
| `TimeoutError` (Batch STT) | Default 600s timeout too short for contamination check batches | Increased timeout to 1800s; added retry loop (3 attempts, 30s backoff) |
| `ReadTimeout` (HTTP) | Network instability between Colab and Sarvam API | Retry loop with graceful fallback (clip passed as clean on total failure) |
| `ASR 400` (audio >30s) | librosa measures 28s but upload occasionally exceeds Sarvam's 30s limit | Clips returning 400 marked as empty transcript and filtered out |
| API key exhaustion | Multiple failed contamination check runs consumed first account's credits | Updated secret from `SARVAM_API_KEY` to `SARVAM_API_KEY1`; all other cells unchanged |

---

## Output Statistics

### English

- 101 speaker turns identified → 113 clips after slicing → **83 accepted**
- Rejected: 30 overlap contamination, 26 too short
- Average clip duration: 14.9s · Average SNR: 34.1 dB

**Speaker distribution:**

| Speaker ID | Clips | Duration |
|---|---|---|
| en_v4_spk4 (likely moderator) | 31 | 7.8 min |
| en_v4_spk1 | 18 | 4.3 min |
| en_v4_spk0 | 15 | 3.8 min |
| en_v4_spk3 | 12 | 3.2 min |
| en_v4_spk2 | 7 | 1.6 min |

### Malayalam

- 162 candidate segments → **160 accepted** (2 ASR failures)
- Rejection rate: 0.0% (all clips passed SNR gate)
- Average clip duration: ~11.2s · Average SNR: ~34 dB

---

## Limitations

- **English rejection rate is high (40.3%)** — debate-format overlap is structural, not incidental. A scripted reading or clean interview would yield more usable audio per hour.
- **English speaker imbalance** — spk4 (7.8 min) vs spk2 (1.6 min). For speaker-adaptive TTS, the quieter speakers lack sufficient data.
- **Neutral-dominant emotion** — 91.6% English, 75% Malayalam clips are neutral. The dataset is suitable for natural, measured TTS but not for expressive emotion training without supplementation.
- **Single-domain content** — English is entirely geopolitical news; Malayalam is predominantly news and lectures. Models may exhibit a news-register bias.
- **Anonymous speaker IDs** — English diarization labels (spk0–spk4) are not mapped to named panelists.

---

## Tools Used

- **Sarvam Batch STT** — diarization (English)
- **Sarvam Saaras v3** — ASR transcription (Malayalam)
- **Sarvam LLM (`sarvam-105b`)** — emotion, style, register tagging
- **yt-dlp** — audio download
- **librosa** — audio loading, trimming, SNR computation
- **Google Colab** — execution environment

---

## Future Improvements

- Local Silero VAD / pyannote.audio for faster, cheaper contamination checking
- Word-level timestamp alignment (WhisperX / Montreal Forced Aligner)
- Speaking rate annotation (syllables/second from Malayalam Unicode transcript)
- Cross-modal consistency check: TTS re-synthesis and round-trip similarity scoring
- Manual speaker identity resolution for English diarization labels
- Emotion diversity supplementation (storytelling, interviews, dramatic readings)
- Dialectal subtype tagging for Malayalam register variation
