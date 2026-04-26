!pip install praat-parselmouth --quiet
from google.colab import drive
import os
import re
import pandas as pd
import wave
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

# Mount Google Drive
drive.mount('/content/drive')

# Folder Paths
cha_folder = "/content/drive/MyDrive/Aphasia Dataset/English/Protocol/ACWT PWA/ACWT TEXT"
wav_folder = "/content/drive/MyDrive/Aphasia Dataset/English/Protocol/ACWT PWA/ACWT AUDIO"
output_folder = "/content/drive/MyDrive/Aphasia Dataset/English/Protocol/ACWT PWA/ACWT Features"
os.makedirs(output_folder, exist_ok=True)

#  Word Categories
function_words = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'my', 'mine', 'your', 'yours',
                  'his', 'her', 'its', 'our', 'ours', 'their', 'theirs', 'a', 'an', 'the',
                  'and', 'but', 'or', 'if', 'while', 'because', 'so', 'of', 'in', 'on', 'at',
                  'by', 'to', 'from', 'up', 'down', 'with', 'as', 'for', 'about', 'than', 'that',
                  'is', 'was', 'are', 'were', 'be', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did'}
filler_words = {'uh', 'um', 'erm', 'mmm'}

#  Process each file
for filename in os.listdir(cha_folder):
    if filename.endswith(".cha"):
        cha_path = os.path.join(cha_folder, filename)
        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(wav_folder, base_name + ".wav")

        # Load audio + get sampling rate
        try:
            with wave.open(wav_path, 'rb') as wav_file:
                sampling_rate = wav_file.getframerate()
        except FileNotFoundError:
            print(f" Audio file not found: {wav_path}. Skipping.")
            continue

        y, sr = librosa.load(wav_path, sr=None)
        snd = parselmouth.Sound(wav_path)
# Initializing variables
        utterances = []
        prev_end_time = None
        patient_ground_truth = "Unknown"
        aq_score = None

        with open(cha_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()

            # ✅ Extract aphasia type & AQ score from @ID line
            for line in lines:
                if line.startswith("@ID:") and "|PAR|" in line:
                    parts = line.strip().split("|")
                    if len(parts) >= 6:
                        patient_ground_truth = parts[5].strip()
                    # Find AQ score from the last numeric field
                    for p in reversed(parts):
                        try:
                            aq_score = float(p.strip())
                            break
                        except:
                            continue
                    break
# Process Each uttrence line
            for line in lines:
                if line.startswith("*") and "" in line:
                    try:
                        speaker, content = line.split(":", 1)
                        text = re.sub(r".*?", "", content).strip()
                        time_matches = re.findall(r"(\d+)_(\d+)", line)

                        if time_matches:
                            start_ms, end_ms = map(int, time_matches[0])
                            duration = (end_ms - start_ms) / 1000.0
                            pause_before = (start_ms - prev_end_time) / 1000.0 if prev_end_time else None
                            prev_end_time = end_ms

                            start_sec = start_ms / 1000.0
                            end_sec = end_ms / 1000.0
                            segment = y[int(sr * start_sec):int(sr * end_sec)]
                            snd_segment = snd.extract_part(from_time=start_sec, to_time=end_sec, preserve_times=True)

                            # 🎯 Acoustic Features
                            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                            mfcc_mean = np.mean(mfccs[1])
                            mfcc_std = np.std(mfccs[1])
                            zcr_mean = np.mean(librosa.feature.zero_crossing_rate(segment))
                            rms_mean = np.mean(librosa.feature.rms(y=segment))

                            pitch = snd_segment.to_pitch()
                            freqs = pitch.selected_array['frequency']
                            voiced_freqs = freqs[freqs > 0]
                            pitch_mean = float(np.mean(voiced_freqs)) if len(voiced_freqs) > 0 else 0.0

                            try:
                                jitter = call(snd_segment, "Get jitter (local)", 0.0001, 0.02, 1.3, 1.6)
                            except:
                                jitter = 0.0
                            try:
                                shimmer = call(snd_segment, "Get shimmer (local)", 0.0001, 0.02, 1.3, 1.6)
                            except:
                                shimmer = 0.0

                            formants = snd_segment.to_formant_burg()
                            try:
                                midpoint = (start_sec + end_sec) / 2
                                formant1 = formants.get_value_at_time(1, midpoint) or 0.0
                                formant2 = formants.get_value_at_time(2, midpoint) or 0.0
                                formant3 = formants.get_value_at_time(3, midpoint) or 0.0
                            except:
                                formant1 = formant2 = formant3 = 0.0

                            # ✏️ Linguistic Features
                            words = re.findall(r"\b[\w']+\b", text.lower())
                            total_words = len(words)
                            unique_words = len(set(words))
                            ttr = unique_words / total_words if total_words > 0 else 0
                            fillers = sum(1 for w in words if w in filler_words)
                            function_count = sum(1 for w in words if w in function_words)
                            avg_word_len = sum(len(w) for w in words) / total_words if total_words > 0 else 0
                            speech_rate = total_words / duration if duration > 0 else 0
                            noun_count = len([w for w in words if w in {"cat", "dog", "man", "woman", "car"}])
                            verb_count = len([w for w in words if w in {"go", "do", "be", "have", "say"}])
                            pronoun_count = len([w for w in words if w in {'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}])

                            speaker_code = speaker.strip("*").strip()
                            speaker_type = "Control" if speaker_code == "INV" else "Patient" if speaker_code == "PAR" else "Unknown"
                            ground_truth = "Not Aphasic" if speaker_code == "INV" else patient_ground_truth

                            #  Assign AQ score
                            if speaker_code == "INV":
                                aq_value = 95.0
                            elif speaker_code == "PAR":
                                aq_value = aq_score
                            else:
                                aq_value = None

                            #  Combine all features
                            utterance = {
                                "speaker": speaker_code,
                                "speaker_type": speaker_type,
                                "ground_truth": ground_truth,
                                "sampling_rate_Hz": sampling_rate,
                                "AQ_score": aq_value,  # ✅ Added AQ score
                                "start_time_sec": round(start_sec, 3),
                                "end_time_sec": round(end_sec, 3),
                                "duration_sec": round(duration, 3),
                                "pause_before_sec": round(pause_before, 3) if pause_before else None,
                                "text": text,
                                "total_words": total_words,
                                "unique_words": unique_words,
                                "ttr": round(ttr, 3),
                                "avg_word_length": round(avg_word_len, 2),
                                "speech_rate_wps": round(speech_rate, 2),
                                "filler_count": fillers,
                                "function_word_count": function_count,
                                "noun_count": noun_count,
                                "verb_count": verb_count,
                                "pronoun_count": pronoun_count,
                                "mfcc_mean": round(mfcc_mean, 3),
                                "mfcc_std": round(mfcc_std, 3),
                                "zcr_mean": round(zcr_mean, 5),
                                "rms_mean": round(rms_mean, 5),
                                "pitch_mean": round(pitch_mean, 2),
                                "jitter": round(jitter, 5),
                                "shimmer": round(shimmer, 5),
                                "formant1": round(formant1, 2),
                                "formant2": round(formant2, 2),
                                "formant3": round(formant3, 2)
                            }

                            utterances.append(utterance)

                    except Exception as e:
                        print(f"❌ Error in line: {line}\n   {e}")

        if utterances:
            df = pd.DataFrame(utterances)
            output_path = os.path.join(output_folder, f"{base_name}_features.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Saved: {output_path}")
        else:
            print(f"⚠️ No valid utterances found in: {filename}")

print(" All files processed!")
