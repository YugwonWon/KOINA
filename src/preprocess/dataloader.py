import os
import parselmouth
import numpy as np

from textgrid import TextGrid, IntervalTier, PointTier
from collections import defaultdict


class TestDataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_pairs = self._get_file_pairs()
        self.speaker_avg_pitch = self._calculate_speaker_avg_pitch()

    def _get_file_pairs(self):
        file_pairs = []
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith(".TextGrid"):
                base_name = file_name[:-9]  # Remove ".TextGrid"
                wav_file = base_name + ".wav"
                if os.path.exists(os.path.join(self.folder_path, wav_file)):
                    file_pairs.append((wav_file, file_name))
        return sorted(file_pairs)
    
    def _calculate_speaker_avg_pitch(self):
        speaker_pitches = defaultdict(list)

        for wav_file, _ in self.file_pairs:
            sound = parselmouth.Sound(os.path.join(self.folder_path, wav_file))
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Filter out unvoiced parts

            # Extract speaker ID from file name
            speaker_id = wav_file.split('_')[-1].split('.')[0]
            speaker_pitches[speaker_id].extend(pitch_values)

        speaker_avg_pitch = {speaker: np.mean(pitches) for speaker, pitches in speaker_pitches.items()}
        return speaker_avg_pitch

    def load_data(self):
        data = {}
        for wav_file, textgrid_file in self.file_pairs:
            sound = parselmouth.Sound(os.path.join(self.folder_path, wav_file))
            textgrid_path = os.path.join(self.folder_path, textgrid_file)
            textgrid = TextGrid.fromFile(textgrid_path)
            data[wav_file] = {
                'sound': sound,
                'textgrid': textgrid
            }
        return data

    def print_textgrid_info(self, textgrid):
        print("TextGrid information:")
        for tier in textgrid.tiers:
            print(f"Tier name: {tier.name}, Type: {type(tier).__name__}")
            if isinstance(tier, IntervalTier):
                for interval in tier.intervals:
                    print(f"  Interval: {interval.minTime} - {interval.maxTime}, Text: {interval.mark}")
            elif isinstance(tier, PointTier):
                for point in tier.points:
                    print(f"  Point: {point.time}, Text: {point.mark}")


if __name__ == "__main__":
    folder_path = "data/sample"
    data_loader = TestDataLoader(folder_path)
    data = data_loader.load_data()
