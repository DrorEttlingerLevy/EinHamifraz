import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

class PlotSumSamples:
   
    def __init__(self, parquet_path, start_time, threshold=0.01, sampling_rate=48):
        self.parquet_path = parquet_path
        self.threshold = np.float32(threshold)
        self.sampling_rate = sampling_rate  
        self.start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")  
        

    def sum_points_above_threshold(self, arr):

        window_size = int(10 * 60 * self.sampling_rate) # 10-minute window
        step_size = int(0.1 * 60 * self.sampling_rate) #every 10-sec
        n_windows = max(0, (len(arr) - window_size) // step_size + 1)
        print(
            f"  sliding windows: {n_windows} (size={window_size}, step={step_size})",
            flush=True,
        )

        sum_samples_per_window = []

        for start in range(0, len(arr) - window_size + 1, step_size):
            window = arr[start : start + window_size]
            count_above_threshold = np.sum(window > self.threshold)
            sum_samples_per_window.append(count_above_threshold)

        print(f"  windows done: {len(sum_samples_per_window)}", flush=True)
        return sum_samples_per_window

    @staticmethod
    def _list_audio_channels(df):
        ch_cols = sorted(
            [c for c in df.columns if str(c).startswith("Ch") and str(c)[2:].isdigit()],
            key=lambda c: int(c[2:]),
        )
        if ch_cols:
            return [(int(c[2:]), c) for c in ch_cols]
        channel_cols = sorted(
            [c for c in df.columns if str(c).startswith("Channel ")],
            key=lambda c: int(str(c).split()[-1]),
        )
        return [(int(str(c).split()[-1]) - 1, c) for c in channel_cols]

    @staticmethod
    def _resolve_column(df, channel_index):
        ch_name = f"Ch{channel_index}"
        channel_name = f"Channel {channel_index + 1}"
        if ch_name in df.columns:
            return ch_name
        if channel_name in df.columns:
            return channel_name
        raise KeyError(
            f"Channel {channel_index + 1} not found; columns: {list(df.columns)}"
        )

    def load_and_concatenate_channel_data(self, channel_index):

        channel_data = []  # Temporary list to store current channel data
        n_files = len(self.parquet_path)
        print(f"Loading channel {channel_index + 1} from {n_files} files...", flush=True)

        for i, file in enumerate(self.parquet_path, start=1):
            print(f"  [{i}/{n_files}] {os.path.basename(file)}", flush=True)
            df = pd.read_parquet(file)  # Load the Parquet file
            col = self._resolve_column(df, channel_index)
            df_np = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
            channel_data.append(df_np)  # Append data to list

        concatenated = np.concatenate(channel_data)  # Return concatenated array
        print(f"  concatenated {len(concatenated)} samples", flush=True)
        return concatenated



    def save_sum_samples(self, sum_samples_dict, output_file):
        # Convert dictionary to DataFrame
        df = pd.DataFrame(dict([(f"Ch{k}", pd.Series(v)) for k, v in sum_samples_dict.items()]))
        # Save to Parquet with ZSTD compression
        step_sec = 0.1 * 60  # or whatever your step is
        df['time'] = [self.start_time + timedelta(seconds=600 + i * step_sec) for i in range(len(df))]

        print(f"Saving {len(df)} rows to {output_file}", flush=True)
        df.to_parquet(output_file, compression='zstd')
        print(f"Sum samples saved to {output_file}", flush=True)

    def process_and_plot_channel(self, channel, output_file):
        """Process a single channel. channel is 1-based (5 = Channel 5)."""
        print(f"Processing channel {channel}", flush=True)
        ch_index = channel - 1
        channel_array = self.load_and_concatenate_channel_data(ch_index)
        sum_samples = self.sum_points_above_threshold(channel_array)
        self.save_sum_samples({channel: sum_samples}, output_file)

    def process_and_plot_all_channels(self, output_file):
        sum_samples_dict = {}

        # Detect channels from first file (Ch0.. or Channel 1..)
        first_file = os.path.basename(self.parquet_path[0])
        print(f"Detecting channels from {first_file}", flush=True)
        df = pd.read_parquet(self.parquet_path[0])
        available_channels = self._list_audio_channels(df)
        names = ", ".join(name for _, name in available_channels)
        print(f"Found {len(available_channels)} channels: {names}", flush=True)

        for i, (ch_index, _) in enumerate(available_channels, start=1):
            print(
                f"Channel {ch_index + 1} ({i}/{len(available_channels)})",
                flush=True,
            )
            channel_array = self.load_and_concatenate_channel_data(ch_index)
            sum_samples = self.sum_points_above_threshold(channel_array)
            sum_samples_dict[ch_index + 1] = sum_samples

        self.save_sum_samples(sum_samples_dict, output_file)


# Define Parquet file paths

parquet_path = [
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241030-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T032_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T033_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T034_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241031-T035_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241102-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241103-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T032_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T033_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T034_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T035_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241104-T036_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T032_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T033_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T034_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241105-T035_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T032_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T033_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T034_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241106-T035_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T032_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T033_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T034_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241107-T035_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T032_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T033_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T034_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T035_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T036_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T037_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T038_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T039_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T040_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T041_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T042_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T043_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T044_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T045_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T046_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T047_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T048_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T049_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T050_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T051_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T052_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T053_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T054_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T055_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T056_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T057_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241108-T058_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T059_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T060_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T061_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T062_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T063_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T064_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T065_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T066_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T067_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T068_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T069_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T070_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T071_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241109-T072_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241110-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241111-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T032_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241112-T033_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241117-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241118-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241119-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T001_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T002_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T003_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T004_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T005_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T006_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T007_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T008_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T009_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T010_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T011_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T012_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T013_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T014_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T015_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T016_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T017_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T032_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T033_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T034_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T035_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241120-T036_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T018_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T019_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T020_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T021_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T022_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T023_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T024_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T025_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T026_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T027_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T028_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T029_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T030_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T031_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T032_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T033_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T034_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T035_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T036_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T037_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T038_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T039_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T040_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T041_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T042_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T043_lower_sr.parquet",
"/Volumes/My Book/exp_1/small_sr/small_MF/241121-T044_lower_sr.parquet",
]



# Channel to process: None = all channels, or set 1-6 for a single channel (e.g. 5 = Channel 5)
channel = None

# Define start time
start_time = "12/31/2020  23:44:50"


# Instantiate the class and process channel(s)
plotter = PlotSumSamples(parquet_path, start_time)
output_file = r'/Volumes/My Book/exp_1/sw/thr_0.01fixed.parquet'



print(
    f"Start: {len(parquet_path)} files, channel={channel if channel is not None else 'all'}, output={output_file}",
    flush=True,
)
if channel is None:
    plotter.process_and_plot_all_channels(output_file)
else:
    plotter.process_and_plot_channel(channel, output_file)



