import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

class PlotSumSamples:
   
    def __init__(self, parquet_path, start_time, threshold=0.0005, sampling_rate=48):
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
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902141500_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902141506_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902150910_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902155038_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902163202_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902171328_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902175452_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902183618_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902191742_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902195908_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902204032_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902212158_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902220322_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902224448_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250902232612_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903000738_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903004902_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903013028_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903021152_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903025318_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903033442_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903041608_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903045732_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903053858_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903062022_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903065242_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903073430_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903081558_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903085722_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903093848_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903102012_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903110138_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903114302_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903122428_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903130552_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903134718_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903142842_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903151008_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903155132_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903163258_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903171422_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903175548_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903183712_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903191838_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903200002_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903204128_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903212252_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903220418_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903224542_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250903232708_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904000832_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904004958_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904013122_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904021248_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904025412_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904033538_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904041702_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904045828_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904053952_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904062118_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904070242_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904074122_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904082308_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904090434_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904094558_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904102724_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904110848_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904115014_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904123138_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904131304_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904135428_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904143554_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904151718_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904155844_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904164008_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904172134_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904180258_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904184424_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904192548_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904200714_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904204838_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904213004_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904221128_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904225254_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250904233418_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905001544_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905005708_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905013834_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905021958_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905030124_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905034248_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905042414_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905050538_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905054704_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905062828_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905064358_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905093958_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905102122_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905110248_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905114412_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905122538_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905130702_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905134828_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905142952_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905151118_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905155242_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905163408_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905171532_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905175658_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905183822_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905191948_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905200112_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905204238_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905212402_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905220528_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905224652_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250905232818_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906000942_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906005108_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906013232_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906021358_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906025522_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906033648_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906041812_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906045938_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906054102_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906062228_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906070352_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906074518_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906082642_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906090808_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906094932_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906103058_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906111222_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906115348_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906123512_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906131638_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906135802_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906143928_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906152052_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906160218_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906164342_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906172508_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906180632_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906184758_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906192922_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906201048_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906205212_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906213338_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906221502_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906225628_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250906233752_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907001918_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907010042_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907014208_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907022332_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907030458_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907034622_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907042748_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907050912_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907055038_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907063202_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907071328_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907075452_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907082532_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907090712_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907094838_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907103002_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907111128_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907115252_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907123418_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907131542_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907135708_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907143832_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907151958_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907160122_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907164248_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907172412_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907180538_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907184702_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907192828_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907200952_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907205118_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907213242_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907221408_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907225532_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250907233658_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908001822_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908005948_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908014112_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908022238_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908030402_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908034528_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908042652_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908050818_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908054942_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908063108_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908063900_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908072038_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908080204_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908084328_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908092454_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908100618_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908104744_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908112908_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908121034_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908125158_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908133324_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908141448_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908145614_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908153738_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908161904_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908170028_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908174154_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908182318_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908190444_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908194608_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908202734_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908210858_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908215024_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908223148_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908231314_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250908235438_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909003604_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909011728_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909015854_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909024018_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909032144_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909040308_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909044434_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909052558_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909060724_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909064848_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909071834_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909080026_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909084150_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909092316_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909100440_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909104606_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909112730_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909120856_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909125020_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909133146_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909141310_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909145436_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909153600_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909161726_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909165850_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909174016_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909182140_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909190306_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909194430_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909202556_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909210720_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909214846_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909223010_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909231136_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250909235300_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910003426_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910011550_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910015716_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910023840_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910032006_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910040130_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910044256_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910052420_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910060546_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910064648_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910072824_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910080950_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910085114_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910093240_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910101404_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910105530_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910113654_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910121820_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910125944_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910134110_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910142234_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910150400_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910154524_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910162650_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910170814_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910174940_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910183104_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910191230_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910195354_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910203520_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910211644_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910215810_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910223934_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250910232100_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911000224_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911004350_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911012514_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911020640_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911024804_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911032930_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911041054_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911045220_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911053344_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911061510_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911064724_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911072904_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911081030_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911085154_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911093320_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911101444_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911105610_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911113736_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911121900_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911130024_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911134150_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911142316_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911150440_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911154604_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911162730_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911170856_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911175020_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911183146_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911191310_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911195434_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911203600_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911211726_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911215850_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911224016_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250911232140_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912000306_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912004430_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912012556_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912020720_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912024846_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912033010_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912041136_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912045300_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912053424_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912061550_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912064022_matched_exp_1_karnel.parquet",
r"H:\Exp_Ein_Hamifratz\MF_sr_all\20250912072202_matched_exp_1_karnel.parquet",

]



# Channel to process: None = all channels, or set 1-6 for a single channel (e.g. 5 = Channel 5)
channel = None

# Define start time
start_time = "12/31/2020  23:44:50"


# Instantiate the class and process channel(s)
plotter = PlotSumSamples(parquet_path, start_time)
output_file = r"H:\Exp_Ein_Hamifratz\output_111026\row1_thr_0.0005.parquet"
print(
    f"Start: {len(parquet_path)} files, channel={channel if channel is not None else 'all'}, output={output_file}",
    flush=True,
)
if channel is None:
    plotter.process_and_plot_all_channels(output_file)
else:
    plotter.process_and_plot_channel(channel, output_file)



