from config import AppConfig
from PIL import Image
from tqdm import tqdm
import pandas as pd
import shutil
from pathlib import Path

def main_actions(config: AppConfig):

    dataset_path = config.dataset_output_path
    dataset_path.mkdir(parents=True, exist_ok=True)
    print(config.dataset_path)
    print(config.dataset_path / 'Coffee Bean.csv')
    shutil.copy(config.dataset_path / 'Coffee Bean.csv', config.dataset_output_path)
    df_info = pd.read_csv(config.dataset_path / "Coffee Bean.csv")

    for image_path in tqdm(df_info.loc[::]['filepaths'], desc="Images preparation"):
        full_path = config.dataset_path / Path(__file__).parent / image_path
        class_name = full_path.parts[-2]  # Folder Name
        stage_name = full_path.parts[-3]  # Train/Test/Val
        class_folder = dataset_path/stage_name/class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        Image.open(config.dataset_path/image_path).resize(size=(112, 112)).save(
            class_folder/Path(image_path).name)


def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()