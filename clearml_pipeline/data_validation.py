import pandas as pd
from config import AppConfig
from PIL import Image
from tqdm import tqdm


def main_actions(config: AppConfig):
    df_info = pd.read_csv(config.dataset_path / "Coffee Bean.csv")

    for image_path in tqdm(df_info.loc[::]["filepaths"], desc="Images validation"):
        Image.open(config.dataset_path / image_path)


def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()
