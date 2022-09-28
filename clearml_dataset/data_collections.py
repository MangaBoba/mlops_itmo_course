from sklearn.datasets import load_digits


def main():
    n_samples = 150
    n_files = 10
    digits = load_digits(return_X_y=True, as_frame=True)
    df = digits[0].join(digits[1])
    print(df)
    rows_in_file = int(n_samples / n_files)
    for i in range(n_files):
        df.iloc[i * rows_in_file: (i + 1) * rows_in_file].to_csv(
            f"./data/data_{i}.csv", index=False
        )


if __name__ == "__main__":
    main()
