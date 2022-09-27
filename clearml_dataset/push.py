import clearml


def main():
    dataset = clearml.Dataset.create(
        dataset_project="CoffeBeans",
        dataset_name="coffe_beans_dataset",
        description="coffe beans dataset",
    )
    dataset.add_files("./data")
    dataset.upload(verbose=True)
    dataset.finalize(verbose=True)


if __name__ == "__main__":
    main()
