import pandas
import hazm
import matplotlib.pyplot as plt
import seaborn


if __name__ == '__main__':
    train_dataset_df = pandas.read_csv('../../../data/Processed/train_data_normed.csv')
    test_dataset_df = pandas.read_csv('../../../data/Processed/test_data_normed.csv')
    valid_dataset_df = pandas.read_csv('../../../data/Processed/valid_data_normed.csv')

    all_datasets = [train_dataset_df, test_dataset_df, valid_dataset_df]
    all_datasets = pandas.concat(all_datasets)
    all_len = []
    for item in all_datasets.text:
        all_len.append(len(hazm.word_tokenize(str(item))))

    data_frame_len = pandas.DataFrame({'lens':all_len})
    data_frame_len.to_excel('lens.xlsx', index=False)
