import pandas as pd
import numpy as np

NUM_AUDIO_CLIPS = 23

# NOTE: make sure to remove the bottom part of the sheet (the totals, mean, std) from the csv

def calculate_accuracy(df):
    a_sums = []
    b_sums = []
    for i in range(len(df.columns) - 2):
        responses = df.iloc[:, i+2]
        model_a_sum = 0
        model_b_sum = 0
        for i in range(NUM_AUDIO_CLIPS*2):
            if ((df["Model"].iloc[i] == "A") and (responses.iloc[i] == df["Correct_answer"].iloc[i])):
                model_a_sum += 1
            if ((df["Model"].iloc[i] == "B") and (responses.iloc[i] == df["Correct_answer"].iloc[i])):
                model_b_sum += 1
        a_sums.append(model_a_sum)
        b_sums.append(model_b_sum)

    # scores
    print("a_scores: " + str(a_sums))
    print("b_scores: " + str(b_sums))
    
    # mean
    print("model_a_mean: " + str(np.mean(a_sums)))
    print("model_b_mean: " + str(np.mean(b_sums)))

    # std
    print("model_a_std: " + str(round(np.std(a_sums), 3)))
    print("model_b_std: " + str(round(np.std(b_sums), 3)))

df = pd.read_csv("eval_survey_responses - Sheet1.csv")
calculate_accuracy(df)

