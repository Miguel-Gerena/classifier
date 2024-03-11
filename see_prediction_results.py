import pandas as pd
import json
import argparse
import os
import matplotlib.pyplot as plt

def load_predictions(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def filter_metadata(metadata_df, patent_numbers):
    # return metadata_df[metadata_df['patent_number'].isin(patent_numbers)]
    return metadata_df[metadata_df['application_number'].isin(patent_numbers)]

def visualize_feature_distribution(output_dir, correct_df, incorrect_df, feature_name, feature_type='categorical'):

    thresh = 5
    plt.figure(figsize=(14, 8))
    
    if feature_type == 'categorical':
        # Combine data for plots
        combined_df = pd.concat([
            correct_df[[feature_name]].assign(Prediction='Correct'),
            incorrect_df[[feature_name]].assign(Prediction='Incorrect')
        ])
        
        if feature_name == 'uspc_class' or feature_name == 'uspc_subclass' or feature_name == 'examiner_full_name':    
            counts = combined_df.groupby([feature_name, 'Prediction']).size().unstack(fill_value=0)
            
            # Calculate the difference and filter based on threshold
            counts['Difference'] = (counts['Correct'] - counts['Incorrect']).abs()
            filtered_counts = counts[counts['Difference'] > thresh]
            
            # Plot only if there are entries to plot
            if not filtered_counts.empty:
                filtered_counts.drop(columns='Difference').plot(kind='bar', width=0.8)
                plt.title(f'Distribution of {feature_name} with Difference > {thresh} for Correct vs. Incorrect Predictions')
                plt.ylabel('Count')
                plt.xticks(rotation=90)
            else:
                print(f"No entries for {feature_name} with difference > {thresh}")
        else:
            combined_df.groupby([feature_name, 'Prediction']).size().unstack().plot(kind='bar', stacked=False)
            plt.title(f'Distribution of {feature_name} for Correct vs. Incorrect Predictions')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
    
    elif feature_type == 'numerical':
        # Plot
        plt.boxplot([correct_df[feature_name].dropna(), incorrect_df[feature_name].dropna()], labels=['Correct', 'Incorrect'])
        plt.title(f'Distribution of {feature_name} for Correct vs. Incorrect Predictions')
        plt.ylabel(feature_name)
        plt.xticks(rotation=0)
    
    save_path = os.path.join(output_dir, feature_name)
    plt.savefig(save_path)
    plt.close()


def main(correct_predictions_path, incorrect_predictions_path, metadata_feather_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # Load predictions and metadata
    correct_patent_numbers = load_predictions(correct_predictions_path)
    incorrect_patent_numbers = load_predictions(incorrect_predictions_path)

    metadata_df = pd.read_feather(metadata_feather_path)

    # print("Sample patent numbers from JSON:", correct_patent_numbers[:5])
    # print("Sample patent numbers from DataFrame:", metadata_df['patent_number'].head().tolist())
    # exit()


    # Filter metadata for correct and incorrect predictions
    correct_metadata_df = filter_metadata(metadata_df, correct_patent_numbers)
    incorrect_metadata_df = filter_metadata(metadata_df, incorrect_patent_numbers)
    # print("CORRECT META:", correct_metadata_df[:2])
    # print("INCORRECT META:", incorrect_metadata_df[:2])
    # exit()

    # # Save filtered metadata to new files
    # correct_metadata_df.to_csv(os.path.join(output_dir, 'correct_metadata.csv'), index=False)
    # incorrect_metadata_df.to_csv(os.path.join(output_dir, 'incorrect_metadata.csv'), index=False)

    visualize_feature_distribution(output_dir, correct_metadata_df, incorrect_metadata_df, 'examiner_full_name', 'categorical')
    visualize_feature_distribution(output_dir, correct_metadata_df, incorrect_metadata_df, 'uspc_class', 'categorical')
    visualize_feature_distribution(output_dir, correct_metadata_df, incorrect_metadata_df, 'uspc_subclass', 'categorical')
    visualize_feature_distribution(output_dir, correct_metadata_df, incorrect_metadata_df, 'decision_as_of_2020', 'categorical')

    # visualize_feature_distribution(output_dir, correct_metadata_df, incorrect_metadata_df, 'date_application_published', 'numerical')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    default_model_name = "claims_distilbert-base-uncased_2_8_2e-05_512_False_all_False_date_3_2_hr_21"

    default_model_path = "CS224N_models/distilbert-base-uncased/claims_distilbert-base-uncased_2_8_2e-05_512_False_all_False_date_3_2_hr_21/epoch_"
    parser.add_argument('--model_path', default=default_model_path + "model", type=str, help='model path to target model')
    parser.add_argument('--model_name', default=default_model_name , type=str, help='target model name')

    args = parser.parse_args()

    correct_predictions_path = os.path.join(args.model_path, 'correct_predictions_patent_num.json')
    incorrect_predictions_path = os.path.join(args.model_path, 'incorrect_predictions_patent_num.json')
    metadata_feather_path = './hupd_metadata_2022-02-22.feather'
    output_dir = './prediction_results/' + args.model_name

    main(correct_predictions_path, incorrect_predictions_path, metadata_feather_path, output_dir)
