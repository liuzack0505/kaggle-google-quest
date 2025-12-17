import pandas as pd
import argparse
import os
import sys

def build_parser():
    parser = argparse.ArgumentParser(description='Blend predictions from 4 models (Raw Weights)')
    
    # 1. Arguments for Weights
    parser.add_argument('--w_bert', type=float, default=0.25, help='Weight for Siamese Bert')
    parser.add_argument('--w_roberta', type=float, default=0.25, help='Weight for Siamese RoBERTa')
    parser.add_argument('--w_xlnet', type=float, default=0.25, help='Weight for Siamese XLNet')
    parser.add_argument('--w_albert', type=float, default=0.25, help='Weight for Double Albert')
    
    # 2. Arguments for File Paths
    parser.add_argument('--path_bert', type=str, required=True, help='Path to bert submission csv')
    parser.add_argument('--path_roberta', type=str, required=True, help='Path to roberta submission csv')
    parser.add_argument('--path_xlnet', type=str, required=True, help='Path to xlnet submission csv')
    parser.add_argument('--path_albert', type=str, required=True, help='Path to albert submission csv')
    
    # 3. Output path
    parser.add_argument('--output_name', type=str, default='submission_blend.csv', help='Output file name')
    
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # --- 1. Load DataFrames ---
    print("Loading submission files...")
    try:
        df_bert = pd.read_csv(args.path_bert)
        df_roberta = pd.read_csv(args.path_roberta)
        df_xlnet = pd.read_csv(args.path_xlnet)
        df_albert = pd.read_csv(args.path_albert)
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
        sys.exit(1)

    # --- 2. Consistency Check ---
    print("Checking consistency...")
    if not (df_bert['qa_id'].equals(df_roberta['qa_id']) and 
            df_bert['qa_id'].equals(df_xlnet['qa_id']) and 
            df_bert['qa_id'].equals(df_albert['qa_id'])):
        print("Error: `qa_id` columns do not match across submission files!")
        sys.exit(1)
        
    # --- 3. Set Weights (Direct Use) ---
    weights = {
        'bert': args.w_bert,
        'roberta': args.w_roberta,
        'xlnet': args.w_xlnet,
        'albert': args.w_albert
    }
    
    print("-" * 30)
    print("Blending Weights (Raw - No Normalization):")
    for k, v in weights.items():
        print(f"  {k}: {v}")
    print("-" * 30)

    # --- 4. Blending ---
    target_cols = [c for c in df_bert.columns if c != 'qa_id']
    
    submission = df_bert.copy()
    
    submission[target_cols] = (
        df_bert[target_cols] * weights['bert'] +
        df_roberta[target_cols] * weights['roberta'] +
        df_xlnet[target_cols] * weights['xlnet'] +
        df_albert[target_cols] * weights['albert']
    )
    
    # --- 5. Save Output ---
    submission.to_csv(args.output_name, index=False)
    print(f"Blending complete! Saved to {args.output_name}")

if __name__ == '__main__':
    main()