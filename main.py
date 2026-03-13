import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_step_1():
    """Step 1: Data Exploration"""
    print("="*60)
    print("STEP 1: DATA EXPLORATION")
    print("="*60)
    
    from src_01_data_exploration import main as exploration_main
    df_clean, sample_10k = exploration_main()
    
    print("✅ Step 1 completed successfully!")
    return df_clean, sample_10k

def run_step_2():
    """Step 2: Data Preprocessing"""
    print("="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)
    
    from src_02_data_preprocessing import main as preprocessing_main
    df_processed, X_train, X_test, y_train, y_test = preprocessing_main()
    
    print("✅ Step 2 completed successfully!")
    return df_processed, X_train, X_test, y_train, y_test

def run_step_3():
    """Step 3: Feature Extraction"""
    print("="*60)
    print("STEP 3: FEATURE EXTRACTION")
    print("="*60)
    
    from src_03_feature_extraction import main as feature_main
    features = feature_main()
    
    print("✅ Step 3 completed successfully!")
    return features

def run_step_4():
    """Step 4: Model Training"""
    print("="*60)
    print("STEP 4: MODEL TRAINING")
    print("="*60)
    
    from src_04_model_training import main as training_main
    trainer, comparison_df = training_main()
    
    print("✅ Step 4 completed successfully!")
    return trainer, comparison_df

def run_step_5():
    """Step 5: Scaling Experiments"""
    print("="*60)
    print("STEP 5: SCALING EXPERIMENTS")
    print("="*60)
    
    from src_05_scaling_experiments import main as scaling_main
    all_results = scaling_main()
    
    print("✅ Step 5 completed successfully!")
    return all_results

def run_all_steps():
    """Run all steps sequentially"""
    print("🚀 Starting Complete Pipeline...")
    
    try:
        # Step 1
        df_clean, sample_10k = run_step_1()
        
        # Step 2
        df_processed, X_train, X_test, y_train, y_test = run_step_2()
        
        # Step 3
        features = run_step_3()
        
        # Step 4
        trainer, comparison_df = run_step_4()
        
        # Step 5
        all_results = run_step_5()
        
        print("\n🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("Check the following directories for results:")
        print("- ../data/ - Processed datasets")
        print("- ../models/ - Trained models")
        print("- ../results/ - Visualizations and metrics")
        
        return True
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Intelligent Product Review Analyzer')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5], 
                       help='Step to run (1-5). Use --step all to run complete pipeline.')
    parser.add_argument('--size', type=str, choices=['10k', '20k', '40k', '80k'], 
                       default='10k', help='Dataset size for experiments')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    print(" Intelligent Product Review Analyzer")
    
    if args.step == 1 or (hasattr(args, 'step') and args.step == 'all'):
        run_step_1()
    elif args.step == 2:
        run_step_2()
    elif args.step == 3:
        run_step_3()
    elif args.step == 4:
        run_step_4()
    elif args.step == 5:
        run_step_5()
    elif args.step == 'all' or not args.step:
        run_all_steps()
    else:
        print("Please specify a step (1-5) or use --step all")
        parser.print_help()

if __name__ == "__main__":
    main()
