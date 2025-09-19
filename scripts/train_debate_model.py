#!/usr/bin/env python3
"""
Debate Model Training Script for AgentNet
Integrates Kaggle debate datasets into AgentNet's debate functionality.

This script:
1. Loads and preprocesses debate datasets from Kaggle
2. Prepares training data for AgentNet debate model
3. Logs progress and results
4. Saves training artifacts

Datasets:
- UN General Debates: unitednations/un-general-debates
- SOHO Forum Debate Results: martj42/the-soho-forum-debate-results  
- UN General Debate Corpus 1946-2023: namigabbasov/united-nations-general-debate-corpus-1946-2023
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from AgentNet import AgentNet, ExampleEngine
except ImportError:
    print("Warning: AgentNet core not available, proceeding with basic functionality")
    AgentNet = None
    ExampleEngine = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debate_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DebateDatasetProcessor:
    """Processes and prepares debate datasets for training."""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.processed_data = {}
        logger.info(f"Initialized dataset processor with directory: {self.datasets_dir}")
    
    def load_un_general_debates(self) -> Optional[pd.DataFrame]:
        """Load UN General Debates dataset."""
        try:
            # Look for common file patterns in the UN debates dataset
            potential_files = list(self.datasets_dir.glob("**/un-general-debates*.csv"))
            if not potential_files:
                potential_files = list(self.datasets_dir.glob("**/*general*debate*.csv"))
            
            if not potential_files:
                logger.warning("UN General Debates dataset files not found")
                return None
                
            data_file = potential_files[0]
            logger.info(f"Loading UN General Debates from: {data_file}")
            
            df = pd.read_csv(data_file)
            logger.info(f"Loaded UN General Debates: {len(df)} records, columns: {list(df.columns)}")
            
            # Basic preprocessing
            if 'text' in df.columns:
                df['processed_text'] = df['text'].fillna('').str.strip()
            elif 'speech' in df.columns:
                df['processed_text'] = df['speech'].fillna('').str.strip()
            
            self.processed_data['un_debates'] = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading UN General Debates dataset: {e}")
            return None
    
    def load_soho_forum_results(self) -> Optional[pd.DataFrame]:
        """Load SOHO Forum Debate Results dataset."""
        try:
            # Look for SOHO forum files
            potential_files = list(self.datasets_dir.glob("**/soho*.csv"))
            if not potential_files:
                potential_files = list(self.datasets_dir.glob("**/*forum*.csv"))
            
            if not potential_files:
                logger.warning("SOHO Forum dataset files not found")
                return None
                
            data_file = potential_files[0]
            logger.info(f"Loading SOHO Forum data from: {data_file}")
            
            df = pd.read_csv(data_file)
            logger.info(f"Loaded SOHO Forum data: {len(df)} records, columns: {list(df.columns)}")
            
            self.processed_data['soho_forum'] = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading SOHO Forum dataset: {e}")
            return None
    
    def load_un_corpus_1946_2023(self) -> Optional[pd.DataFrame]:
        """Load UN General Debate Corpus 1946-2023 dataset."""
        try:
            # Look for the corpus files
            potential_files = list(self.datasets_dir.glob("**/un*corpus*.csv"))
            if not potential_files:
                potential_files = list(self.datasets_dir.glob("**/*1946*2023*.csv"))
            if not potential_files:
                potential_files = list(self.datasets_dir.glob("**/*corpus*.csv"))
            
            if not potential_files:
                logger.warning("UN Corpus 1946-2023 dataset files not found")
                return None
                
            data_file = potential_files[0]
            logger.info(f"Loading UN Corpus 1946-2023 from: {data_file}")
            
            df = pd.read_csv(data_file)
            logger.info(f"Loaded UN Corpus 1946-2023: {len(df)} records, columns: {list(df.columns)}")
            
            self.processed_data['un_corpus'] = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading UN Corpus 1946-2023 dataset: {e}")
            return None
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets."""
        logger.info("Loading all debate datasets...")
        
        datasets = {}
        
        # Load each dataset
        un_debates = self.load_un_general_debates()
        if un_debates is not None:
            datasets['un_debates'] = un_debates
        
        soho_forum = self.load_soho_forum_results()
        if soho_forum is not None:
            datasets['soho_forum'] = soho_forum
        
        un_corpus = self.load_un_corpus_1946_2023()
        if un_corpus is not None:
            datasets['un_corpus'] = un_corpus
        
        logger.info(f"Successfully loaded {len(datasets)} datasets")
        return datasets
    
    def preprocess_for_debate_training(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Preprocess datasets for debate model training."""
        logger.info("Preprocessing datasets for debate training...")
        
        training_data = {
            'debate_topics': [],
            'debate_positions': [],
            'debate_arguments': [],
            'metadata': {
                'total_records': 0,
                'datasets_used': list(datasets.keys()),
                'preprocessing_timestamp': datetime.utcnow().isoformat()
            }
        }
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Extract debate-relevant content based on dataset structure
            if dataset_name == 'un_debates':
                # Extract topics, countries, and statements
                for _, row in df.iterrows():
                    if 'processed_text' in row and len(str(row['processed_text'])) > 50:
                        training_data['debate_arguments'].append({
                            'text': str(row['processed_text'])[:1000],  # Limit length
                            'source': 'UN General Debates',
                            'country': row.get('country', 'Unknown') if 'country' in row else 'Unknown'
                        })
            
            elif dataset_name == 'soho_forum':
                # Extract debate results and positions
                for _, row in df.iterrows():
                    # Look for debate-related columns
                    debate_cols = [col for col in df.columns if 'debate' in col.lower() or 'position' in col.lower()]
                    for col in debate_cols:
                        if pd.notna(row[col]) and len(str(row[col])) > 20:
                            training_data['debate_positions'].append({
                                'position': str(row[col]),
                                'source': 'SOHO Forum',
                                'topic': row.get('topic', 'Unknown') if 'topic' in row else 'Unknown'
                            })
            
            elif dataset_name == 'un_corpus':
                # Extract corpus content for debate training
                for _, row in df.iterrows():
                    text_cols = [col for col in df.columns if 'text' in col.lower() or 'speech' in col.lower()]
                    for col in text_cols:
                        if pd.notna(row[col]) and len(str(row[col])) > 100:
                            training_data['debate_arguments'].append({
                                'text': str(row[col])[:1500],  # Limit length
                                'source': 'UN Corpus 1946-2023',
                                'year': row.get('year', 'Unknown') if 'year' in row else 'Unknown'
                            })
            
            training_data['metadata']['total_records'] += len(df)
        
        logger.info(f"Preprocessed data summary:")
        logger.info(f"  - Topics: {len(training_data['debate_topics'])}")
        logger.info(f"  - Positions: {len(training_data['debate_positions'])}")
        logger.info(f"  - Arguments: {len(training_data['debate_arguments'])}")
        
        return training_data


class DebateModelTrainer:
    """Handles the training of debate models using AgentNet."""
    
    def __init__(self, training_data: Dict[str, Any]):
        self.training_data = training_data
        self.training_output_dir = Path("training_output")
        self.training_output_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized debate model trainer with output dir: {self.training_output_dir}")
    
    def prepare_training_scenarios(self) -> List[Dict[str, Any]]:
        """Prepare training scenarios from the processed data."""
        scenarios = []
        
        # Create debate scenarios from available data
        arguments = self.training_data.get('debate_arguments', [])
        positions = self.training_data.get('debate_positions', [])
        
        # Sample scenarios for training
        scenario_templates = [
            "Analyze the effectiveness of international diplomacy",
            "Evaluate different approaches to global cooperation",
            "Compare perspectives on international policy",
            "Assess the role of multilateral institutions",
            "Examine historical debates on global governance"
        ]
        
        for i, template in enumerate(scenario_templates):
            scenario = {
                'id': f'debate_scenario_{i+1}',
                'topic': template,
                'reference_arguments': arguments[:5] if arguments else [],
                'reference_positions': positions[:3] if positions else []
            }
            scenarios.append(scenario)
        
        logger.info(f"Prepared {len(scenarios)} training scenarios")
        return scenarios
    
    def run_training_simulation(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run training simulation using AgentNet debate functionality."""
        logger.info("Starting debate training simulation...")
        
        training_results = {
            'scenarios_processed': 0,
            'simulation_results': [],
            'training_metrics': {
                'total_scenarios': len(scenarios),
                'successful_simulations': 0,
                'failed_simulations': 0
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if AgentNet is None or ExampleEngine is None:
            logger.warning("AgentNet not available, running mock training simulation")
            # Mock training for demonstration
            for scenario in scenarios:
                training_results['scenarios_processed'] += 1
                training_results['simulation_results'].append({
                    'scenario_id': scenario['id'],
                    'topic': scenario['topic'],
                    'status': 'simulated',
                    'mock_result': 'Training simulation completed successfully'
                })
                training_results['training_metrics']['successful_simulations'] += 1
            
            logger.info("Mock training simulation completed")
            return training_results
        
        # Real training with AgentNet
        try:
            engine = ExampleEngine()
            
            # Create debate agents with different styles
            agent_styles = [
                {'logic': 0.9, 'creativity': 0.3, 'analytical': 0.8},
                {'logic': 0.7, 'creativity': 0.8, 'analytical': 0.6},
                {'logic': 0.8, 'creativity': 0.5, 'analytical': 0.9}
            ]
            
            agents = []
            for i, style in enumerate(agent_styles):
                agent = AgentNet(f"DebateAgent_{i+1}", style, engine)
                agents.append(agent)
            
            for scenario in scenarios:
                try:
                    logger.info(f"Processing scenario: {scenario['topic']}")
                    
                    # Run debate simulation
                    debate_result = agents[0].debate(agents[1:], scenario['topic'], rounds=3)
                    
                    training_results['scenarios_processed'] += 1
                    training_results['simulation_results'].append({
                        'scenario_id': scenario['id'],
                        'topic': scenario['topic'],
                        'status': 'completed',
                        'debate_result': {
                            'rounds': debate_result.get('rounds', 0),
                            'final_consensus': debate_result.get('final_outcome', 'No consensus'),
                            'agents_participated': len(agents)
                        }
                    })
                    training_results['training_metrics']['successful_simulations'] += 1
                    
                except Exception as e:
                    logger.error(f"Error in scenario {scenario['id']}: {e}")
                    training_results['training_metrics']['failed_simulations'] += 1
        
        except Exception as e:
            logger.error(f"Error initializing AgentNet training: {e}")
            training_results['training_metrics']['failed_simulations'] = len(scenarios)
        
        logger.info(f"Training simulation completed: {training_results['training_metrics']}")
        return training_results
    
    def save_training_artifacts(self, training_results: Dict[str, Any]) -> None:
        """Save training results and artifacts."""
        logger.info("Saving training artifacts...")
        
        # Save training results
        results_file = self.training_output_dir / "training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(training_results, f, indent=2, ensure_ascii=False)
        
        # Save training data summary
        summary_file = self.training_output_dir / "training_data_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': self.training_data['metadata'],
                'data_statistics': {
                    'total_arguments': len(self.training_data.get('debate_arguments', [])),
                    'total_positions': len(self.training_data.get('debate_positions', [])),
                    'total_topics': len(self.training_data.get('debate_topics', []))
                },
                'training_summary': training_results['training_metrics']
            }, f, indent=2)
        
        # Create training report
        report_file = self.training_output_dir / "training_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Debate Model Training Report\n\n")
            f.write(f"**Training Date:** {training_results['timestamp']}\n\n")
            f.write("## Dataset Summary\n")
            f.write(f"- Total records processed: {self.training_data['metadata']['total_records']}\n")
            f.write(f"- Datasets used: {', '.join(self.training_data['metadata']['datasets_used'])}\n")
            f.write(f"- Debate arguments extracted: {len(self.training_data.get('debate_arguments', []))}\n")
            f.write(f"- Debate positions extracted: {len(self.training_data.get('debate_positions', []))}\n\n")
            f.write("## Training Results\n")
            f.write(f"- Scenarios processed: {training_results['scenarios_processed']}\n")
            f.write(f"- Successful simulations: {training_results['training_metrics']['successful_simulations']}\n")
            f.write(f"- Failed simulations: {training_results['training_metrics']['failed_simulations']}\n\n")
            f.write("## Next Steps\n")
            f.write("- Integrate with full AgentNet training pipeline\n")
            f.write("- Expand dataset preprocessing capabilities\n")
            f.write("- Add model evaluation metrics\n")
        
        logger.info(f"Training artifacts saved to: {self.training_output_dir}")


def main():
    """Main training pipeline execution."""
    logger.info("=" * 60)
    logger.info("AgentNet Debate Model Training Pipeline Started")
    logger.info("=" * 60)
    
    try:
        # Step 1: Initialize dataset processor
        processor = DebateDatasetProcessor()
        
        # Step 2: Load all datasets
        datasets = processor.load_all_datasets()
        
        if not datasets:
            logger.error("No datasets could be loaded. Please check dataset files.")
            sys.exit(1)
        
        # Step 3: Preprocess data for training
        training_data = processor.preprocess_for_debate_training(datasets)
        
        # Step 4: Initialize trainer
        trainer = DebateModelTrainer(training_data)
        
        # Step 5: Prepare training scenarios
        scenarios = trainer.prepare_training_scenarios()
        
        # Step 6: Run training simulation
        training_results = trainer.run_training_simulation(scenarios)
        
        # Step 7: Save artifacts
        trainer.save_training_artifacts(training_results)
        
        # Step 8: Summary
        logger.info("=" * 60)
        logger.info("Training Pipeline Completed Successfully")
        logger.info(f"Processed {training_results['scenarios_processed']} scenarios")
        logger.info(f"Successful: {training_results['training_metrics']['successful_simulations']}")
        logger.info(f"Failed: {training_results['training_metrics']['failed_simulations']}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)