"""
utils/state_manager.py
Enhanced state management system for Phase 1 AutoML Pipeline
Migrated and enhanced from your original setup.py
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, asdict

@dataclass
class JobMetadata:
    """Metadata for a pipeline job"""
    job_id: str
    timestamp: str
    dataset_name: str
    target_variable: str
    mode: str
    status: str
    config_used: Dict[str, Any]

class StateManager:
    """Enhanced state management with structured job tracking"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.storage_config = config.get('storage', {})
        self.base_path = self.storage_config.get('base_path', 'storage')
        self.models_path = self.storage_config.get('models_path', 'storage/models/jobs')
        
        # Initialize storage structure
        self._initialize_storage()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _initialize_storage(self):
        """Initialize the storage directory structure"""
        directories = [
            self.base_path,
            self.models_path,
            self.storage_config.get('results_path', 'storage/results'),
            self.storage_config.get('cache_path', 'storage/cache'),
            'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Create model registry if it doesn't exist
        registry_path = os.path.join(self.base_path, 'models', 'registry.json')
        if not os.path.exists(registry_path):
            self._initialize_model_registry()
    
    def _initialize_model_registry(self):
        """Initialize the model registry"""
        registry = {
            'version': '1.0.0',
            'created': datetime.now().isoformat(),
            'jobs': {},
            'statistics': {
                'total_jobs': 0,
                'successful_jobs': 0,
                'failed_jobs': 0,
                'avg_processing_time': 0.0
            }
        }
        
        registry_path = os.path.join(self.base_path, 'models', 'registry.json')
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def create_job(self, dataset_name: str, target_variable: str, mode: str = "auto") -> str:
        """Create a new job with unique ID and directory structure"""
        # Generate job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{timestamp}"
        
        # Create job directory structure
        job_path = os.path.join(self.models_path, job_id)
        os.makedirs(job_path, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['suggested_models', 'all_models', 'preprocessors', 'reports']
        for subdir in subdirs:
            os.makedirs(os.path.join(job_path, subdir), exist_ok=True)
        
        # Create job metadata
        metadata = JobMetadata(
            job_id=job_id,
            timestamp=datetime.now().isoformat(),
            dataset_name=dataset_name,
            target_variable=target_variable,
            mode=mode,
            status="initialized",
            config_used=self.config
        )
        
        # Save job metadata
        self._save_job_metadata(job_id, metadata)
        
        # Initialize job state
        self._initialize_job_state(job_id, metadata)
        
        # Update registry
        self._update_registry_on_job_creation(job_id, metadata)
        
        self.logger.info(f"Created new job: {job_id}")
        return job_id
    
    def _save_job_metadata(self, job_id: str, metadata: JobMetadata):
        """Save job metadata to file"""
        job_path = os.path.join(self.models_path, job_id)
        metadata_path = os.path.join(job_path, 'metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def _initialize_job_state(self, job_id: str, metadata: JobMetadata):
        """Initialize comprehensive job state structure"""
        state = {
            "job_info": asdict(metadata),
            "dataset_profile": {},
            "pipeline_execution": {
                "steps_completed": [],
                "current_step": "initialized",
                "progress_percentage": 0,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_duration": None,
                "errors": []
            },
            "preprocessing_results": {
                "outliers_removed": 0,
                "features_removed": [],
                "transformations_applied": {},
                "scaling_applied": False,
                "preprocessing_pipeline_saved": False
            },
            "feature_engineering": {
                "auto_sklearn_enabled": False,
                "features_created": 0,
                "polynomial_features": 0,
                "interaction_features": 0,
                "statistical_features": 0,
                "final_feature_count": 0,
                "feature_selection_applied": False,
                "feature_names": []
            },
            "model_suggestions": {
                "dataset_characteristics": {},
                "suggested_models": [],
                "reasoning": {}
            },
            "model_results": {
                "suggested_models_performance": {},
                "all_models_performance": {},
                "best_model": {},
                "model_comparison": {},
                "hyperparameter_tuning_results": {}
            },
            "inference_setup": {
                "model_ready": False,
                "preprocessing_pipeline_path": None,
                "model_path": None,
                "feature_names": [],
                "target_variable": metadata.target_variable
            },
            "evaluation_metrics": {
                "train_scores": {},
                "test_scores": {},
                "cv_scores": {},
                "residual_analysis": {},
                "feature_importance": {}
            }
        }
        
        self._save_job_state(job_id, state)
        return state
    
    def update_job_status(self, job_id: str, status: str):
        """Update job status"""
        # Update metadata
        metadata_path = os.path.join(self.models_path, job_id, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['status'] = status
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Update state
        state = self.get_job_state(job_id)
        if state:
            state['job_info']['status'] = status
            if status in ['completed', 'failed']:
                state['pipeline_execution']['end_time'] = datetime.now().isoformat()
                
                # Calculate total duration
                start_time = datetime.fromisoformat(state['pipeline_execution']['start_time'])
                end_time = datetime.now()
                duration = end_time - start_time
                state['pipeline_execution']['total_duration'] = str(duration)
            
            self._save_job_state(job_id, state)
        
        # Update registry
        self._update_registry_on_status_change(job_id, status)
    
    def update_dataset_profile(self, job_id: str, profile: Dict[str, Any]):
        """Update dataset profile in job state"""
        state = self.get_job_state(job_id)
        if state:
            state['dataset_profile'] = profile
            self._save_job_state(job_id, state)
    
    def update_preprocessing_results(self, job_id: str, results: Dict[str, Any]):
        """Update preprocessing results"""
        state = self.get_job_state(job_id)
        if state:
            state['preprocessing_results'].update(results)
            self._save_job_state(job_id, state)
    
    def update_feature_engineering_results(self, job_id: str, results: Dict[str, Any]):
        """Update feature engineering results"""
        state = self.get_job_state(job_id)
        if state:
            state['feature_engineering'].update(results)
            self._save_job_state(job_id, state)
    
    def update_model_suggestions(self, job_id: str, suggestions: List[Dict], reasoning: Dict):
        """Update model suggestions"""
        state = self.get_job_state(job_id)
        if state:
            state['model_suggestions']['suggested_models'] = suggestions
            state['model_suggestions']['reasoning'] = reasoning
            self._save_job_state(job_id, state)
    
    def update_model_results(self, job_id: str, model_name: str, results: Dict[str, Any], is_suggested: bool = True):
        """Update individual model results"""
        state = self.get_job_state(job_id)
        if state:
            if is_suggested:
                state['model_results']['suggested_models_performance'][model_name] = results
            else:
                state['model_results']['all_models_performance'][model_name] = results
            
            # Update best model if this is better
            current_best = state['model_results'].get('best_model', {})
            current_best_score = current_best.get('test_score', -float('inf'))
            
            if results.get('test_score', -float('inf')) > current_best_score:
                state['model_results']['best_model'] = {
                    'model_name': model_name,
                    **results,
                    'is_suggested': is_suggested
                }
            
            self._save_job_state(job_id, state)
    
    def setup_inference(self, job_id: str, model_path: str, preprocessor_path: str, feature_names: List[str]):
        """Setup inference configuration"""
        state = self.get_job_state(job_id)
        if state:
            state['inference_setup'] = {
                'model_ready': True,
                'preprocessing_pipeline_path': preprocessor_path,
                'model_path': model_path,
                'feature_names': feature_names,
                'target_variable': state['job_info']['target_variable']
            }
            self._save_job_state(job_id, state)
    
    def get_job_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current job state"""
        state_path = os.path.join(self.models_path, job_id, 'state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                return json.load(f)
        return None
    
    def _save_job_state(self, job_id: str, state: Dict[str, Any]):
        """Save job state to file"""
        state_path = os.path.join(self.models_path, job_id, 'state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_job_path(self, job_id: str) -> str:
        """Get job directory path"""
        return os.path.join(self.models_path, job_id)
    
    def get_model_path(self, job_id: str, model_name: str, is_suggested: bool = True) -> str:
        """Get path for saving/loading a specific model"""
        job_path = self.get_job_path(job_id)
        model_dir = 'suggested_models' if is_suggested else 'all_models'
        return os.path.join(job_path, model_dir, f"{model_name}.pkl")
    
    def get_best_model_path(self, job_id: str) -> str:
        """Get path for the best model"""
        return os.path.join(self.get_job_path(job_id), 'best_model.pkl')
    
    def get_preprocessor_path(self, job_id: str) -> str:
        """Get path for the preprocessing pipeline"""
        return os.path.join(self.get_job_path(job_id), 'preprocessors', 'pipeline.pkl')
    
    def list_jobs(self, limit: int = 50) -> List[Dict]:
        """List recent jobs"""
        jobs = []
        
        if not os.path.exists(self.models_path):
            return jobs
        
        job_dirs = [d for d in os.listdir(self.models_path) if d.startswith('job_')]
        job_dirs.sort(reverse=True)  # Most recent first
        
        for job_dir in job_dirs[:limit]:
            metadata_path = os.path.join(self.models_path, job_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                jobs.append(metadata)
        
        return jobs
    
    def cleanup_old_jobs(self):
        """Clean up old jobs based on configuration"""
        max_jobs = self.storage_config.get('max_jobs_to_keep', 50)
        if not self.storage_config.get('cleanup_old_jobs', True):
            return
        
        jobs = self.list_jobs(limit=1000)  # Get all jobs
        if len(jobs) <= max_jobs:
            return
        
        # Keep the most recent jobs, remove the rest
        jobs_to_remove = jobs[max_jobs:]
        
        for job in jobs_to_remove:
            job_id = job['job_id']
            job_path = self.get_job_path(job_id)
            
            try:
                shutil.rmtree(job_path)
                self.logger.info(f"Cleaned up old job: {job_id}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup job {job_id}: {e}")
    
    def _update_registry_on_job_creation(self, job_id: str, metadata: JobMetadata):
        """Update registry when a new job is created"""
        registry_path = os.path.join(self.base_path, 'models', 'registry.json')
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        registry['jobs'][job_id] = {
            'metadata': asdict(metadata),
            'created': datetime.now().isoformat()
        }
        registry['statistics']['total_jobs'] += 1
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _update_registry_on_status_change(self, job_id: str, status: str):
        """Update registry when job status changes"""
        registry_path = os.path.join(self.base_path, 'models', 'registry.json')
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        if job_id in registry['jobs']:
            registry['jobs'][job_id]['metadata']['status'] = status
            
            if status == 'completed':
                registry['statistics']['successful_jobs'] += 1
            elif status == 'failed':
                registry['statistics']['failed_jobs'] += 1
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        registry_path = os.path.join(self.base_path, 'models', 'registry.json')
        
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            return registry.get('statistics', {})
        
        return {}