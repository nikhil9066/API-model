"""
utils/progress_tracker.py
Comprehensive progress tracking system for Phase 1 AutoML Pipeline
"""

import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os

@dataclass
class ProgressStep:
    """Individual progress step"""
    name: str
    description: str
    weight: float  # Relative weight for progress calculation
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    sub_steps: List['ProgressStep'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def progress_percentage(self) -> float:
        if self.status == "completed":
            return 100.0
        elif self.status == "failed":
            return 0.0
        elif self.status == "running":
            if self.sub_steps:
                completed_weight = sum(step.weight for step in self.sub_steps if step.status == "completed")
                total_weight = sum(step.weight for step in self.sub_steps)
                return (completed_weight / total_weight) * 100.0 if total_weight > 0 else 0.0
            return 50.0  # Assume 50% if running without sub-steps
        return 0.0  # pending

class ProgressTracker:
    """Comprehensive progress tracking with console and file logging"""
    
    def __init__(self, job_id: str, config: Dict, console_output: bool = True, log_to_file: bool = True):
        self.job_id = job_id
        self.config = config
        self.console_output = console_output
        self.log_to_file = log_to_file
        
        # Progress tracking
        self.steps: List[ProgressStep] = []
        self.current_step_index = 0
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        # Logging setup
        self.logger = self._setup_logging()
        
        # Progress callbacks
        self.progress_callbacks: List[Callable] = []
        
        # Define pipeline steps for Phase 1
        self._initialize_pipeline_steps()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"progress_{self.job_id}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        if self.log_to_file:
            log_dir = self.config.get('logging', {}).get('log_dir', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"progress_{self.job_id}.log")
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_pipeline_steps(self):
        """Initialize the standard Phase 1 pipeline steps"""
        self.steps = [
            ProgressStep(
                name="data_validation",
                description="Data Validation & Profiling",
                weight=0.1,
                sub_steps=[
                    ProgressStep("file_validation", "Validating file format and size", 0.3),
                    ProgressStep("data_loading", "Loading dataset", 0.2),
                    ProgressStep("quality_check", "Checking data quality", 0.3),
                    ProgressStep("profiling", "Creating dataset profile", 0.2)
                ]
            ),
            ProgressStep(
                name="preprocessing",
                description="Data Preprocessing",
                weight=0.25,
                sub_steps=[
                    ProgressStep("outlier_detection", "Detecting and removing outliers", 0.4),
                    ProgressStep("correlation_analysis", "Analyzing feature correlations", 0.3),
                    ProgressStep("skewness_handling", "Handling skewed features", 0.3)
                ]
            ),
            ProgressStep(
                name="feature_engineering",
                description="Automated Feature Engineering",
                weight=0.25,
                sub_steps=[
                    ProgressStep("auto_sklearn", "Auto-sklearn feature generation", 0.5),
                    ProgressStep("polynomial_features", "Creating polynomial features", 0.2),
                    ProgressStep("interaction_features", "Creating interaction features", 0.2),
                    ProgressStep("feature_selection", "Selecting best features", 0.1)
                ]
            ),
            ProgressStep(
                name="model_selection",
                description="Smart Model Selection & Training",
                weight=0.3,
                sub_steps=[
                    ProgressStep("dataset_analysis", "Analyzing dataset characteristics", 0.1),
                    ProgressStep("model_suggestions", "Generating model suggestions", 0.1),
                    ProgressStep("model_training", "Training selected models", 0.6),
                    ProgressStep("hyperparameter_tuning", "Hyperparameter optimization", 0.2)
                ]
            ),
            ProgressStep(
                name="evaluation",
                description="Model Evaluation & Results",
                weight=0.1,
                sub_steps=[
                    ProgressStep("performance_evaluation", "Evaluating model performance", 0.4),
                    ProgressStep("model_comparison", "Comparing model results", 0.3),
                    ProgressStep("result_saving", "Saving models and results", 0.3)
                ]
            )
        ]
    
    def start_step(self, step_name: str, description: Optional[str] = None, metadata: Optional[Dict] = None):
        """Start a pipeline step"""
        step = self._find_step(step_name)
        if not step:
            # Create new step if not found
            step = ProgressStep(step_name, description or step_name, 1.0)
            self.steps.append(step)
        
        step.status = "running"
        step.start_time = datetime.now()
        if metadata:
            step.metadata.update(metadata)
        
        self._log_progress(f"🚀 Starting: {step.description}")
        self._update_progress()
    
    def complete_step(self, step_name: str, metadata: Optional[Dict] = None):
        """Complete a pipeline step"""
        step = self._find_step(step_name)
        if step:
            step.status = "completed"
            step.end_time = datetime.now()
            if metadata:
                step.metadata.update(metadata)
            
            duration = step.duration.total_seconds() if step.duration else 0
            self._log_progress(f"✅ Completed: {step.description} ({duration:.1f}s)")
            self._update_progress()
    
    def fail_step(self, step_name: str, error_message: str):
        """Mark a step as failed"""
        step = self._find_step(step_name)
        if step:
            step.status = "failed"
            step.end_time = datetime.now()
            step.metadata["error"] = error_message
            
            self._log_progress(f"❌ Failed: {step.description} - {error_message}")
            self._update_progress()
    
    def update_step_progress(self, step_name: str, message: str, metadata: Optional[Dict] = None):
        """Update progress within a running step"""
        step = self._find_step(step_name)
        if step and step.status == "running":
            if metadata:
                step.metadata.update(metadata)
            
            self._log_progress(f"⚡ {step.description}: {message}")
    
    def start_sub_step(self, parent_step: str, sub_step_name: str, description: str):
        """Start a sub-step within a main step"""
        parent = self._find_step(parent_step)
        if parent:
            sub_step = self._find_sub_step(parent, sub_step_name)
            if sub_step:
                sub_step.status = "running"
                sub_step.start_time = datetime.now()
                
                self._log_progress(f"  🔸 {description}...")
                self._update_progress()
    
    def complete_sub_step(self, parent_step: str, sub_step_name: str):
        """Complete a sub-step"""
        parent = self._find_step(parent_step)
        if parent:
            sub_step = self._find_sub_step(parent, sub_step_name)
            if sub_step:
                sub_step.status = "completed"
                sub_step.end_time = datetime.now()
                
                duration = sub_step.duration.total_seconds() if sub_step.duration else 0
                self._log_progress(f"  ✅ {sub_step.description} ({duration:.1f}s)")
                self._update_progress()
    
    def get_overall_progress(self) -> float:
        """Calculate overall pipeline progress percentage"""
        if not self.steps:
            return 0.0
        
        total_weight = sum(step.weight for step in self.steps)
        completed_weight = 0.0
        
        for step in self.steps:
            step_progress = step.progress_percentage / 100.0
            completed_weight += step.weight * step_progress
        
        return (completed_weight / total_weight) * 100.0 if total_weight > 0 else 0.0
    
    def get_eta(self) -> Optional[str]:
        """Estimate time remaining"""
        progress = self.get_overall_progress()
        if progress <= 0:
            return None
        
        elapsed = datetime.now() - self.start_time
        if progress >= 100:
            return "Complete"
        
        estimated_total = elapsed.total_seconds() / (progress / 100.0)
        remaining = estimated_total - elapsed.total_seconds()
        
        if remaining <= 0:
            return "Almost done"
        
        return self._format_duration(remaining)
    
    def complete_pipeline(self):
        """Mark the entire pipeline as complete"""
        self.end_time = datetime.now()
        total_duration = self.end_time - self.start_time
        
        self._log_progress(f"✨ Pipeline completed in {self._format_duration(total_duration.total_seconds())}")
        
        # Generate completion summary
        self._generate_completion_summary()
    
    def add_progress_callback(self, callback: Callable):
        """Add a callback function to be called on progress updates"""
        self.progress_callbacks.append(callback)
    
    def _find_step(self, step_name: str) -> Optional[ProgressStep]:
        """Find a step by name"""
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    def _find_sub_step(self, parent: ProgressStep, sub_step_name: str) -> Optional[ProgressStep]:
        """Find a sub-step by name"""
        for sub_step in parent.sub_steps:
            if sub_step.name == sub_step_name:
                return sub_step
        return None
    
    def _log_progress(self, message: str):
        """Log progress message to console and/or file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if self.console_output:
            print(f"[{timestamp}] {message}")
        
        if self.log_to_file and self.logger:
            self.logger.info(message)
    
    def _update_progress(self):
        """Update progress and call callbacks"""
        progress = self.get_overall_progress()
        eta = self.get_eta()
        
        # Progress bar for console
        if self.console_output:
            self._display_progress_bar(progress, eta)
        
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback({
                    'job_id': self.job_id,
                    'progress': progress,
                    'eta': eta,
                    'current_step': self._get_current_step_info(),
                    'steps': self._serialize_steps()
                })
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")
    
    def _display_progress_bar(self, progress: float, eta: Optional[str]):
        """Display progress bar in console"""
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        eta_str = f" | ETA: {eta}" if eta else ""
        print(f"\r📊 Progress: [{bar}] {progress:.1f}%{eta_str}", end='', flush=True)
        
        if progress >= 100:
            print()  # New line when complete
    
    def _get_current_step_info(self) -> Dict:
        """Get information about the current step"""
        running_steps = [step for step in self.steps if step.status == "running"]
        if running_steps:
            step = running_steps[0]
            return {
                'name': step.name,
                'description': step.description,
                'progress': step.progress_percentage,
                'metadata': step.metadata
            }
        return {}
    
    def _serialize_steps(self) -> List[Dict]:
        """Serialize steps for JSON storage"""
        return [
            {
                'name': step.name,
                'description': step.description,
                'status': step.status,
                'progress': step.progress_percentage,
                'start_time': step.start_time.isoformat() if step.start_time else None,
                'end_time': step.end_time.isoformat() if step.end_time else None,
                'duration': step.duration.total_seconds() if step.duration else None,
                'metadata': step.metadata,
                'sub_steps': [
                    {
                        'name': sub.name,
                        'description': sub.description,
                        'status': sub.status,
                        'progress': sub.progress_percentage
                    } for sub in step.sub_steps
                ]
            } for step in self.steps
        ]
    
    def _generate_completion_summary(self):
        """Generate a summary of the completed pipeline"""
        total_duration = self.end_time - self.start_time
        
        summary = {
            'job_id': self.job_id,
            'total_duration': self._format_duration(total_duration.total_seconds()),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'steps_summary': []
        }
        
        for step in self.steps:
            step_summary = {
                'name': step.name,
                'description': step.description,
                'status': step.status,
                'duration': self._format_duration(step.duration.total_seconds()) if step.duration else "N/A"
            }
            summary['steps_summary'].append(step_summary)
        
        # Log summary
        self._log_progress("\n" + "="*60)
        self._log_progress("📋 PIPELINE EXECUTION SUMMARY")
        self._log_progress("="*60)
        self._log_progress(f"Job ID: {self.job_id}")
        self._log_progress(f"Total Duration: {summary['total_duration']}")
        self._log_progress("\nStep Breakdown:")
        
        for step_summary in summary['steps_summary']:
            status_icon = "✅" if step_summary['status'] == "completed" else "❌" if step_summary['status'] == "failed" else "⏸️"
            self._log_progress(f"  {status_icon} {step_summary['description']}: {step_summary['duration']}")
        
        self._log_progress("="*60)
        
        return summary
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {int(secs)}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"
    
    def save_progress_state(self, file_path: str):
        """Save current progress state to file"""
        state = {
            'job_id': self.job_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'overall_progress': self.get_overall_progress(),
            'steps': self._serialize_steps()
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_progress_state(self, file_path: str):
        """Load progress state from file"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        self.job_id = state['job_id']
        self.start_time = datetime.fromisoformat(state['start_time'])
        if state['end_time']:
            self.end_time = datetime.fromisoformat(state['end_time'])
        
        # Reconstruct steps from saved state
        # This would need more complex logic for full restoration
        pass