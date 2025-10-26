"""
MLflow Integration for Deepfake Detection
Tracks experiments, model performance, and predictions
"""

import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any, Optional
import json
import os


class MLflowService:
    """MLflow experiment tracking service"""

    def __init__(self, experiment_name: str = "deepfake_detection", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracking

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local file storage)
        """
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local directory
            mlruns_dir = Path(__file__).parent.parent.parent / "mlruns"
            mlruns_dir.mkdir(exist_ok=True)

            # Convert Windows path to proper file URI
            # Windows: C:/path -> file:///C:/path
            # Linux: /path -> file:///path
            posix_path = mlruns_dir.absolute().as_posix()
            if posix_path[1] == ':':  # Windows drive letter
                tracking_uri = f"file:///{posix_path}"
            else:  # Linux/Mac
                tracking_uri = f"file://{posix_path}"

            mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.run_id = None

        print(f"[OK] MLflow initialized - Experiment: {experiment_name}")
        print(f"   Tracking URI: {mlflow.get_tracking_uri()}")

    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run"""
        self.active_run = mlflow.start_run(run_name=run_name)
        self.run_id = self.active_run.info.run_id
        return self.active_run

    def end_run(self):
        """End the current MLflow run"""
        if mlflow.active_run():
            mlflow.end_run()
            self.run_id = None

    def log_prediction(self,
                       prediction_type: str,  # 'image', 'video', 'webcam'
                       result: Dict[str, Any],
                       metadata: Optional[Dict] = None):
        """
        Log a prediction to MLflow

        Args:
            prediction_type: Type of prediction (image/video/webcam)
            result: Detection result dictionary
            metadata: Additional metadata (filename, resolution, etc.)
        """
        with mlflow.start_run(nested=True):
            # Log prediction type
            mlflow.log_param("prediction_type", prediction_type)

            # Log main metrics
            if 'confidence' in result:
                mlflow.log_metric("confidence", result['confidence'])
            if 'fake_probability' in result:
                mlflow.log_metric("fake_probability", result['fake_probability'])
            if 'real_probability' in result:
                mlflow.log_metric("real_probability", result['real_probability'])

            # Log prediction
            if 'prediction' in result:
                mlflow.log_param("prediction", result['prediction'])

            # Log processing time
            if 'processing_time' in result:
                mlflow.log_metric("processing_time_sec", result['processing_time'])

            # Log model ensemble results
            if 'model_predictions' in result:
                for model_name, model_result in result['model_predictions'].items():
                    mlflow.log_metric(f"{model_name}_fake_prob", model_result.get('fake_prob', 0))
                    mlflow.log_metric(f"{model_name}_confidence",
                                    max(model_result.get('fake_prob', 0),
                                        model_result.get('real_prob', 0)))

            # Log models used
            if 'models_used' in result:
                mlflow.log_param("models_used", ",".join(result['models_used']))
                mlflow.log_metric("num_models", result.get('total_models', 0))

            # Log metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"meta_{key}", value)
                    else:
                        mlflow.log_param(f"meta_{key}", str(value))

            # For video predictions
            if prediction_type == 'video' and 'overall_result' in result:
                overall = result['overall_result']
                mlflow.log_metric("video_fake_frame_ratio", overall.get('fake_frame_ratio', 0))
                mlflow.log_metric("video_confidence", overall.get('confidence', 0))
                mlflow.log_param("video_prediction", overall.get('prediction', 'UNKNOWN'))

                if 'processing_info' in result:
                    proc_info = result['processing_info']
                    mlflow.log_metric("frames_processed", proc_info.get('frames_processed', 0))
                    mlflow.log_metric("processing_fps", proc_info.get('processing_fps', 0))

    def log_batch_results(self, results: list, batch_type: str = "image"):
        """
        Log batch processing results

        Args:
            results: List of detection results
            batch_type: Type of batch (image/video)
        """
        with mlflow.start_run(run_name=f"batch_{batch_type}_{len(results)}"):
            # Aggregate statistics
            total = len(results)
            fake_count = sum(1 for r in results if r.get('prediction') == 'FAKE')
            real_count = total - fake_count

            avg_confidence = sum(r.get('confidence', 0) for r in results) / total if total > 0 else 0
            avg_processing_time = sum(r.get('processing_time', 0) for r in results) / total if total > 0 else 0

            # Log batch metrics
            mlflow.log_param("batch_type", batch_type)
            mlflow.log_metric("batch_total", total)
            mlflow.log_metric("batch_fake_count", fake_count)
            mlflow.log_metric("batch_real_count", real_count)
            mlflow.log_metric("batch_fake_ratio", fake_count / total if total > 0 else 0)
            mlflow.log_metric("batch_avg_confidence", avg_confidence)
            mlflow.log_metric("batch_avg_processing_time", avg_processing_time)

            # Log individual predictions as artifacts
            results_json = json.dumps(results, indent=2)
            mlflow.log_text(results_json, "batch_results.json")

    def log_model_info(self, model_config: Dict):
        """Log model configuration and architecture"""
        with mlflow.start_run(run_name="model_info"):
            # Log model config
            for model_name, config in model_config.get('models', {}).items():
                if config.get('enabled'):
                    mlflow.log_param(f"{model_name}_enabled", True)
                    mlflow.log_param(f"{model_name}_weight", config.get('weight', 0))

            # Log ensemble config
            ensemble = model_config.get('ensemble', {})
            mlflow.log_param("ensemble_method", ensemble.get('method', 'unknown'))
            mlflow.log_param("ensemble_threshold", ensemble.get('threshold', 0.5))

    def get_experiment_stats(self) -> Dict:
        """Get statistics from the current experiment"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if not experiment:
            return {}

        # Get all runs in the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            return {
                'total_runs': 0,
                'total_predictions': 0
            }

        return {
            'experiment_name': self.experiment_name,
            'experiment_id': experiment.experiment_id,
            'total_runs': len(runs),
            'avg_confidence': runs['metrics.confidence'].mean() if 'metrics.confidence' in runs.columns else 0,
            'total_fake': runs[runs.get('params.prediction') == 'FAKE'].shape[0],
            'total_real': runs[runs.get('params.prediction') == 'REAL'].shape[0]
        }
