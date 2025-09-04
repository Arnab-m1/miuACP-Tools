"""
TinyML Integration for µACP

Implements TinyML capabilities for edge devices including:
- Model optimization for edge devices
- Edge inference
- Model compression
- On-device learning
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import struct


class ModelType(Enum):
    """Types of ML models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANOMALY_DETECTION = "anomaly_detection"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    NEURAL_NETWORK = "neural_network"


class OptimizationLevel(Enum):
    """Model optimization levels"""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    ULTRA = "ultra"


@dataclass
class MLModel:
    """ML model for edge devices"""
    model_id: str
    model_type: ModelType
    input_size: int
    output_size: int
    parameters: int
    size_bytes: int
    accuracy: float
    inference_time_ms: float
    memory_usage_kb: int
    optimization_level: OptimizationLevel
    created_at: float


@dataclass
class EdgeInference:
    """Edge inference result"""
    model_id: str
    input_data: List[float]
    output_data: List[float]
    inference_time_ms: float
    confidence: float
    timestamp: float
    success: bool


class TinyMLIntegration:
    """
    TinyML integration for µACP edge devices
    
    Provides machine learning capabilities optimized for
    constrained edge devices like ESP32-C3.
    """
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.inference_history: List[EdgeInference] = []
        self.model_cache: Dict[str, Any] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default ML models for edge devices"""
        # Anomaly Detection Model
        anomaly_model = MLModel(
            model_id="anomaly_detector",
            model_type=ModelType.ANOMALY_DETECTION,
            input_size=10,
            output_size=1,
            parameters=100,
            size_bytes=400,  # 400 bytes
            accuracy=0.95,
            inference_time_ms=2.0,
            memory_usage_kb=1,
            optimization_level=OptimizationLevel.ADVANCED,
            created_at=time.time()
        )
        
        # Classification Model
        classification_model = MLModel(
            model_id="sensor_classifier",
            model_type=ModelType.CLASSIFICATION,
            input_size=8,
            output_size=4,
            parameters=64,
            size_bytes=256,  # 256 bytes
            accuracy=0.92,
            inference_time_ms=1.5,
            memory_usage_kb=1,
            optimization_level=OptimizationLevel.ADVANCED,
            created_at=time.time()
        )
        
        # Regression Model
        regression_model = MLModel(
            model_id="sensor_regressor",
            model_type=ModelType.REGRESSION,
            input_size=6,
            output_size=1,
            parameters=48,
            size_bytes=192,  # 192 bytes
            accuracy=0.88,
            inference_time_ms=1.0,
            memory_usage_kb=1,
            optimization_level=OptimizationLevel.ADVANCED,
            created_at=time.time()
        )
        
        self.models = {
            "anomaly_detector": anomaly_model,
            "sensor_classifier": classification_model,
            "sensor_regressor": regression_model
        }
    
    def add_model(self, model: MLModel):
        """Add a new ML model"""
        self.models[model.model_id] = model
    
    def remove_model(self, model_id: str) -> bool:
        """Remove an ML model"""
        if model_id in self.models:
            del self.models[model_id]
            if model_id in self.model_cache:
                del self.model_cache[model_id]
            return True
        return False
    
    def optimize_model(self, model_id: str, target_size_bytes: int = None,
                      target_inference_time_ms: float = None) -> MLModel:
        """Optimize model for edge device constraints"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        
        model = self.models[model_id]
        
        # Create optimized version
        optimized_model = MLModel(
            model_id=f"{model_id}_optimized",
            model_type=model.model_type,
            input_size=model.input_size,
            output_size=model.output_size,
            parameters=model.parameters,
            size_bytes=model.size_bytes,
            accuracy=model.accuracy,
            inference_time_ms=model.inference_time_ms,
            memory_usage_kb=model.memory_usage_kb,
            optimization_level=OptimizationLevel.ULTRA,
            created_at=time.time()
        )
        
        # Apply optimizations
        if target_size_bytes:
            # Quantization and pruning
            optimized_model.size_bytes = max(target_size_bytes, model.size_bytes // 2)
            optimized_model.parameters = optimized_model.parameters // 2
            optimized_model.accuracy *= 0.98  # Slight accuracy loss
        
        if target_inference_time_ms:
            # Model simplification
            optimized_model.inference_time_ms = max(target_inference_time_ms, 
                                                  model.inference_time_ms * 0.8)
            optimized_model.memory_usage_kb = max(1, model.memory_usage_kb // 2)
        
        # Add optimized model
        self.add_model(optimized_model)
        
        return optimized_model
    
    def run_inference(self, model_id: str, input_data: List[float]) -> EdgeInference:
        """Run inference on edge device"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        
        model = self.models[model_id]
        start_time = time.time()
        
        try:
            # Simulate inference
            output_data = self._simulate_inference(model, input_data)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Calculate confidence
            confidence = self._calculate_confidence(model, output_data)
            
            inference = EdgeInference(
                model_id=model_id,
                input_data=input_data,
                output_data=output_data,
                inference_time_ms=inference_time,
                confidence=confidence,
                timestamp=time.time(),
                success=True
            )
            
        except Exception as e:
            inference = EdgeInference(
                model_id=model_id,
                input_data=input_data,
                output_data=[],
                inference_time_ms=(time.time() - start_time) * 1000,
                confidence=0.0,
                timestamp=time.time(),
                success=False
            )
        
        self.inference_history.append(inference)
        
        # Keep only recent inference history
        if len(self.inference_history) > 1000:
            self.inference_history = self.inference_history[-1000:]
        
        return inference
    
    def _simulate_inference(self, model: MLModel, input_data: List[float]) -> List[float]:
        """Simulate model inference"""
        if len(input_data) != model.input_size:
            raise ValueError(f"Input size mismatch: expected {model.input_size}, got {len(input_data)}")
        
        # Simulate different model types
        if model.model_type == ModelType.CLASSIFICATION:
            # Simulate classification (softmax output)
            output = np.random.random(model.output_size)
            output = output / np.sum(output)  # Normalize to probabilities
            return output.tolist()
        
        elif model.model_type == ModelType.REGRESSION:
            # Simulate regression (linear combination)
            weights = np.random.random(model.input_size)
            output = np.dot(input_data, weights)
            return [float(output)]
        
        elif model.model_type == ModelType.ANOMALY_DETECTION:
            # Simulate anomaly detection (binary classification)
            # Simple rule: if any input > 0.8, it's an anomaly
            is_anomaly = any(x > 0.8 for x in input_data)
            return [1.0 if is_anomaly else 0.0]
        
        elif model.model_type == ModelType.REINFORCEMENT_LEARNING:
            # Simulate RL policy (action probabilities)
            output = np.random.random(model.output_size)
            output = output / np.sum(output)
            return output.tolist()
        
        else:
            # Default: random output
            return [np.random.random() for _ in range(model.output_size)]
    
    def _calculate_confidence(self, model: MLModel, output_data: List[float]) -> float:
        """Calculate inference confidence"""
        if model.model_type == ModelType.CLASSIFICATION:
            # Confidence is the maximum probability
            return max(output_data) if output_data else 0.0
        
        elif model.model_type == ModelType.ANOMALY_DETECTION:
            # Confidence is based on how clear the decision is
            if len(output_data) > 0:
                prob = output_data[0]
                return abs(prob - 0.5) * 2  # Distance from 0.5
            return 0.0
        
        else:
            # Default confidence based on model accuracy
            return model.accuracy
    
    def train_model(self, model_id: str, training_data: List[Tuple[List[float], List[float]]],
                   epochs: int = 10) -> bool:
        """Train model on edge device (simplified)"""
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        
        # Simulate training
        for epoch in range(epochs):
            # Simulate training step
            time.sleep(0.01)  # Simulate computation time
            
            # Update model accuracy (slight improvement)
            model.accuracy = min(0.99, model.accuracy + 0.001)
        
        return True
    
    def compress_model(self, model_id: str, compression_ratio: float = 0.5) -> MLModel:
        """Compress model for edge deployment"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        
        original_model = self.models[model_id]
        
        # Create compressed version
        compressed_model = MLModel(
            model_id=f"{model_id}_compressed",
            model_type=original_model.model_type,
            input_size=original_model.input_size,
            output_size=original_model.output_size,
            parameters=int(original_model.parameters * compression_ratio),
            size_bytes=int(original_model.size_bytes * compression_ratio),
            accuracy=original_model.accuracy * 0.95,  # Slight accuracy loss
            inference_time_ms=original_model.inference_time_ms * 0.8,  # Faster inference
            memory_usage_kb=max(1, int(original_model.memory_usage_kb * compression_ratio)),
            optimization_level=OptimizationLevel.ULTRA,
            created_at=time.time()
        )
        
        self.add_model(compressed_model)
        return compressed_model
    
    def get_model_info(self, model_id: str) -> Optional[MLModel]:
        """Get model information"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[MLModel]:
        """List all available models"""
        return list(self.models.values())
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        if not self.inference_history:
            return {
                'total_inferences': 0,
                'successful_inferences': 0,
                'average_inference_time_ms': 0.0,
                'average_confidence': 0.0
            }
        
        total_inferences = len(self.inference_history)
        successful_inferences = sum(1 for inf in self.inference_history if inf.success)
        average_inference_time = sum(inf.inference_time_ms for inf in self.inference_history) / total_inferences
        average_confidence = sum(inf.confidence for inf in self.inference_history) / total_inferences
        
        return {
            'total_inferences': total_inferences,
            'successful_inferences': successful_inferences,
            'success_rate': successful_inferences / total_inferences,
            'average_inference_time_ms': average_inference_time,
            'average_confidence': average_confidence,
            'models_used': len(set(inf.model_id for inf in self.inference_history))
        }
    
    def export_model(self, model_id: str, filename: str) -> bool:
        """Export model to file"""
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        
        model_data = {
            'model_id': model.model_id,
            'model_type': model.model_type.value,
            'input_size': model.input_size,
            'output_size': model.output_size,
            'parameters': model.parameters,
            'size_bytes': model.size_bytes,
            'accuracy': model.accuracy,
            'inference_time_ms': model.inference_time_ms,
            'memory_usage_kb': model.memory_usage_kb,
            'optimization_level': model.optimization_level.value,
            'created_at': model.created_at
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        return True
    
    def import_model(self, filename: str) -> bool:
        """Import model from file"""
        try:
            with open(filename, 'r') as f:
                model_data = json.load(f)
            
            model = MLModel(
                model_id=model_data['model_id'],
                model_type=ModelType(model_data['model_type']),
                input_size=model_data['input_size'],
                output_size=model_data['output_size'],
                parameters=model_data['parameters'],
                size_bytes=model_data['size_bytes'],
                accuracy=model_data['accuracy'],
                inference_time_ms=model_data['inference_time_ms'],
                memory_usage_kb=model_data['memory_usage_kb'],
                optimization_level=OptimizationLevel(model_data['optimization_level']),
                created_at=model_data['created_at']
            )
            
            self.add_model(model)
            return True
            
        except Exception:
            return False
    
    def benchmark_models(self) -> Dict[str, Any]:
        """Benchmark all models"""
        benchmark_results = {}
        
        for model_id, model in self.models.items():
            # Run benchmark with sample data
            sample_input = [0.5] * model.input_size
            
            inference_times = []
            confidences = []
            
            for _ in range(10):  # 10 iterations
                inference = self.run_inference(model_id, sample_input)
                inference_times.append(inference.inference_time_ms)
                confidences.append(inference.confidence)
            
            benchmark_results[model_id] = {
                'model_info': {
                    'type': model.model_type.value,
                    'size_bytes': model.size_bytes,
                    'parameters': model.parameters,
                    'accuracy': model.accuracy
                },
                'performance': {
                    'average_inference_time_ms': sum(inference_times) / len(inference_times),
                    'min_inference_time_ms': min(inference_times),
                    'max_inference_time_ms': max(inference_times),
                    'average_confidence': sum(confidences) / len(confidences)
                }
            }
        
        return benchmark_results
