# Updated Frontend Component - Feature Tree
# Displays feature instances instead of raw face counts
# Replaces FeatureTree.tsx

import React, { useState } from 'react';
import { ChevronDown, AlertCircle } from 'lucide-react';

interface FeatureParameter {
  [key: string]: number | null;
}

interface FeatureInstance {
  instance_id: number;
  feature_type: string;
  face_ids: number[];
  confidence: number;
  geometric_primitive: string | null;
  parameters: FeatureParameter;
  bounding_box: object | null;
}

interface MLFeatures {
  feature_instances: FeatureInstance[];
  feature_summary: {
    [key: string]: number;
  };
  num_features_detected: number;
  num_faces_analyzed: number;
  face_predictions: Array<{
    face_id: number;
    predicted_class: string;
    confidence: number;
  }>;
  clustering_method?: string;
  inference_time_sec?: number;
}

interface Props {
  mlFeatures: MLFeatures;
}

export const FeatureTree: React.FC<Props> = ({ mlFeatures }) => {
  const [expandedInstances, setExpandedInstances] = useState<Set<number>>(new Set());
  const [showDebugInfo, setShowDebugInfo] = useState(false);

  if (!mlFeatures || !mlFeatures.feature_instances) {
    return (
      <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
        <div className="flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-amber-600" />
          <span className="text-sm text-amber-700">No feature data available</span>
        </div>
      </div>
    );
  }

  const toggleExpanded = (instanceId: number) => {
    const newExpanded = new Set(expandedInstances);
    if (newExpanded.has(instanceId)) {
      newExpanded.delete(instanceId);
    } else {
      newExpanded.add(instanceId);
    }
    setExpandedInstances(newExpanded);
  };

  const getFeatureColor = (featureType: string): string => {
    const colors: { [key: string]: string } = {
      hole: 'bg-blue-100 text-blue-900',
      boss: 'bg-green-100 text-green-900',
      pocket: 'bg-purple-100 text-purple-900',
      slot: 'bg-orange-100 text-orange-900',
      chamfer: 'bg-red-100 text-red-900',
      fillet: 'bg-cyan-100 text-cyan-900',
      groove: 'bg-indigo-100 text-indigo-900',
      step: 'bg-yellow-100 text-yellow-900',
      default: 'bg-gray-100 text-gray-900',
    };
    return colors[featureType] || colors['default'];
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.75) return 'text-blue-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-4">
      {/* Header */}
      <div className="border-b pb-3">
        <h3 className="text-lg font-semibold text-gray-900">
          🤖 ML Feature Recognition (NEW)
        </h3>
        <p className="text-sm text-gray-600 mt-1">
          {mlFeatures.num_features_detected} feature instances detected from{' '}
          {mlFeatures.num_faces_analyzed} faces
          {mlFeatures.inference_time_sec && (
            <span className="ml-2 text-gray-500">
              ({mlFeatures.inference_time_sec}s)
            </span>
          )}
        </p>
      </div>

      {/* Feature Summary Stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-blue-50 p-3 rounded-lg">
          <div className="text-2xl font-bold text-blue-900">
            {mlFeatures.num_features_detected}
          </div>
          <div className="text-xs text-blue-700">Feature Instances</div>
        </div>
        <div className="bg-purple-50 p-3 rounded-lg">
          <div className="text-2xl font-bold text-purple-900">
            {mlFeatures.num_faces_analyzed}
          </div>
          <div className="text-xs text-purple-700">Faces Analyzed</div>
        </div>
        <div className="bg-green-50 p-3 rounded-lg">
          <div className="text-2xl font-bold text-green-900">
            {(
              mlFeatures.feature_instances.reduce(
                (sum, inst) => sum + inst.confidence,
                0
              ) / mlFeatures.feature_instances.length || 0
            ).toFixed(2)}
          </div>
          <div className="text-xs text-green-700">Avg Confidence</div>
        </div>
      </div>

      {/* Feature Distribution */}
      {Object.keys(mlFeatures.feature_summary).length > 1 && (
        <div className="bg-gray-50 p-3 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">
            Feature Distribution
          </h4>
          <div className="grid grid-cols-4 gap-2">
            {Object.entries(mlFeatures.feature_summary).map(([featureType, count]) => {
              if (featureType === 'total_features' || count === 0) return null;
              return (
                <div
                  key={featureType}
                  className={`px-2 py-1 rounded text-xs font-medium ${getFeatureColor(
                    featureType
                  )}`}
                >
                  {featureType}: {count}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Feature Instances List */}
      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-gray-700">Feature Instances</h4>
        {mlFeatures.feature_instances.length === 0 ? (
          <div className="p-3 bg-gray-50 rounded text-sm text-gray-600">
            No features detected. Part may be a simple geometric shape.
          </div>
        ) : (
          mlFeatures.feature_instances.map((instance) => (
            <div
              key={instance.instance_id}
              className="border border-gray-200 rounded-lg overflow-hidden"
            >
              {/* Instance Header */}
              <button
                onClick={() => toggleExpanded(instance.instance_id)}
                className="w-full px-3 py-2 hover:bg-gray-50 flex items-center justify-between text-left"
              >
                <div className="flex items-center gap-2 flex-1">
                  <ChevronDown
                    className={`w-4 h-4 transition-transform ${
                      expandedInstances.has(instance.instance_id)
                        ? 'rotate-0'
                        : '-rotate-90'
                    }`}
                  />
                  <span
                    className={`px-2 py-1 rounded text-xs font-semibold ${getFeatureColor(
                      instance.feature_type
                    )}`}
                  >
                    {instance.feature_type}
                  </span>
                  <span className="text-sm text-gray-600">
                    ID: {instance.instance_id}
                  </span>
                </div>
                <span
                  className={`text-sm font-medium ${getConfidenceColor(
                    instance.confidence
                  )}`}
                >
                  {(instance.confidence * 100).toFixed(1)}%
                </span>
              </button>

              {/* Expanded Details */}
              {expandedInstances.has(instance.instance_id) && (
                <div className="border-t bg-gray-50 px-3 py-2 space-y-2">
                  {/* Geometric Primitive */}
                  {instance.geometric_primitive && (
                    <div className="flex justify-between">
                      <span className="text-xs text-gray-600">
                        Geometric Primitive:
                      </span>
                      <span className="text-xs font-medium text-gray-900">
                        {instance.geometric_primitive}
                      </span>
                    </div>
                  )}

                  {/* Face IDs */}
                  <div className="flex justify-between">
                    <span className="text-xs text-gray-600">Composed of Faces:</span>
                    <span className="text-xs font-mono text-gray-900">
                      {instance.face_ids.join(', ')}
                    </span>
                  </div>

                  {/* Parameters */}
                  {Object.keys(instance.parameters).length > 0 && (
                    <div className="border-t pt-2 mt-2">
                      <p className="text-xs font-semibold text-gray-700 mb-1">
                        Parameters:
                      </p>
                      <div className="grid grid-cols-2 gap-1">
                        {Object.entries(instance.parameters).map(([key, value]) => (
                          <div key={key} className="text-xs">
                            <span className="text-gray-600">{key}:</span>
                            <span className="ml-1 font-medium text-gray-900">
                              {value !== null ? value : 'N/A'}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Debug Information */}
      <button
        onClick={() => setShowDebugInfo(!showDebugInfo)}
        className="text-xs text-gray-500 hover:text-gray-700 mt-4"
      >
        {showDebugInfo ? '▼' : '▶'} Debug Information
      </button>

      {showDebugInfo && (
        <div className="bg-gray-50 border border-gray-200 rounded p-3 space-y-2">
          <div className="text-xs">
            <p className="text-gray-600">
              <strong>Clustering Method:</strong>{' '}
              {mlFeatures.clustering_method || 'N/A'}
            </p>
            <p className="text-gray-600">
              <strong>Total Face Predictions:</strong>{' '}
              {mlFeatures.face_predictions.length}
            </p>
          </div>

          {/* Raw Face Predictions (collapsed) */}
          <details className="text-xs">
            <summary className="cursor-pointer text-gray-700 hover:text-gray-900">
              Show raw face predictions
            </summary>
            <div className="mt-2 max-h-48 overflow-y-auto bg-white border border-gray-200 rounded p-2">
              {mlFeatures.face_predictions.slice(0, 20).map((pred) => (
                <div key={pred.face_id} className="py-1 text-gray-600">
                  Face {pred.face_id}: <strong>{pred.predicted_class}</strong>{' '}
                  ({(pred.confidence * 100).toFixed(1)}%)
                </div>
              ))}
              {mlFeatures.face_predictions.length > 20 && (
                <div className="text-gray-500 italic">
                  ... and {mlFeatures.face_predictions.length - 20} more
                </div>
              )}
            </div>
          </details>
        </div>
      )}
    </div>
  );
};

export default FeatureTree;