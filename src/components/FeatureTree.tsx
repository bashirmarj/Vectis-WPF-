import React, { useState, useMemo } from "react";
import "./FeatureTree.css";

interface FeatureInstance {
  instance_id: number;
  feature_type: string;
  face_ids: number[];
  confidence: number;
  geometric_primitive?: string;
  parameters?: Record<string, number | string>;
}

interface MLFeatures {
  feature_instances?: FeatureInstance[];
  feature_summary?: Record<string, number>;
  num_features_detected?: number;
  num_faces_analyzed?: number;
  face_predictions?: Array<{
    face_id: number;
    predicted_class: string;
    confidence: number;
  }>;
  inference_time_sec?: number;
}

interface FeatureTreeProps {
  mlFeatures?: MLFeatures;
  isLoading?: boolean;
}

const FeatureTree: React.FC<FeatureTreeProps> = ({ mlFeatures, isLoading = false }) => {
  const [expandedInstances, setExpandedInstances] = useState<Set<number>>(new Set());
  const [showRawPredictions, setShowRawPredictions] = useState(false);

  const toggleInstanceExpanded = (instanceId: number) => {
    const newSet = new Set(expandedInstances);
    if (newSet.has(instanceId)) {
      newSet.delete(instanceId);
    } else {
      newSet.add(instanceId);
    }
    setExpandedInstances(newSet);
  };

  const featureInstances = useMemo(() => {
    return mlFeatures?.feature_instances || [];
  }, [mlFeatures]);

  const featureSummary = useMemo(() => {
    return mlFeatures?.feature_summary || {};
  }, [mlFeatures]);

  const facePredictions = useMemo(() => {
    return mlFeatures?.face_predictions || [];
  }, [mlFeatures]);

  const numFeaturesDetected = useMemo(() => {
    return mlFeatures?.num_features_detected || 0;
  }, [mlFeatures]);

  const numFacesAnalyzed = useMemo(() => {
    return mlFeatures?.num_faces_analyzed || 0;
  }, [mlFeatures]);

  const inferenceTime = useMemo(() => {
    return mlFeatures?.inference_time_sec || 0;
  }, [mlFeatures]);

  if (isLoading) {
    return (
      <div className="feature-tree loading">
        <div className="spinner"></div>
        <p>Running ML feature recognition...</p>
      </div>
    );
  }

  if (!mlFeatures) {
    return (
      <div className="feature-tree empty">
        <p>Upload a STEP file to see feature analysis</p>
      </div>
    );
  }

  if (mlFeatures.error) {
    return (
      <div className="feature-tree error">
        <h3>❌ ML Inference Error</h3>
        <p>{mlFeatures.error}</p>
      </div>
    );
  }

  return (
    <div className="feature-tree">
      {/* Header */}
      <div className="feature-tree-header">
        <h2>🤖 ML Feature Recognition</h2>
        <p className="subtitle">
          {numFeaturesDetected} feature instances detected from {numFacesAnalyzed} faces in {inferenceTime}s
        </p>
      </div>

      {/* Feature Distribution */}
      <div className="feature-distribution">
        <h3>📊 Feature Distribution</h3>
        <div className="distribution-grid">
          {Object.entries(featureSummary).map(([featureType, count]) => {
            if (featureType === "total_features") return null;

            return (
              <div key={featureType} className="distribution-item">
                <span className="feature-type">{featureType}</span>
                <span className="count">{count}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Feature Instances */}
      <div className="feature-instances">
        <h3>📋 Feature Instances ({featureInstances.length})</h3>

        {featureInstances.length === 0 ? (
          <div className="no-features">
            <p>No feature instances detected</p>
          </div>
        ) : (
          <div className="instances-list">
            {featureInstances.map((instance) => (
              <div key={instance.instance_id} className="instance-card">
                <div className="instance-header" onClick={() => toggleInstanceExpanded(instance.instance_id)}>
                  <span className="toggle-icon">{expandedInstances.has(instance.instance_id) ? "▼" : "▶"}</span>

                  <div className="instance-title">
                    <span className="instance-id">[ID:{instance.instance_id}]</span>
                    <span className="feature-type-badge">{instance.feature_type}</span>
                    <span className="confidence">{(instance.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>

                {expandedInstances.has(instance.instance_id) && (
                  <div className="instance-details">
                    {/* Faces */}
                    <div className="detail-section">
                      <h4>Faces ({instance.face_ids.length})</h4>
                      <div className="face-ids">
                        {instance.face_ids.map((faceId) => (
                          <span key={faceId} className="face-id">
                            {faceId}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Geometric Primitive */}
                    {instance.geometric_primitive && (
                      <div className="detail-section">
                        <h4>Geometric Primitive</h4>
                        <p className="primitive">{instance.geometric_primitive}</p>
                      </div>
                    )}

                    {/* Parameters */}
                    {instance.parameters && Object.keys(instance.parameters).length > 0 && (
                      <div className="detail-section">
                        <h4>Parameters</h4>
                        <div className="parameters">
                          {Object.entries(instance.parameters).map(([key, value]) => (
                            <div key={key} className="parameter">
                              <span className="param-name">{key}:</span>
                              <span className="param-value">
                                {typeof value === "number" ? value.toFixed(2) : value}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Confidence */}
                    <div className="detail-section">
                      <h4>Confidence Score</h4>
                      <div className="confidence-bar">
                        <div className="confidence-fill" style={{ width: `${instance.confidence * 100}%` }}></div>
                      </div>
                      <p className="confidence-value">{(instance.confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Statistics */}
      <div className="feature-statistics">
        <h3>📈 Analysis Statistics</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Features Detected</span>
            <span className="stat-value">{numFeaturesDetected}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Faces Analyzed</span>
            <span className="stat-value">{numFacesAnalyzed}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Avg Confidence</span>
            <span className="stat-value">
              {featureInstances.length > 0
                ? (
                    (featureInstances.reduce((sum, inst) => sum + inst.confidence, 0) / featureInstances.length) *
                    100
                  ).toFixed(0)
                : "N/A"}
              %
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Inference Time</span>
            <span className="stat-value">{inferenceTime.toFixed(2)}s</span>
          </div>
        </div>
      </div>

      {/* Toggle Raw Predictions */}
      <div className="feature-debug">
        <button className="toggle-raw-button" onClick={() => setShowRawPredictions(!showRawPredictions)}>
          {showRawPredictions ? "▼" : "▶"} Raw Face Predictions ({facePredictions.length})
        </button>

        {showRawPredictions && (
          <div className="raw-predictions">
            <div className="predictions-list">
              {facePredictions.map((pred, idx) => (
                <div key={idx} className="prediction-item">
                  <span className="face-id">Face {pred.face_id}</span>
                  <span className="class">{pred.predicted_class}</span>
                  <span className="confidence">{(pred.confidence * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FeatureTree;
