import { useMemo, useState } from "react";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "/detect";

const PIPE_STREAMS = {
  hollow: {
    label: "Hollow Pipe",
    model: "best_hollow.onnx",
    breaker: "hollow_specific/type_breaker.py",
    defaultConfidence: 0.25,
  },
  solid: {
    label: "Solid Pipe",
    model: "best_solid.onnx",
    breaker: "solid_specific/type_breaker.py",
    defaultConfidence: 0.6,
  },
};

const SAMPLE_IMAGES = [
  { id: "hollow_7.jpg", label: "Hollow Pipe 7", type: "hollow", desc: "Hollow steel pipe stack" },
  { id: "hollow_8.jpg", label: "Hollow Pipe 8", type: "hollow", desc: "Hollow pipe bundle" },
  { id: "hollow_9.jpg", label: "Hollow Pipe 9", type: "hollow", desc: "Hollow pipe bundle (wide)" },
  { id: "solid_1.jpg", label: "Solid Rod 1", type: "solid", desc: "Solid round steel bar stack" },
  { id: "solid_2.jpg", label: "Solid Rod 2", type: "solid", desc: "Solid steel bar stack" },
  { id: "solid_3.jpg", label: "Solid Rod 3", type: "solid", desc: "Solid steel rod bundle" },
  { id: "solid_5.jpg", label: "Solid Rod 5", type: "solid", desc: "Solid bar end faces" },
  { id: "solid_6.jpg", label: "Solid Rod 6", type: "solid", desc: "Solid steel bar bundle" },
  { id: "solid_12.jpg", label: "Solid Rod 12", type: "solid", desc: "Dense solid rod bundle" },
  { id: "solid_14.jpg", label: "Solid Rod 14", type: "solid", desc: "Solid round bar pallet" },
];

function App() {
  const [pipeType, setPipeType] = useState("hollow");
  const [sourceMode, setSourceMode] = useState("sample"); // "sample" | "upload"
  const [sampleFilter, setSampleFilter] = useState("all"); // "all" | "hollow" | "solid"
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [sampleLoading, setSampleLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [confidenceByType, setConfidenceByType] = useState({ hollow: 0.25, solid: 0.6 });
  const [iou, setIou] = useState(0.45);
  const [imageSize, setImageSize] = useState(640);
  const [glareFilter, setGlareFilter] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const activeStream = PIPE_STREAMS[pipeType];
  const confidence = confidenceByType[pipeType];

  const imageMode = useMemo(() => {
    if (!file) return "Waiting";
    return glareFilter ? "Normalized" : "Raw image";
  }, [file, glareFilter]);

  const selectPipeType = (nextType) => {
    setPipeType(nextType);
    setResult(null);
    setError("");
  };

  const setActiveConfidence = (value) => {
    setConfidenceByType((current) => ({
      ...current,
      [pipeType]: value,
    }));
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0] || null;
    setSelectedSampleId("");
    setFile(selectedFile);
    setResult(null);
    setError("");

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setPreviewUrl(selectedFile ? URL.createObjectURL(selectedFile) : "");
  };

  const loadSampleImage = async (sample) => {
    if (!sample) return;
    setSampleLoading(true);
    setError("");
    setResult(null);

    // Automatically align pipe stream with sample type
    if (sample.type !== pipeType) {
      setPipeType(sample.type);
    }

    try {
      const response = await fetch(`/samples/${sample.id}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch sample image (status ${response.status})`);
      }
      const blob = await response.blob();
      const sampleFile = new File([blob], sample.id, { type: "image/jpeg" });

      if (previewUrl && previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl);
      }

      setFile(sampleFile);
      setSelectedSampleId(sample.id);
      setPreviewUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(`Could not load ${sample.id}: ${err.message}`);
    } finally {
      setSampleLoading(false);
    }
  };

  const filteredSamples = useMemo(() => {
    if (sampleFilter === "all") return SAMPLE_IMAGES;
    return SAMPLE_IMAGES.filter((s) => s.type === sampleFilter);
  }, [sampleFilter]);

  const handleDetect = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);
    formData.append("pipeType", pipeType);
    formData.append("confidence", String(confidence));
    formData.append("iou", String(iou));
    formData.append("imageSize", String(imageSize));
    formData.append("glareFilter", glareFilter ? "true" : "false");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok || !payload.success) {
        throw new Error(payload.error || "Inference processing failed.");
      }

      setResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-title">Pipe Stream</div>
        <p className="sidebar-note">Only one stream is active for each run.</p>

        <button
          className={`pipe-card ${pipeType === "hollow" ? "active" : ""}`}
          type="button"
          onClick={() => selectPipeType("hollow")}
        >
          <span className="pipe-check">{pipeType === "hollow" ? "✓" : ""}</span>
          <span>
            <strong>Hollow Pipe</strong>
            <small>Hollow ONNX model and breaker</small>
          </span>
        </button>

        <button
          className={`pipe-card ${pipeType === "solid" ? "active" : ""}`}
          type="button"
          onClick={() => selectPipeType("solid")}
        >
          <span className="pipe-check">{pipeType === "solid" ? "✓" : ""}</span>
          <span>
            <strong>Solid Pipe</strong>
            <small>Solid ONNX model and breaker</small>
          </span>
        </button>

        <div className="control-group">
          <label htmlFor="confidence">Confidence cutoff</label>
          <input
            id="confidence"
            type="range"
            min="0.05"
            max="1"
            step="0.05"
            value={confidence}
            onChange={(event) => setActiveConfidence(Number(event.target.value))}
          />
          <div className="control-row">
            <span>Low</span>
            <strong>{confidence.toFixed(2)}</strong>
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="iou">IOU cutoff</label>
          <input
            id="iou"
            type="range"
            min="0.05"
            max="1"
            step="0.05"
            value={iou}
            onChange={(event) => setIou(Number(event.target.value))}
          />
          <div className="control-row">
            <span>NMS</span>
            <strong>{iou.toFixed(2)}</strong>
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="imageSize">YOLO processing size</label>
          <input
            id="imageSize"
            type="range"
            min="320"
            max="1280"
            step="32"
            value={imageSize}
            onChange={(event) => setImageSize(Number(event.target.value))}
          />
          <div className="control-row">
            <span>Pixels</span>
            <strong>{imageSize}</strong>
          </div>
        </div>

        <div className="control-group">
          <div className="toggle-row">
            <label htmlFor="glareFilter">Glare reduction</label>
            <input
              id="glareFilter"
              type="checkbox"
              checked={glareFilter}
              onChange={(event) => setGlareFilter(event.target.checked)}
            />
          </div>
        </div>

        <div className="control-group profile">
          <strong>Active stream</strong>
          <span>Pipe type: {activeStream.label}</span>
          <span>Model: {activeStream.model}</span>
          <span>Breaker: {activeStream.breaker}</span>
        </div>
      </aside>

      <main className="main">
        <header className="header">
          <div>
            <div className="eyebrow">Computer vision dashboard</div>
            <h1>Industrial Pipe Counter</h1>
            <p className="subtitle">
              Select the pipe stream, choose a sample inspection test or upload your own image,
              and run the matching ONNX model with its own type-breaker rules.
            </p>
          </div>
          <div className="status-pill">Server ready</div>
        </header>

        {/* Top Section: Controls and Detection Analytics */}
        <div className="dashboard-grid">
          <section className="dashboard-column">
            <div className="section-panel">
              <div className="panel-title">Source Image</div>
              <p className="panel-note">Run test on attached samples or upload an inspection photo.</p>
            </div>

            {/* Source Mode Switcher */}
            <div className="source-tabs">
              <button
                type="button"
                className={`source-tab ${sourceMode === "sample" ? "active" : ""}`}
                onClick={() => setSourceMode("sample")}
              >
                ⚡ Choose Sample Test
              </button>
              <button
                type="button"
                className={`source-tab ${sourceMode === "upload" ? "active" : ""}`}
                onClick={() => setSourceMode("upload")}
              >
                📁 Upload Custom Image
              </button>
            </div>

            {sourceMode === "sample" ? (
              <div className="sample-section">
                <div className="sample-filter-row">
                  <span className="sample-label">Filter:</span>
                  <div className="sample-chips">
                    <button
                      type="button"
                      className={`sample-chip ${sampleFilter === "all" ? "active" : ""}`}
                      onClick={() => setSampleFilter("all")}
                    >
                      All ({SAMPLE_IMAGES.length})
                    </button>
                    <button
                      type="button"
                      className={`sample-chip ${sampleFilter === "hollow" ? "active" : ""}`}
                      onClick={() => setSampleFilter("hollow")}
                    >
                      Hollow Pipes (3)
                    </button>
                    <button
                      type="button"
                      className={`sample-chip ${sampleFilter === "solid" ? "active" : ""}`}
                      onClick={() => setSampleFilter("solid")}
                    >
                      Solid Pipes (7)
                    </button>
                  </div>
                </div>

                <div className="sample-dropdown-box">
                  <select
                    className="sample-select"
                    value={selectedSampleId}
                    onChange={(e) => {
                      const found = SAMPLE_IMAGES.find((s) => s.id === e.target.value);
                      if (found) loadSampleImage(found);
                    }}
                  >
                    <option value="">-- Choose a sample test image --</option>
                    <optgroup label="Hollow Pipe Samples">
                      {SAMPLE_IMAGES.filter((s) => s.type === "hollow").map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.id} — {s.label}
                        </option>
                      ))}
                    </optgroup>
                    <optgroup label="Solid Pipe Samples">
                      {SAMPLE_IMAGES.filter((s) => s.type === "solid").map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.id} — {s.label}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                </div>

                <div className="sample-grid">
                  {filteredSamples.map((sample) => {
                    const isSelected = selectedSampleId === sample.id;
                    return (
                      <button
                        key={sample.id}
                        type="button"
                        className={`sample-card ${isSelected ? "selected" : ""}`}
                        onClick={() => loadSampleImage(sample)}
                        disabled={sampleLoading}
                      >
                        <div className="sample-card-head">
                          <span className={`sample-badge ${sample.type}`}>
                            {sample.type === "hollow" ? "Hollow" : "Solid"}
                          </span>
                          {isSelected && <span className="sample-check">✓</span>}
                        </div>
                        <strong className="sample-title">{sample.id}</strong>
                        <span className="sample-sub">{sample.label}</span>
                      </button>
                    );
                  })}
                </div>

                {sampleLoading && (
                  <div className="sample-loading-pill">Loading sample image into inspection buffer...</div>
                )}
              </div>
            ) : (
              <input
                className="uploader"
                type="file"
                accept="image/jpeg,image/png"
                onChange={handleFileChange}
              />
            )}

            <div className="action-bar">
              <button
                className="primary-button"
                type="button"
                onClick={handleDetect}
                disabled={!file || loading || sampleLoading}
              >
                {loading ? "Processing..." : selectedSampleId ? `Test Run: ${selectedSampleId}` : "Run Detection"}
              </button>
              <span className="hint">
                {file
                  ? `${activeStream.label} selected (${imageMode.toLowerCase()})`
                  : "Select a sample or upload an image"}
              </span>
            </div>
          </section>

          <section className="dashboard-column">
            <div className="section-panel">
              <div className="panel-title">Detection Analytics</div>
              <p className="panel-note">Inspection results, batch acceptance and pipe counts.</p>
            </div>

            {error && <div className="message">{error}</div>}

            {result?.invalidDetection && (
              <div className="invalid-result">
                <strong>Invalid detection.</strong>
                <span>Please choose the correct pipe type for this batch.</span>
                <small>{result.reason}</small>
              </div>
            )}

            {result?.accepted ? (
              <div className="kpi-grid">
                <div className="kpi-card">
                  <div className="kpi-label">Count</div>
                  <div className="kpi-value">{result.count}</div>
                  <div className="kpi-foot">{activeStream.label} accepted</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Confidence</div>
                  <div className="kpi-value">{result.avgConfidence.toFixed(1)}%</div>
                  <div className="kpi-foot">Mean model score</div>
                </div>
              </div>
            ) : (
              !error && !result?.invalidDetection && (
                <div className="kpi-placeholder">
                  <div className="kpi-placeholder-text">
                    <strong>Awaiting Detection Execution</strong>
                    <span>Click "{selectedSampleId ? `Test Run: ${selectedSampleId}` : 'Run Detection'}" to analyze the image with the {activeStream.label} model.</span>
                  </div>
                </div>
              )
            )}

            {result?.geminiAnalysis && result.geminiAnalysis.trim() !== "" && result.geminiAnalysis.toLowerCase() !== "none" && (
              <div className="gemini-panel">
                <div className="gemini-header">Gemini Structural & Glare Analysis</div>
                <div className="gemini-body">{result.geminiAnalysis}</div>
              </div>
            )}
          </section>
        </div>

        {/* Bottom Section: Side-by-Side Synchronized Image Comparison */}
        <div className="inspection-comparison-row">
          <div className="image-card">
            <div className="image-card-header">
              <div className="image-card-title">
                <span>Source Inspection Image</span>
                {file && <span className="image-status-pill">{file.name || selectedSampleId}</span>}
              </div>
              <span className="image-card-sub">
                {activeStream.label} aligned ({imageMode})
              </span>
            </div>
            {previewUrl ? (
              <div className="image-frame">
                <img className="comparison-image" src={previewUrl} alt="Inspection source" />
              </div>
            ) : (
              <div className="image-frame empty">
                <div className="empty-state-content">
                  <strong>No Source Image</strong>
                  <span>Select a sample test image above or upload your own inspection file.</span>
                </div>
              </div>
            )}
          </div>

          <div className="image-card">
            <div className="image-card-header">
              <div className="image-card-title">
                <span>Processed Detection Output</span>
                {result?.accepted && (
                  <span className="image-status-pill success">{result.count} Pipes Detected</span>
                )}
              </div>
              <span className="image-card-sub">
                {result?.accepted ? `${result.avgConfidence.toFixed(1)}% Confidence` : "Awaiting run"}
              </span>
            </div>
            {result?.annotatedImage ? (
              <div className="image-frame">
                <img className="comparison-image" src={result.annotatedImage} alt="Detection output" />
              </div>
            ) : (
              <div className="image-frame empty">
                <div className="empty-state-content">
                  <strong>Awaiting Detection Output</strong>
                  <span>
                    {loading
                      ? "Running neural network inference..."
                      : "The annotated bounding ring output will appear here once processed."}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
