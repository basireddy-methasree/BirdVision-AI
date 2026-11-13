
import { useEffect, useRef, useState } from 'react'
import './App.css'

const BACKEND_BASE = 'http://127.0.0.1:8000'
const API_BASE = `${BACKEND_BASE}/api`
const PREDICT_URL = `${API_BASE}/predict`
const HISTORY_URL = `${API_BASE}/history`

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [activeVisualization, setActiveVisualization] = useState('overlay')

  // Bird details state
  const [birdDetails, setBirdDetails] = useState(null)
  const [detailsLoading, setDetailsLoading] = useState(false)
  const [detailsError, setDetailsError] = useState('')

  // Prediction history state
  const [history, setHistory] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [deletingId, setDeletingId] = useState(null)

  const fileInputRef = useRef(null)

  // Clean up preview URL
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  // Fetch prediction history
  const fetchHistory = async () => {
    setHistoryLoading(true)
    setHistoryError('')

    try {
      const response = await fetch(HISTORY_URL)
      const data = await response.json()

      if (!response.ok || !data.success) {
        throw new Error(data.detail || 'Unable to load prediction history.')
      }

      setHistory(data.history || [])
    } catch (err) {
      console.error('History error:', err)
      setHistoryError(
        err.message === 'Failed to fetch'
          ? 'Cannot connect to the backend server.'
          : err.message
      )
    } finally {
      setHistoryLoading(false)
    }
  }

  // Load history when application opens
  useEffect(() => {
    fetchHistory()
  }, [])

  // Image validation and selection
  const handleImage = (file) => {
    setError('')
    setPrediction(null)
    setBirdDetails(null)
    setDetailsError('')
    setDetailsLoading(false)
    setActiveVisualization('overlay')

    if (!file) return

    const allowedTypes = ['image/jpeg', 'image/png']

    if (!allowedTypes.includes(file.type)) {
      setError('Please select a JPG, JPEG or PNG image.')
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('Image size must be less than 10 MB.')
      return
    }

    setSelectedImage(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]

    if (file) {
      handleImage(file)
    }

    event.target.value = ''
  }

  const openFileBrowser = () => {
    fileInputRef.current?.click()
  }

  // Drag and drop
  const handleDragOver = (event) => {
    event.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (event) => {
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setIsDragging(false)
    }
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)

    const file = event.dataTransfer.files?.[0]
    handleImage(file)
  }

  // Remove selected image
  const removeImage = () => {
    setSelectedImage(null)
    setPreviewUrl('')
    setPrediction(null)
    setBirdDetails(null)
    setDetailsError('')
    setDetailsLoading(false)
    setError('')
    setActiveVisualization('overlay')

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Bird identification
  const identifyBird = async () => {
    if (!selectedImage) {
      setError('Please select a bird image first.')
      return
    }

    setIsLoading(true)
    setError('')
    setPrediction(null)
    setBirdDetails(null)
    setDetailsError('')

    try {
      const formData = new FormData()
      formData.append('file', selectedImage)

      const response = await fetch(PREDICT_URL, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Unable to identify the bird.')
      }

      if (!data.success) {
        throw new Error('Prediction was not successful.')
      }

      if (!data.gradcam) {
        throw new Error(
          'Prediction succeeded, but Grad-CAM images were not returned.'
        )
      }

      setPrediction(data)
      setActiveVisualization('overlay')

      // Refresh history after successful prediction
      await fetchHistory()

      // Navigate to prediction results
      setTimeout(() => {
        document.getElementById('results')?.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        })
      }, 100)

    } catch (err) {
      console.error('Prediction error:', err)

      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot connect to the AI server. Please check whether the backend is running.'
          : err.message || 'Something went wrong during prediction.'
      )
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch detailed information about the predicted bird
  const fetchBirdDetails = async () => {
    if (!prediction?.predicted_species) return

    setDetailsLoading(true)
    setDetailsError('')
    setBirdDetails(null)

    try {
      const species = encodeURIComponent(prediction.predicted_species)
      const response = await fetch(`${API_BASE}/birds/${species}`)
      const data = await response.json()

      if (!response.ok || !data.success) {
        throw new Error(data.detail || 'Unable to load bird details.')
      }

      setBirdDetails(data.bird_details)
    } catch (err) {
      console.error('Bird details error:', err)
      setDetailsError(err.message || 'Unable to load bird details.')
    } finally {
      setDetailsLoading(false)
    }
  }

  // Delete history record
  const handleDeleteHistory = async (predictionId) => {
    const confirmed = window.confirm(
      'Are you sure you want to delete this prediction record?'
    )

    if (!confirmed) return

    setDeletingId(predictionId)
    setHistoryError('')

    try {
      const response = await fetch(
        `${HISTORY_URL}/${predictionId}`,
        { method: 'DELETE' }
      )

      const data = await response.json()

      if (!response.ok || !data.success) {
        throw new Error(data.detail || 'Unable to delete prediction.')
      }

      setHistory((previousHistory) =>
        previousHistory.filter((item) => item.id !== predictionId)
      )

    } catch (err) {
      console.error('Delete history error:', err)
      setHistoryError(err.message || 'Unable to delete prediction.')
    } finally {
      setDeletingId(null)
    }
  }

  // Grad-CAM visualization
  const visualizationImages = prediction?.gradcam

  const currentVisualization = {
    original: visualizationImages?.original_image,
    heatmap: visualizationImages?.heatmap_image,
    overlay: visualizationImages?.overlay_image
  }

  const visualizationLabels = {
    original: 'Original Image',
    heatmap: 'Grad-CAM Heatmap',
    overlay: 'Grad-CAM Overlay'
  }

  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'Date unavailable'

    const date = new Date(
      dateString.replace(' ', 'T') + 'Z'
    )

    if (Number.isNaN(date.getTime())) {
      return dateString
    }

    return date.toLocaleString()
  }

  return (
    <div className="app">

      {/* SIDEBAR */}
      <aside className="sidebar">

        <div className="brand">
          <span className="brand-icon">🪶</span>

          <div>
            <h2>BirdVision</h2>
            <p>AI SPECIES IDENTIFIER</p>
          </div>
        </div>

        <p className="menu-label">WORKSPACE</p>

        <nav className="navigation">
          <a className="nav-item active" href="#dashboard">
            <span>▦</span> Dashboard
          </a>

          <a className="nav-item" href="#identify">
            <span>⌕</span> Identify Bird
          </a>

          <a className="nav-item" href="#results">
            <span>◉</span> AI Results
          </a>

          <a className="nav-item" href="#history">
            <span>◷</span> Prediction History
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <span>🌿</span>
            <p>Discover. Identify. Protect.</p>
          </div>

          <p className="version">BIRD VISION AI · V1.0</p>
        </div>

      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content" id="dashboard">

        {/* TOP NAVIGATION */}
        <header className="topbar">
          <p className="breadcrumb">Workspace / Dashboard</p>

          <div className="profile">
            <span className="status-dot"></span>
            <span>AI Model Ready</span>
            <div className="avatar">BV</div>
          </div>
        </header>

        {/* HERO SECTION */}
        <section className="welcome-section">

          <div className="welcome-content">
            <p className="eyebrow">
              YOUR AI-POWERED BIRD COMPANION
            </p>

            <h1>
              Discover the world<br />
              of <span>birds.</span>
            </h1>

            <p className="welcome-description">
              Explore, identify, and learn about bird species
              using the power of artificial intelligence.
            </p>

            <a className="primary-button" href="#identify">
              Start identifying <span>→</span>
            </a>
          </div>

          <div className="hero-decoration">🦜</div>

        </section>

        {/* STATISTICS */}
        <section className="stats-grid">

          <div className="stat-card">
            <span className="stat-icon">🐦</span>
            <p>SPECIES IN MODEL</p>
            <h2>20</h2>
            <span>Trained bird categories</span>
          </div>

          <div className="stat-card">
            <span className="stat-icon">🎯</span>
            <p>TEST ACCURACY</p>
            <h2>96%</h2>
            <span>Top-1 test accuracy</span>
          </div>

          <div className="stat-card">
            <span className="stat-icon">🧠</span>
            <p>AI ARCHITECTURE</p>
            <h2>ResNet-50</h2>
            <span>Transfer learning model</span>
          </div>

        </section>

        {/* BIRD IDENTIFICATION */}
        <section className="identify-section" id="identify">

          <div className="section-heading">
            <div>
              <p className="eyebrow">AI RECOGNITION</p>
              <h2>Identify a bird</h2>
              <p>
                Upload a bird photograph to discover its species.
              </p>
            </div>

            <span className="model-tag">
              ● MODEL READY
            </span>
          </div>

          {/* FILE INPUT */}
          <input
            ref={fileInputRef}
            id="bird-upload"
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            hidden
            onChange={handleFileChange}
          />

          {/* UPLOAD AREA */}
          <div
            className={`upload-box ${isDragging ? 'drag-active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >

            {previewUrl ? (

              <div className="preview-container">

                <div className="preview-image-wrapper">
                  <img
                    src={previewUrl}
                    alt="Selected bird"
                    className="image-preview"
                  />

                  <span className="image-ready-badge">
                    ✓ Image selected
                  </span>
                </div>

                <h3>Image ready for identification!</h3>

                <p className="file-name">
                  {selectedImage?.name}
                </p>

                <p className="file-size">
                  {selectedImage
                    ? (selectedImage.size / (1024 * 1024)).toFixed(2) + ' MB'
                    : ''}
                </p>

                <div className="preview-actions">
                  <button
                    className="upload-button secondary-button"
                    type="button"
                    onClick={removeImage}
                    disabled={isLoading}
                  >
                    Remove image
                  </button>

                  <button
                    className="upload-button"
                    type="button"
                    onClick={openFileBrowser}
                    disabled={isLoading}
                  >
                    Change image
                  </button>
                </div>

                <button
                  className="upload-button identify-button"
                  type="button"
                  onClick={identifyBird}
                  disabled={isLoading}
                >
                  {isLoading
                    ? 'Analyzing image...'
                    : '🧠 Identify Bird'}
                </button>

              </div>

            ) : (

              <div className="empty-upload">
                <div className="upload-icon">↑</div>

                <h3>Drop your bird image here</h3>

                <p>
                  Drag and drop your image or browse your computer.
                </p>

                <button
                  className="upload-button"
                  type="button"
                  onClick={openFileBrowser}
                >
                  Choose an image
                </button>

                <span className="file-hint">
                  JPG, JPEG or PNG · Maximum size 10 MB
                </span>
              </div>

            )}

            {error && (
              <p className="upload-error" role="alert">
                {error}
              </p>
            )}

          </div>

          {/* LOADING MESSAGE */}
          {isLoading && (
            <div className="prediction-placeholder">
              <span>🔍</span>

              <div>
                <h3>Analyzing your bird image...</h3>
                <p>
                  ResNet-50 is identifying the species and generating
                  the Grad-CAM explanation.
                </p>
              </div>
            </div>
          )}

        </section>

        {/* PREDICTION RESULTS */}
        {prediction && (
          <section className="prediction-results" id="results">

            <p className="eyebrow">IDENTIFICATION COMPLETE</p>
            <h2>Prediction Results</h2>

            {/* MAIN PREDICTION */}
            <div className="prediction-main-card">
              <span className="result-bird-icon">🐦</span>

              <div>
                <p>IDENTIFIED SPECIES</p>
                <h2>{prediction.predicted_species}</h2>
                <h3>{prediction.confidence}%</h3>
                <span>Model confidence score</span>
              </div>
            </div>

            {/* TOP 5 PREDICTIONS */}
            <div className="top-predictions-card">
              <h3>Top 5 Predictions</h3>

              <p>
                Other species considered by the AI model.
              </p>

              {prediction.top_5_predictions.map((item, index) => (
                <div className="prediction-row" key={item.species}>

                  <div className="prediction-row-heading">
                    <span>
                      {index + 1}. {item.species}
                    </span>

                    <strong>{item.confidence}%</strong>
                  </div>

                  <div className="confidence-track">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${item.confidence}%`
                      }}
                    />
                  </div>

                </div>
              ))}
            </div>

            {/* BIRD DETAILS */}
            <div className="bird-details-section">
              <div className="bird-details-intro">
                <div>
                  <p className="eyebrow">EXPLORE THIS SPECIES</p>
                  <h2>Learn more about this bird</h2>
                  <p>Discover its habitat, diet, appearance and conservation status.</p>
                </div>

                <button
                  className="upload-button bird-details-button"
                  type="button"
                  onClick={fetchBirdDetails}
                  disabled={detailsLoading}
                >
                  {detailsLoading
                    ? 'Loading details...'
                    : birdDetails
                      ? '↻ Refresh details'
                      : 'Learn More About This Bird'}
                </button>
              </div>

              {detailsError && (
                <p className="upload-error" role="alert">{detailsError}</p>
              )}

              {detailsLoading && (
                <div className="bird-details-message" aria-live="polite">
                  Loading bird information...
                </div>
              )}

              {birdDetails && (
                <div className="bird-details-card">
                  <div className="bird-details-title">
                    <span className="bird-details-icon">🐦</span>
                    <div>
                      <h3>{birdDetails.common_name || prediction.predicted_species}</h3>
                      <p className="bird-scientific-name">
                        {birdDetails.scientific_name || 'Scientific name not specified'}
                      </p>
                    </div>
                  </div>

                  <div className="bird-details-grid">
                    <div className="bird-detail-item">
                      <span>HABITAT</span>
                      <p>{birdDetails.habitat || 'Information unavailable.'}</p>
                    </div>
                    <div className="bird-detail-item">
                      <span>GEOGRAPHICAL DISTRIBUTION</span>
                      <p>{birdDetails.distribution || 'Information unavailable.'}</p>
                    </div>
                    <div className="bird-detail-item">
                      <span>DIET</span>
                      <p>{birdDetails.diet || 'Information unavailable.'}</p>
                    </div>
                    <div className="bird-detail-item">
                      <span>PHYSICAL CHARACTERISTICS</span>
                      <p>{birdDetails.physical_characteristics || 'Information unavailable.'}</p>
                    </div>
                    <div className="bird-detail-item bird-conservation-item">
                      <span>CONSERVATION STATUS</span>
                      <p>{birdDetails.conservation_status || 'Status not specified.'}</p>
                    </div>
                  </div>

                  <p className="bird-details-note">
                    Species information is provided as an educational reference. Verify conservation information
                    with an authoritative source before using it for research or conservation decisions.
                  </p>
                </div>
              )}
            </div>

            {/* GRAD-CAM EXPLANATION */}
            <div className="gradcam-section">

              <div className="gradcam-heading">
                <p className="eyebrow">MODEL EXPLAINABILITY</p>
                <h2>Visual explanation</h2>

                <p>
                  Explore which image regions influenced the
                  ResNet-50 prediction.
                </p>
              </div>

              {/* VISUALIZATION TABS */}
              <div className="visualization-tabs">

                <button
                  type="button"
                  className={
                    activeVisualization === 'original'
                      ? 'visualization-tab active'
                      : 'visualization-tab'
                  }
                  onClick={() => setActiveVisualization('original')}
                >
                  Original
                </button>

                <button
                  type="button"
                  className={
                    activeVisualization === 'heatmap'
                      ? 'visualization-tab active'
                      : 'visualization-tab'
                  }
                  onClick={() => setActiveVisualization('heatmap')}
                >
                  Heatmap
                </button>

                <button
                  type="button"
                  className={
                    activeVisualization === 'overlay'
                      ? 'visualization-tab active'
                      : 'visualization-tab'
                  }
                  onClick={() => setActiveVisualization('overlay')}
                >
                  Overlay
                </button>

              </div>

              {/* SELECTED VISUALIZATION */}
              <div className="gradcam-main-view">
                <h3>
                  {visualizationLabels[activeVisualization]}
                </h3>

                {currentVisualization[activeVisualization] ? (
                  <img
                    src={currentVisualization[activeVisualization]}
                    alt={visualizationLabels[activeVisualization]}
                    className="gradcam-main-image"
                  />
                ) : (
                  <p className="upload-error">
                    Visualization image is unavailable.
                  </p>
                )}
              </div>

              {/* THREE VISUALIZATION CARDS */}
              <div className="gradcam-grid">

                {Object.entries(visualizationLabels).map(([key, label]) => (
                  <button
                    type="button"
                    key={key}
                    className={`gradcam-card ${
                      activeVisualization === key ? 'selected' : ''
                    }`}
                    onClick={() => setActiveVisualization(key)}
                  >
                    <img
                      src={currentVisualization[key]}
                      alt={label}
                    />

                    <span>{label}</span>
                  </button>
                ))}

              </div>

              <div className="gradcam-information">
                <strong>Understanding Grad-CAM</strong>

                <p>
                  Grad-CAM uses gradients from the model's final
                  convolutional layer to highlight image regions
                  that contribute to the selected prediction.
                  Warmer colors indicate stronger relative
                  activation in this visualization; they do not
                  represent a probability or prove that a region
                  contains a particular bird feature.
                </p>
              </div>

            </div>

            <p className="prediction-disclaimer">
              Confidence scores represent the model's predictions
              and do not guarantee real-world identification accuracy.
            </p>

          </section>
        )}

        {/* PREDICTION HISTORY */}
        <section className="history-section" id="history">

          <div className="section-heading">
            <div>
              <p className="eyebrow">PREVIOUS IDENTIFICATIONS</p>
              <h2>Prediction History</h2>

              <p>
                Review your previously identified bird species.
              </p>
            </div>

            <button
              className="upload-button history-refresh"
              type="button"
              onClick={fetchHistory}
              disabled={historyLoading}
            >
              {historyLoading ? 'Loading...' : '↻ Refresh'}
            </button>
          </div>

          {/* HISTORY SUMMARY */}
          <div className="history-summary">
            <div>
              <span className="history-summary-icon">📋</span>

              <div>
                <p>TOTAL PREDICTIONS</p>
                <h3>{history.length}</h3>
              </div>
            </div>
          </div>

          {/* HISTORY ERROR */}
          {historyError && (
            <p className="upload-error" role="alert">
              {historyError}
            </p>
          )}

          {/* HISTORY LOADING */}
          {historyLoading ? (

            <div className="history-empty">
              <span>🔄</span>
              <h3>Loading prediction history...</h3>
            </div>

          ) : history.length === 0 ? (

            <div className="history-empty">
              <span>🐦</span>
              <h3>No prediction history yet</h3>

              <p>
                Upload and identify your first bird image.
                Your successful predictions will appear here.
              </p>

              <a href="#identify" className="primary-button">
                Identify a bird →
              </a>
            </div>

          ) : (

            <div className="history-list">

              {history.map((item) => (

                <article className="history-card" key={item.id}>

                  {/* SAVED BIRD IMAGE THUMBNAIL */}
                  <div className="history-image-wrapper">

                    {item.image_url ? (
                      <img
                        className="history-image"
                        src={`${BACKEND_BASE}${item.image_url}`}
                        alt={`Uploaded bird for ${item.predicted_species}`}
                        loading="lazy"
                        onError={(event) => {
                          event.currentTarget.style.display = 'none'
                        }}
                      />
                    ) : (
                      <div
                        className="history-bird-icon"
                        aria-label="No saved image"
                      >
                        🐦
                      </div>
                    )}

                  </div>

                  {/* PREDICTION DETAILS */}
                  <div className="history-details">

                    <span className="history-record-id">
                      PREDICTION #{item.id}
                    </span>

                    <h3>{item.predicted_species}</h3>

                    <p className="history-filename">
                      📷 {item.filename}
                    </p>

                    <p className="history-date">
                      🕒 {formatDate(item.created_at)}
                    </p>

                  </div>

                  {/* CONFIDENCE */}
                  <div className="history-confidence">
                    <span>CONFIDENCE</span>

                    <strong>
                      {Number(item.confidence).toFixed(2)}%
                    </strong>
                  </div>

                  {/* DELETE */}
                  <button
                    className="history-delete-button"
                    type="button"
                    onClick={() => handleDeleteHistory(item.id)}
                    disabled={deletingId === item.id}
                    title="Delete prediction"
                  >
                    {deletingId === item.id ? 'Deleting...' : 'Delete'}
                  </button>

                </article>

              ))}

            </div>

          )}

        </section>

        {/* FOOTER */}
        <footer>
          BirdVision AI · Supporting biodiversity through technology.
        </footer>

      </main>

    </div>
  )
}

export default App
