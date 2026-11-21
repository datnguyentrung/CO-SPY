import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import './App.css'

type Health = {
  status: 'ok' | 'loading' | 'degraded'
  device: string
}

type ModelResult = {
  model_name: string
  prediction: 'AI_GENERATED' | 'REAL'
  confidence: number
  real_probability: number
  ai_probability: number
  processing_time_ms: number
  raw_score: number
}

type FusionBranches = {
  semantic: ModelResult
  artifact: ModelResult
  npr: ModelResult
}

type Prediction = {
  original: ModelResult
  fusion_v3: ModelResult & { branches?: FusionBranches }
  total_processing_time_ms: number
  device: string
}

type ResultPanelProps = {
  architecture: string
  branches?: FusionBranches
  loading: boolean
  result?: ModelResult
  subtitle: string
  title: string
}

type BranchResultProps = {
  label: string
  result: ModelResult
}

const imageTypes = new Set(['image/jpeg', 'image/png', 'image/webp'])
const percent = (value: number) => `${(value * 100).toFixed(2)}%`

function ResultPanel({ architecture, branches, loading, result, subtitle, title }: ResultPanelProps) {
  const isAi = result?.prediction === 'AI_GENERATED'

  return (
    <article className="model-result" aria-busy={loading}>
      <div className="model-heading">
        <div>
          <h2>{title}</h2>
          <p className="model-role">{subtitle}</p>
          <p className="architecture">{architecture}</p>
        </div>
        {result && (
          <span className={`outcome ${isAi ? 'ai' : 'real'}`}>
            <i className="outcome-indicator" aria-hidden="true" />
            <span className="outcome-text">{isAi ? 'AI generated' : 'Real image'}</span>
          </span>
        )}
      </div>

      {loading ? (
        <div className="result-state loading-state" role="status">
          <span className="spinner" />
          <div>
            <strong>Analyzing image</strong>
            <p>Running the detector with its evaluation transforms.</p>
          </div>
        </div>
      ) : result ? (
        <div className="result-content">
          <div className="confidence-row">
            <div>
              <span>Confidence</span>
              <strong>{percent(result.confidence)}</strong>
            </div>
            <div className="time-stat">
              <span>Inference</span>
              <b>{result.processing_time_ms} ms</b>
            </div>
          </div>

          <div className="probability-bar" aria-label={`Real ${percent(result.real_probability)}, AI generated ${percent(result.ai_probability)}`}>
            <span className="real-segment" style={{ width: percent(result.real_probability) }} />
            <span className="ai-segment" style={{ width: percent(result.ai_probability) }} />
          </div>

          <dl className="probabilities">
            <div>
              <dt><i className="real-key" />Real</dt>
              <dd>{percent(result.real_probability)}</dd>
            </div>
            <div>
              <dt><i className="ai-key" />AI generated</dt>
              <dd>{percent(result.ai_probability)}</dd>
            </div>
          </dl>

          {branches && (
            <section className="branch-signals" aria-label="Fusion v3 branch signals">
              <div className="branch-signals-heading">
                <h3>Branch signals</h3>
                <p>Independent predictions, not fusion weights.</p>
              </div>
              <div className="branch-grid">
                <BranchResult label="Semantic" result={branches.semantic} />
                <BranchResult label="Artifact-VAE" result={branches.artifact} />
                <BranchResult label="NPR" result={branches.npr} />
              </div>
            </section>
          )}
        </div>
      ) : (
        <div className="result-state empty-state">
          <span>Waiting for input</span>
          <p>The live prediction and probabilities will appear here.</p>
        </div>
      )}
    </article>
  )
}

function BranchResult({ label, result }: BranchResultProps) {
  const isAi = result.prediction === 'AI_GENERATED'

  return (
    <article className="branch-result">
      <div className="branch-heading">
        <h3>{label}</h3>
        <span className={`branch-outcome ${isAi ? 'ai' : 'real'}`}>
          {isAi ? 'AI generated' : 'Real image'}
        </span>
      </div>
      <strong>{percent(result.confidence)}</strong>
      <div
        className="probability-bar branch-probability-bar"
        aria-label={`${label}: Real ${percent(result.real_probability)}, AI generated ${percent(result.ai_probability)}`}
      >
        <span className="real-segment" style={{ width: percent(result.real_probability) }} />
        <span className="ai-segment" style={{ width: percent(result.ai_probability) }} />
      </div>
      <div className="branch-probabilities">
        <span>Real {percent(result.real_probability)}</span>
        <span>AI {percent(result.ai_probability)}</span>
      </div>
    </article>
  )
}

export default function App() {
  const input = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    const poll = async () => {
      try {
        const response = await fetch('/api/health')
        if (live) setHealth(await response.json())
      } catch {
        if (live) setHealth(null)
      }
    }
    void poll()
    const id = window.setInterval(poll, 5000)
    return () => {
      live = false
      window.clearInterval(id)
    }
  }, [])

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview)
  }, [preview])

  const choose = (next?: File) => {
    setError(null)
    setPrediction(null)
    if (!next) return
    if (!imageTypes.has(next.type)) {
      setError('Choose a JPG, PNG, or WEBP image.')
      return
    }
    if (next.size > 15 * 1024 * 1024) {
      setError('Choose an image smaller than 15 MB.')
      return
    }
    if (preview) URL.revokeObjectURL(preview)
    setFile(next)
    setPreview(URL.createObjectURL(next))
  }

  const clear = () => {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
    setPrediction(null)
    setError(null)
    if (input.current) input.current.value = ''
  }

  const analyze = async () => {
    if (!file || loading) return
    setLoading(true)
    setError(null)
    setPrediction(null)
    try {
      const body = new FormData()
      body.append('image', file)
      const response = await fetch('/api/predict', { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Unable to analyze this image. Please try again.')
      setPrediction(data)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to analyze this image. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const healthLabel = health?.status === 'ok'
    ? `Models ready · ${health.device.toUpperCase()}`
    : health?.status === 'loading'
      ? `Loading models · ${health.device.toUpperCase()}`
      : health
        ? 'Models need attention'
        : 'Connecting to backend'

  const handleDrop = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    choose(event.dataTransfer.files[0])
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span><b>CO-SPY</b><small>Image forensics</small></span>
        </div>
        <div className={`system-status ${health?.status === 'ok' ? 'ready' : ''}`}>
          <i aria-hidden="true" />
          <span>{healthLabel}</span>
        </div>
      </header>

      <main>
        <section className="page-heading">
          <div>
            <h1>One image. Two detectors.</h1>
            <p>Compare the original CO-SPY pipeline with the improved Fusion v3 model using the same source image.</p>
          </div>
          <button className="analyze" type="button" onClick={analyze} disabled={!file || loading || health?.status !== 'ok'}>
            {loading ? 'Analyzing…' : 'Analyze image'}
          </button>
        </section>

        {health?.status === 'degraded' && (
          <div className="notice" role="status">The backend is running, but one or more models are unavailable. Check the backend console for checkpoint details.</div>
        )}
        {error && <div className="error" role="alert">{error}</div>}

        <section className="comparison-workspace">
          <article
            className={`source-panel ${dragging ? 'dragging' : ''}`}
            onDragOver={(event: DragEvent) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <div className="panel-heading">
              <div>
                <h2>Source image</h2>
              </div>
              {file && <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>}
            </div>

            {preview ? (
              <div className="image-stage">
                <img src={preview} alt={`Preview of ${file?.name}`} />
              </div>
            ) : (
              <button className="upload-stage" type="button" onClick={() => input.current?.click()}>
                <span className="upload-symbol" aria-hidden="true">↥</span>
                <strong>Drop an image here</strong>
                <span>or click to browse · JPG, PNG, WEBP · max 15 MB</span>
              </button>
            )}

            <div className="source-footer">
              <div className="file-name">
                <span>{file ? 'Selected file' : 'No image selected'}</span>
                {file && <strong title={file.name}>{file.name}</strong>}
              </div>
              <div className="file-actions">
                {file && <button className="text-action" type="button" onClick={clear}>Remove</button>}
                <button className="secondary-action" type="button" onClick={() => input.current?.click()}>
                  {file ? 'Replace image' : 'Choose image'}
                </button>
              </div>
            </div>
            <input
              ref={input}
              type="file"
              hidden
              aria-label="Choose a source image"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event: ChangeEvent<HTMLInputElement>) => choose(event.target.files?.[0])}
            />
          </article>

          <section className="results-panel" aria-label="Model comparison results">
            <div className="results-heading">
              <div>
                <h2>Detection results</h2>
              </div>
              {prediction && <span>Total {prediction.total_processing_time_ms} ms</span>}
            </div>
            <ResultPanel
              title="CO-SPY Original"
              subtitle="Model gốc · Trước cải tiến"
              architecture="Semantic + Artifact + Fusion"
              result={prediction?.original}
              loading={loading}
            />
            <ResultPanel
              title="CO-SPY Fusion v3"
              subtitle="Model sau cải tiến"
              architecture="Semantic + Artifact + NPR + Fusion v3"
              result={prediction?.fusion_v3}
              branches={prediction?.fusion_v3.branches}
              loading={loading}
            />
          </section>
        </section>
      </main>
    </div>
  )
}
