import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type LinePoint = {
  x: number
  y: number
}

type LineState = {
  start: LinePoint
  end: LinePoint
}

type Counts = {
  right_to_left: number
  left_to_right: number
}

type StatusResponse = {
  line: {
    start: LinePoint
    end: LinePoint
  }
  counts: Counts
  updated_at: number
}

type DragTarget = 'start' | 'end'

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

const defaultLine: LineState = {
  start: { x: 0.5, y: 0.2 },
  end: { x: 0.5, y: 0.8 },
}

const App = () => {
  const backendBase = useMemo(() => {
    const base = import.meta.env.VITE_BACKEND_URL?.toString().trim()
    if (!base) {
      return 'http://localhost:8000'
    }
    return base.endsWith('/') ? base : `${base}/`
  }, [])

  const makeHttpUrl = useCallback(
    (path: string) => new URL(path, backendBase).toString(),
    [backendBase],
  )

  const makeWsUrl = useCallback(
    (path: string) => {
      const url = new URL(path, backendBase)
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
      return url.toString()
    },
    [backendBase],
  )

  const containerRef = useRef<HTMLDivElement | null>(null)
  const [containerSize, setContainerSize] = useState({ width: 960, height: 540 })
  const [line, setLine] = useState<LineState>(defaultLine)
  const [counts, setCounts] = useState<Counts>({
    right_to_left: 0,
    left_to_right: 0,
  })
  const [statusLoaded, setStatusLoaded] = useState(false)
  const [isSyncing, setIsSyncing] = useState(false)
  const [connectionState, setConnectionState] = useState<'connecting' | 'open' | 'closed'>('connecting')
  const [hasWebsocket, setHasWebsocket] = useState(false)
  const [dragging, setDragging] = useState<DragTarget | null>(null)

  const lineRef = useRef<LineState>(line)
  const draggingRef = useRef<DragTarget | null>(null)

  useEffect(() => {
    lineRef.current = line
  }, [line])

  useEffect(() => {
    const node = containerRef.current
    if (!node) return

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        setContainerSize({
          width: Math.max(width, 1),
          height: Math.max(height, 1),
        })
      }
    })
    observer.observe(node)

    return () => observer.disconnect()
  }, [])

  const applyStatus = useCallback(
    (data: StatusResponse) => {
      setCounts(data.counts)
      if (!draggingRef.current) {
        setLine({
          start: { x: data.line.start.x, y: data.line.start.y },
          end: { x: data.line.end.x, y: data.line.end.y },
        })
      }
      setStatusLoaded(true)
    },
    [],
  )

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(makeHttpUrl('/status'), {
        cache: 'no-store',
      })
      if (!response.ok) {
        throw new Error(`Status request failed: ${response.status}`)
      }
      const data = (await response.json()) as StatusResponse
      applyStatus(data)
    } catch (error) {
      console.error('[status]', error)
    }
  }, [applyStatus, makeHttpUrl])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  useEffect(() => {
    let socket: WebSocket | null = null
    let reconnectTimer: number | null = null
    let intentionalClose = false

    const connect = () => {
      setConnectionState('connecting')
      socket = new WebSocket(makeWsUrl('/ws/status'))

      socket.onopen = () => {
        setHasWebsocket(true)
        setConnectionState('open')
      }

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as StatusResponse
          applyStatus(payload)
        } catch (error) {
          console.error('[ws] failed to parse payload', error)
        }
      }

      socket.onerror = (event) => {
        console.error('[ws] error', event)
        socket?.close()
      }

      socket.onclose = () => {
        setHasWebsocket(false)
        setConnectionState('closed')
        if (!intentionalClose) {
          reconnectTimer = window.setTimeout(connect, 5000)
        }
      }
    }

    connect()

    return () => {
      intentionalClose = true
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer)
      }
      socket?.close()
    }
  }, [applyStatus, makeWsUrl])

  useEffect(() => {
    if (hasWebsocket) {
      return
    }
    const interval = window.setInterval(fetchStatus, 5000)
    return () => {
      window.clearInterval(interval)
    }
  }, [fetchStatus, hasWebsocket])

  const persistLine = useCallback(
    async (nextLine: LineState) => {
      setIsSyncing(true)
      try {
        const response = await fetch(makeHttpUrl('/line'), {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(nextLine),
        })
        if (!response.ok) {
          throw new Error(`Failed to update line: ${response.status}`)
        }
        const data = (await response.json()) as StatusResponse
        applyStatus(data)
      } catch (error) {
        console.error('[line]', error)
      } finally {
        setIsSyncing(false)
      }
    },
    [applyStatus, makeHttpUrl],
  )

  const updatePointFromEvent = useCallback(
    (event: PointerEvent) => {
      const rect = containerRef.current?.getBoundingClientRect()
      const target = draggingRef.current
      if (!rect || !target) return

      const x = clamp01((event.clientX - rect.left) / rect.width)
      const y = clamp01((event.clientY - rect.top) / rect.height)

      setLine((current) => {
        const next =
          target === 'start'
            ? { ...current, start: { x, y } }
            : { ...current, end: { x, y } }
        lineRef.current = next
        return next
      })
    },
    [],
  )

  const handlePointerDown = useCallback(
    (target: DragTarget) => (event: React.PointerEvent<SVGCircleElement>) => {
      event.preventDefault()
      draggingRef.current = target
      setDragging(target)

      const handleMove = (moveEvent: PointerEvent) => {
        updatePointFromEvent(moveEvent)
      }

      const finish = () => {
        window.removeEventListener('pointermove', handleMove)
        window.removeEventListener('pointerup', finish)
        window.removeEventListener('pointercancel', finish)
        const latest = lineRef.current
        draggingRef.current = null
        setDragging(null)
        if (latest) {
          persistLine(latest)
        }
      }

      window.addEventListener('pointermove', handleMove)
      window.addEventListener('pointerup', finish)
      window.addEventListener('pointercancel', finish)
    },
    [persistLine, updatePointFromEvent],
  )

  const linePixels = useMemo(
    () => ({
      start: {
        x: line.start.x * containerSize.width,
        y: line.start.y * containerSize.height,
      },
      end: {
        x: line.end.x * containerSize.width,
        y: line.end.y * containerSize.height,
      },
    }),
    [containerSize, line],
  )

  const streamUrl = useMemo(() => makeHttpUrl('/stream'), [makeHttpUrl])

  return (
    <div className="app">
      <header className="app__header">
        <h1>Visitor Flow Monitor</h1>
        <div className={`status status--${connectionState}`}>
          WebSocket:
          <span>
            {connectionState === 'connecting'
              ? '接続中...'
              : connectionState === 'open'
                ? '接続済み'
                : '未接続'}
          </span>
        </div>
      </header>

      <main className="layout">
        <section className="layout__primary">
          <div className="video-wrapper" ref={containerRef}>
            <video
              className="video"
              src={streamUrl}
              autoPlay
              muted
              playsInline
              controls={false}
              crossOrigin="anonymous"
            />
            <svg className="overlay" width={containerSize.width} height={containerSize.height}>
              <line
                x1={linePixels.start.x}
                y1={linePixels.start.y}
                x2={linePixels.end.x}
                y2={linePixels.end.y}
                className="overlay__line"
              />
              <circle
                className={`overlay__handle ${dragging === 'start' ? 'overlay__handle--dragging' : ''}`}
                cx={linePixels.start.x}
                cy={linePixels.start.y}
                r={10}
                onPointerDown={handlePointerDown('start')}
              />
              <circle
                className={`overlay__handle ${dragging === 'end' ? 'overlay__handle--dragging' : ''}`}
                cx={linePixels.end.x}
                cy={linePixels.end.y}
                r={10}
                onPointerDown={handlePointerDown('end')}
              />
            </svg>
            {!statusLoaded && <div className="loading">初期データ取得中...</div>}
          </div>
          <p className="hint">ラインの端点をドラッグして配置し直すと即座にバックエンドへ保存されます。</p>
        </section>

        <aside className="layout__aside">
          <div className="card">
            <h2>現在のカウント</h2>
            <dl>
              <div>
                <dt>右 → 左</dt>
                <dd>{counts.right_to_left}</dd>
              </div>
              <div>
                <dt>左 → 右</dt>
                <dd>{counts.left_to_right}</dd>
              </div>
            </dl>
          </div>

          <div className="card">
            <h2>同期状態</h2>
            <p>{isSyncing ? 'ライン情報を同期中...' : '最新のライン設定が反映されています。'}</p>
            <p>バックエンド: {backendBase}</p>
          </div>
        </aside>
      </main>
    </div>
  )
}

export default App
