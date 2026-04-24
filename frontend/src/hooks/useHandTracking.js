/**
 * useHandTracking - Two hand support version
 * Detects up to 2 hands and sends both to backend
 */

import { useEffect, useRef, useState, useCallback } from 'react'

export function useHandTracking({ onResult, enabled = true }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const handsRef = useRef(null)
  const cameraRef = useRef(null)
  const animFrameRef = useRef(null)

  const [isReady, setIsReady] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  const drawLandmarks = useCallback((ctx, results) => {
    const canvas = canvasRef.current
    if (!canvas || !ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return

    const connections = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],
      [0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],
      [5,9],[9,13],[13,17],
    ]
    const W = canvas.width
    const H = canvas.height

    // Draw each hand with different color
    const handColors = ['rgba(0,245,255,0.7)', 'rgba(191,0,255,0.7)']
    const tipColors  = ['#00ff88', '#ffee00']

    results.multiHandLandmarks.forEach((lms, handIdx) => {
      const lineColor = handColors[handIdx] || handColors[0]
      const tipColor  = tipColors[handIdx]  || tipColors[0]

      ctx.strokeStyle = lineColor
      ctx.lineWidth = 2
      connections.forEach(([a, b]) => {
        ctx.beginPath()
        ctx.moveTo(lms[a].x * W, lms[a].y * H)
        ctx.lineTo(lms[b].x * W, lms[b].y * H)
        ctx.stroke()
      })

      lms.forEach((lm, i) => {
        const isTip = [4,8,12,16,20].includes(i)
        ctx.beginPath()
        ctx.arc(lm.x * W, lm.y * H, isTip ? 6 : 4, 0, 2 * Math.PI)
        ctx.fillStyle = isTip ? tipColor : lineColor
        ctx.shadowColor = ctx.fillStyle
        ctx.shadowBlur = 8
        ctx.fill()
        ctx.shadowBlur = 0
      })

      // Label hand
      ctx.fillStyle = lineColor
      ctx.font = '12px monospace'
      const label = results.multiHandedness?.[handIdx]?.label || `Hand ${handIdx+1}`
      ctx.fillText(label, lms[0].x * W + 5, lms[0].y * H - 5)
    })
  }, [])

  useEffect(() => {
    if (!enabled) return
    let active = true

    // Suppress WebGL alert
    const origAlert = window.alert
    window.alert = (msg) => {
      if (typeof msg === 'string' &&
         (msg.toLowerCase().includes('webgl') ||
          msg.toLowerCase().includes('canvas') ||
          msg.toLowerCase().includes('video frame'))) {
        console.warn('[Suppressed]:', msg)
        return
      }
      origAlert.call(window, msg)
    }

    async function initMediaPipe() {
      try {
        setIsLoading(true)
        setError(null)

        const { Hands } = await import('@mediapipe/hands')
        const { Camera } = await import('@mediapipe/camera_utils')

        if (!active) return

        const hands = new Hands({
          locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        })

        // Enable 2 hands
        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.7,
          minTrackingConfidence: 0.5,
        })

        hands.onResults((results) => {
          if (!active) return
          const canvas = canvasRef.current
          if (!canvas) return
          const ctx = canvas.getContext('2d')
          drawLandmarks(ctx, results)

          if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const lms1 = results.multiHandLandmarks[0]
            const lms2 = results.multiHandLandmarks[1] || null
            onResult?.(lms1, results.multiHandedness?.[0], lms2)
          }
        })

        handsRef.current = hands

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        })

        if (!active) { stream.getTracks().forEach(t => t.stop()); return }

        const video = videoRef.current
        if (!video) return
        video.srcObject = stream
        video.play()

        const camera = new Camera(video, {
          onFrame: async () => {
            if (!active || !handsRef.current) return
            await handsRef.current.send({ image: video })
          },
          width: 640,
          height: 480,
        })

        cameraRef.current = camera
        camera.start()

        setIsReady(true)
        setIsLoading(false)

      } catch (err) {
        if (!active) return
        console.error('MediaPipe init error:', err)
        setError(err.message || 'Failed to initialize camera/MediaPipe')
        setIsLoading(false)
      }
    }

    initMediaPipe()


    return () => {
      active = false
      cameraRef.current?.stop()
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      }
      handsRef.current?.close()
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      window.alert = origAlert
    }
  }, [enabled, drawLandmarks])

  return { videoRef, canvasRef, isReady, isLoading, error }
}