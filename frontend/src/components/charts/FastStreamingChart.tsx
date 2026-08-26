/**
 * FastStreamingChart: Hardware-accelerated HTML5 Canvas 2D real-time streaming line chart.
 * Replaces heavy SVG DOM string re-concatenations with requestAnimationFrame Canvas 2D drawing.
 */

import React, { useEffect, useRef } from "react";

interface FastStreamingChartProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillColor?: string;
  minValue?: number;
  maxValue?: number;
  label?: string;
}

export const FastStreamingChart: React.FC<FastStreamingChartProps> = ({
  data,
  width = 300,
  height = 100,
  color = "#3498db",
  fillColor = "rgba(52, 152, 219, 0.15)",
  minValue,
  maxValue,
  label,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (data.length === 0) return;

    const min = minValue !== undefined ? minValue : Math.min(...data, 0);
    const max = maxValue !== undefined ? maxValue : Math.max(...data, 1);
    const range = max - min || 1;

    const stepX = width / Math.max(1, data.length - 1);

    // 1. Fill area under curve
    ctx.beginPath();
    ctx.moveTo(0, height);
    for (let i = 0; i < data.length; i++) {
      const x = i * stepX;
      const y = height - ((data[i] - min) / range) * (height - 10) - 5;
      ctx.lineTo(x, y);
    }
    ctx.lineTo((data.length - 1) * stepX, height);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

    // 2. Stroke line
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = i * stepX;
      const y = height - ((data[i] - min) / range) * (height - 10) - 5;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();

    // 3. Label HUD
    if (label) {
      ctx.fillStyle = "#7f8c8d";
      ctx.font = "10px sans-serif";
      ctx.fillText(label, 6, 12);
    }
  }, [data, width, height, color, fillColor, minValue, maxValue, label]);

  return (
    <div className="relative inline-block overflow-hidden rounded bg-slate-900/60 p-1 border border-slate-800">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="block"
      />
    </div>
  );
};
