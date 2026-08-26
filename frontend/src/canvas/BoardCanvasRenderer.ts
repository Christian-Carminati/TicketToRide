/**
 * High-performance 3-Layer Canvas 2D Board Renderer.
 * Offloads cartography rasterization to an Offscreen Canvas cache and renders tracks in batched GPU passes.
 */

import { SpatialHashGrid } from "./SpatialHashGrid";

export interface CanvasCity {
  name: string;
  x: number;
  y: number;
}

export interface CanvasRoute {
  id: string;
  city_a: string;
  city_b: string;
  length: number;
  color: string;
  claimed_by_player: number | null;
  parallel_index: number;
  total_parallel: number;
}

const COLOR_HEX: Record<string, string> = {
  purple: "#9b59b6",
  white: "#ecf0f1",
  blue: "#2980b9",
  yellow: "#f1c40f",
  orange: "#e67e22",
  black: "#2c3e50",
  red: "#e74c3c",
  green: "#27ae60",
  gray: "#7f8c8d",
  locomotive: "#d4af37",
};

export class BoardCanvasRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private offscreenBg: HTMLCanvasElement | null = null;
  private spatialHash: SpatialHashGrid;
  private hoveredRouteId: string | null = null;
  private selectedRouteId: string | null = null;
  private animOffset: number = 0;
  private width: number = 1100;
  private height: number = 700;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("Could not create 2D canvas context");
    this.ctx = ctx;
    this.spatialHash = new SpatialHashGrid(50);
  }

  public setDimensions(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.offscreenBg = null; // Invalidate background cache
  }

  public initBackgroundCache(): void {
    const bg = document.createElement("canvas");
    bg.width = this.width;
    bg.height = this.height;
    const bgCtx = bg.getContext("2d");
    if (!bgCtx) return;

    // 1. Vintage parchment base gradient
    const grad = bgCtx.createLinearGradient(0, 0, this.width, this.height);
    grad.addColorStop(0, "#f4ecd8");
    grad.addColorStop(0.5, "#ede0c4");
    grad.addColorStop(1, "#dfcca6");
    bgCtx.fillStyle = grad;
    bgCtx.fillRect(0, 0, this.width, this.height);

    // 2. Graticule coordinate lines
    bgCtx.strokeStyle = "rgba(139, 107, 73, 0.12)";
    bgCtx.lineWidth = 1;
    for (let x = 50; x < this.width; x += 60) {
      bgCtx.beginPath();
      bgCtx.moveTo(x, 0);
      bgCtx.lineTo(x, this.height);
      bgCtx.stroke();
    }
    for (let y = 50; y < this.height; y += 60) {
      bgCtx.beginPath();
      bgCtx.moveTo(0, y);
      bgCtx.lineTo(this.width, y);
      bgCtx.stroke();
    }

    // 3. Ornate border
    bgCtx.strokeStyle = "#8b6b49";
    bgCtx.lineWidth = 3;
    bgCtx.strokeRect(10, 10, this.width - 20, this.height - 20);
    bgCtx.strokeStyle = "rgba(139, 107, 73, 0.4)";
    bgCtx.lineWidth = 1;
    bgCtx.strokeRect(15, 15, this.width - 30, this.height - 30);

    this.offscreenBg = bg;
  }

  public rebuildSpatialIndex(routes: CanvasRoute[], cities: Record<string, CanvasCity>): void {
    this.spatialHash.clear();
    for (const r of routes) {
      const c1 = cities[r.city_a];
      const c2 = cities[r.city_b];
      if (!c1 || !c2) continue;

      const pOffset = (r.parallel_index - (r.total_parallel - 1) / 2) * 12;
      const dx = c2.x - c1.x;
      const dy = c2.y - c1.y;
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;

      this.spatialHash.insert({
        id: r.id,
        x1: c1.x + nx * pOffset,
        y1: c1.y + ny * pOffset,
        x2: c2.x + nx * pOffset,
        y2: c2.y + ny * pOffset,
        radius: 12,
        data: r,
      });
    }
  }

  public hitTest(px: number, py: number): string | null {
    const item = this.spatialHash.queryPoint(px, py, 14);
    return item ? item.id : null;
  }

  public setHoveredRoute(routeId: string | null): void {
    this.hoveredRouteId = routeId;
  }

  public setSelectedRoute(routeId: string | null): void {
    this.selectedRouteId = routeId;
  }

  public render(routes: CanvasRoute[], cities: Record<string, CanvasCity>): void {
    if (!this.offscreenBg) {
      this.initBackgroundCache();
    }

    // 1. Layer 1: Instant 0.1ms background blit
    if (this.offscreenBg) {
      this.ctx.drawImage(this.offscreenBg, 0, 0);
    }

    // 2. Layer 2: Dynamic batched tracks
    this.animOffset = (this.animOffset + 0.3) % 16;

    for (const r of routes) {
      const c1 = cities[r.city_a];
      const c2 = cities[r.city_b];
      if (!c1 || !c2) continue;

      const pOffset = (r.parallel_index - (r.total_parallel - 1) / 2) * 12;
      const dx = c2.x - c1.x;
      const dy = c2.y - c1.y;
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;

      const x1 = c1.x + nx * pOffset;
      const y1 = c1.y + ny * pOffset;
      const x2 = c2.x + nx * pOffset;
      const y2 = c2.y + ny * pOffset;

      const isHovered = this.hoveredRouteId === r.id;
      const isSelected = this.selectedRouteId === r.id;
      const isClaimed = r.claimed_by_player !== null;

      // Bed / ballast
      this.ctx.strokeStyle = "rgba(40, 30, 20, 0.4)";
      this.ctx.lineWidth = 10;
      this.ctx.lineCap = "round";
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();

      // Track color
      const baseColor = COLOR_HEX[r.color] || COLOR_HEX.gray;
      this.ctx.strokeStyle = isClaimed
        ? r.claimed_by_player === 0
          ? "#3498db"
          : "#e74c3c"
        : baseColor;
      this.ctx.lineWidth = 6;
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();

      // Segment dash markings
      this.ctx.strokeStyle = "#2c2c2c";
      this.ctx.lineWidth = 2;
      this.ctx.setLineDash([8, 4]);
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();
      this.ctx.setLineDash([]);

      // Pulsing highlight for hover/selected
      if (isHovered || isSelected) {
        this.ctx.strokeStyle = isSelected ? "#f39c12" : "#ffffff";
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([4, 4]);
        this.ctx.lineDashOffset = -this.animOffset;
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        this.ctx.lineDashOffset = 0;
      }
    }

    // 3. City Nodes
    for (const [name, c] of Object.entries(cities)) {
      this.ctx.fillStyle = "#2c3e50";
      this.ctx.beginPath();
      this.ctx.arc(c.x, c.y, 6, 0, Math.PI * 2);
      this.ctx.fill();

      this.ctx.strokeStyle = "#ffffff";
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      this.ctx.fillStyle = "#1a1a1a";
      this.ctx.font = "bold 11px sans-serif";
      this.ctx.textAlign = "center";
      this.ctx.fillText(name, c.x, c.y - 10);
    }
  }
}
