/**
 * High-performance 2D Spatial Hash Grid for O(1) route hit-testing.
 */

export interface SpatialRouteItem {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  radius: number;
  data: any;
}

export class SpatialHashGrid {
  private cellSize: number;
  private grid: Map<string, SpatialRouteItem[]>;

  constructor(cellSize: number = 40) {
    this.cellSize = cellSize;
    this.grid = new Map();
  }

  private getKey(cx: number, cy: number): string {
    return `${cx}:${cy}`;
  }

  public clear(): void {
    this.grid.clear();
  }

  public insert(item: SpatialRouteItem): void {
    const minX = Math.min(item.x1, item.x2) - item.radius;
    const maxX = Math.max(item.x1, item.x2) + item.radius;
    const minY = Math.min(item.y1, item.y2) - item.radius;
    const maxY = Math.max(item.y1, item.y2) + item.radius;

    const startX = Math.floor(minX / this.cellSize);
    const endX = Math.floor(maxX / this.cellSize);
    const startY = Math.floor(minY / this.cellSize);
    const endY = Math.floor(maxY / this.cellSize);

    for (let cx = startX; cx <= endX; cx++) {
      for (let cy = startY; cy <= endY; cy++) {
        const key = this.getKey(cx, cy);
        let list = this.grid.get(key);
        if (!list) {
          list = [];
          this.grid.set(key, list);
        }
        list.push(item);
      }
    }
  }

  public queryPoint(px: number, py: number, tolerance: number = 10): SpatialRouteItem | null {
    const cx = Math.floor(px / this.cellSize);
    const cy = Math.floor(py / this.cellSize);
    const list = this.grid.get(this.getKey(cx, cy));
    if (!list) return null;

    let closestItem: SpatialRouteItem | null = null;
    let minDistance = tolerance;

    for (const item of list) {
      const d = this.distToSegment(px, py, item.x1, item.y1, item.x2, item.y2);
      if (d <= minDistance) {
        minDistance = d;
        closestItem = item;
      }
    }

    return closestItem;
  }

  private distToSegment(px: number, py: number, x1: number, y1: number, x2: number, y2: number): number {
    const l2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    if (l2 === 0) return Math.hypot(px - x1, py - y1);

    let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
    t = Math.max(0, Math.min(1, t));

    const projX = x1 + t * (x2 - x1);
    const projY = y1 + t * (y2 - y1);
    return Math.hypot(px - projX, py - projY);
  }
}
