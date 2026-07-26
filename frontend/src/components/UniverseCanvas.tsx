// frontend/src/components/UniverseCanvas.tsx

import { useEffect, useRef } from "react";
import type { MovieNode } from "../types/movie";

type Props = {
  data: MovieNode[];
};

export default function UniverseCanvas({ data }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;

    // =====================================================
    // CANVAS SETUP
    // =====================================================

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }

    resize();
    window.addEventListener("resize", resize);

    // =====================================================
    // CAMERA (core of system)
    // =====================================================

    const camera = {
      x: 0,
      y: 0,
      zoom: 20
    };

    // =====================================================
    // INPUT STATE
    // =====================================================

    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    let mouseX = 0;
    let mouseY = 0;

    // =====================================================
    // PREPROCESS: SAFE SUBSETS (IMPORTANT FOR PERFORMANCE)
    // =====================================================

    // deterministic sampling helper (prevents flicker)
    function sampleArray<T>(arr: T[], max: number): T[] {
      if (arr.length <= max) return arr;

      const step = Math.floor(arr.length / max);
      const out: T[] = [];

      for (let i = 0; i < arr.length; i += step) {
        out.push(arr[i]);
        if (out.length >= max) break;
      }

      return out;
    }

    // =====================================================
    // VIEWPORT FILTER
    // =====================================================

    function inView(x: number, y: number) {
      const screen = worldToScreen(x, y);
      return (
        screen.x >= -100 &&
        screen.x <= canvas.width + 100 &&
        screen.y >= -100 &&
        screen.y <= canvas.height + 100
      );
    }

    // =====================================================
    // WORLD → SCREEN
    // =====================================================

    function worldToScreen(x: number, y: number) {
      return {
        x: (x - camera.x) * camera.zoom + canvas.width / 2,
        y: (y - camera.y) * camera.zoom + canvas.height / 2
      };
    }

    // =====================================================
    // DISTANCE (hover)
    // =====================================================

    function dist(x1: number, y1: number, x2: number, y2: number) {
      return Math.hypot(x1 - x2, y1 - y2);
    }

    // =====================================================
    // LOD SYSTEM (CORE LOGIC)
    // =====================================================

    function getLOD() {
      if (camera.zoom < 12) return 0;   // regions
      if (camera.zoom < 60) return 1;   // landmarks
      return 2;                         // full
    }

    // =====================================================
    // COLOR SYSTEM (stable)
    // =====================================================

    function clusterColor(id: number) {
      const colors = [
        "rgba(255,120,120,0.7)",
        "rgba(120,200,255,0.7)",
        "rgba(180,120,255,0.7)",
        "rgba(255,200,120,0.7)",
        "rgba(120,255,160,0.7)"
      ];
      return colors[Math.abs(id) % colors.length];
    }

    // =====================================================
    // INPUT EVENTS
    // =====================================================

    canvas.addEventListener("mousedown", (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    });

    window.addEventListener("mouseup", () => {
      dragging = false;
    });

    window.addEventListener("mousemove", (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;

      if (!dragging) return;

      camera.x -= (e.clientX - lastX) / camera.zoom;
      camera.y -= (e.clientY - lastY) / camera.zoom;

      lastX = e.clientX;
      lastY = e.clientY;
    });

    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();

        const factor = 1.08;

        if (e.deltaY < 0) {
          camera.zoom *= factor;
        } else {
          camera.zoom /= factor;
        }

        // IMPORTANT: wider zoom range
        camera.zoom = Math.max(3, Math.min(camera.zoom, 800));
      },
      { passive: false }
    );

    // =====================================================
    // DRAW LOOP
    // =====================================================

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const mode = getLOD();

      let hoverText: string | null = null;
      let hoverDist = Infinity;

      // =====================================================
      // REGION MODE (LOW ZOOM)
      // =====================================================

      if (mode === 0) {
        const regions = sampleArray(
          [...new Map(data.map(d => [d.cluster, d])).values()],
          12 // HARD CAP
        );

        for (const r of regions) {
          const pos = worldToScreen(r.umap_x, r.umap_y);

          const radius = 28;

          ctx.beginPath();
          ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = clusterColor(r.cluster);
          ctx.fill();

          ctx.fillStyle = "white";
          ctx.font = "13px Arial";
          ctx.fillText(r.cluster_label ?? "Region", pos.x + 10, pos.y);

          const d = dist(mouseX, mouseY, pos.x, pos.y);
          if (d < radius && d < hoverDist) {
            hoverText = r.cluster_label ?? "Region";
            hoverDist = d;
          }
        }
      }

      // =====================================================
      // LANDMARK MODE (MID ZOOM)
      // =====================================================

      else if (mode === 1) {
        const landmarks = sampleArray(data, 2500); // HARD CAP

        for (const m of landmarks) {
          const pos = worldToScreen(m.umap_x, m.umap_y);

          if (!inView(m.umap_x, m.umap_y)) continue;

          const radius = Math.sqrt(m.visual_size) * 1.5;

          ctx.beginPath();
          ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = clusterColor(m.cluster);
          ctx.fill();

          // only important labels
          if (m.rating_count > 4000) {
            ctx.fillStyle = "white";
            ctx.font = "11px Arial";
            ctx.fillText(m.title, pos.x + 6, pos.y);
          }

          const d = dist(mouseX, mouseY, pos.x, pos.y);
          if (d < radius && d < hoverDist) {
            hoverText = m.title;
            hoverDist = d;
          }
        }
      }

      // =====================================================
      // FULL MODE (HIGH ZOOM)
      // =====================================================

      else {
        const visible = data.filter(d => inView(d.umap_x, d.umap_y));

        const capped = sampleArray(visible, 6000); // HARD SAFETY CAP

        for (const m of capped) {
          const pos = worldToScreen(m.umap_x, m.umap_y);

          const radius = Math.max(1.2, Math.sqrt(m.visual_size) * 0.8);

          ctx.beginPath();
          ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = clusterColor(m.cluster);
          ctx.fill();

          const d = dist(mouseX, mouseY, pos.x, pos.y);
          if (d < radius && d < hoverDist) {
            hoverText = m.title;
            hoverDist = d;
          }
        }
      }

      // =====================================================
      // HOVER TOOLTIP
      // =====================================================

      if (hoverText) {
        ctx.fillStyle = "rgba(0,0,0,0.75)";
        ctx.fillRect(mouseX + 10, mouseY + 10, 220, 26);

        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.fillText(hoverText, mouseX + 15, mouseY + 28);
      }

      requestAnimationFrame(draw);
    }

    draw();

    // =====================================================
    // CLEANUP
    // =====================================================

    return () => {
      window.removeEventListener("resize", resize);
    };
  }, [data]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        display: "block",
        background: "#050816",
        cursor: "grab"
      }}
    />
  );
}