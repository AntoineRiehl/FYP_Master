//frontend/src/components/universe/colors.ts

export function clusterColor(id: number) {

    const colors = [
        "rgba(255,120,120,0.7)",
        "rgba(120,200,255,0.7)",
        "rgba(180,120,255,0.7)",
        "rgba(255,200,120,0.7)",
        "rgba(120,255,160,0.7)"
    ];

    return colors[Math.abs(id) % colors.length];
}