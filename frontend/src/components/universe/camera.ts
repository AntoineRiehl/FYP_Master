//frontend/src/components/universe/camera.ts

export type Camera = {
    x: number;
    y: number;
    zoom: number;
};


export function createCamera(): Camera {
    return {
        x: 0,
        y: 0,
        zoom: 20
    };
}


export function worldToScreen(
    camera: Camera,
    canvas: HTMLCanvasElement,
    x: number,
    y: number
) {
    return {
        x: (x - camera.x) * camera.zoom + canvas.width / 2,
        y: (y - camera.y) * camera.zoom + canvas.height / 2
    };
}