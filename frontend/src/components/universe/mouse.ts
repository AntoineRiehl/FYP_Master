//frontend/src/components/universe/mouse.ts

import type { Camera } from "./camera";


export function setupMouse(
    canvas: HTMLCanvasElement,
    camera: Camera
) {

    let dragging = false;

    let lastX = 0;
    let lastY = 0;


    const mouse = {
        x: 0,
        y: 0
    };


    function mouseDown(e: MouseEvent) {

        dragging = true;

        lastX = e.clientX;
        lastY = e.clientY;
    }


    function mouseUp() {

        dragging = false;
    }


    function mouseMove(e: MouseEvent) {

        mouse.x = e.clientX;
        mouse.y = e.clientY;


        if (!dragging) return;


        camera.x -=
            (e.clientX - lastX) / camera.zoom;


        camera.y -=
            (e.clientY - lastY) / camera.zoom;


        lastX = e.clientX;
        lastY = e.clientY;
    }


    function wheel(e: WheelEvent) {

        e.preventDefault();


        const factor = 1.08;


        if (e.deltaY < 0) {
            camera.zoom *= factor;
        }
        else {
            camera.zoom /= factor;
        }


        camera.zoom =
            Math.max(
                3,
                Math.min(camera.zoom, 800)
            );
    }


    canvas.addEventListener(
        "mousedown",
        mouseDown
    );

    window.addEventListener(
        "mouseup",
        mouseUp
    );

    window.addEventListener(
        "mousemove",
        mouseMove
    );

    canvas.addEventListener(
        "wheel",
        wheel,
        {
            passive:false
        }
    );


    return {

        mouse,

        cleanup() {

            canvas.removeEventListener(
                "mousedown",
                mouseDown
            );

            window.removeEventListener(
                "mouseup",
                mouseUp
            );

            window.removeEventListener(
                "mousemove",
                mouseMove
            );

            canvas.removeEventListener(
                "wheel",
                wheel
            );
        }
    };
}