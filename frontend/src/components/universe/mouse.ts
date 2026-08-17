//frontend/src/components/universe/mouse.ts

import type { Camera } from "./camera";


export function setupMouse(
    canvas: HTMLCanvasElement,
    camera: Camera,
    onClick: (x:number, y:number)=>void
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


    function mouseClick(e: MouseEvent) {

        onClick(
            e.offsetX,
            e.offsetY
        );
    }


    function mouseMove(e: MouseEvent) {

        mouse.x = e.offsetX;
        mouse.y = e.offsetY;


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

        const mouseX = e.offsetX;
        const mouseY = e.offsetY;


        const worldX =
            camera.x +
            (mouseX - canvas.width / 2)
            /
            camera.zoom;


        const worldY =
            camera.y +
            (mouseY - canvas.height / 2)
            /
            camera.zoom;


        const factor = 1.08;


        if (e.deltaY < 0) {

            camera.zoom *= factor;

        } else {

            camera.zoom /= factor;

        }


        camera.zoom = Math.max(
            3,
            Math.min(camera.zoom, 10000000)
        );


        camera.x =
            worldX -
            (mouseX - canvas.width / 2)
            /
            camera.zoom;


        camera.y =
            worldY -
            (mouseY - canvas.height / 2)
            /
            camera.zoom;

    }


    canvas.addEventListener(
        "mousedown",
        mouseDown
    );


    canvas.addEventListener(
        "click",
        mouseClick
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


            canvas.removeEventListener(
                "click",
                mouseClick
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