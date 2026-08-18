// frontend/src/components/UniverseCanvas.tsx

// frontend/src/components/UniverseCanvas.tsx

import {
    useEffect,
    useRef
} from "react";


import type {
    AtlasData,
    AtlasNode
} from "../types/atlas";


import type {
    ColorMode
} from "../config/colorMode";


import {
    createCamera
} from "./universe/camera";


import {
    setupMouse
} from "./universe/mouse";


import {
    renderUniverse
} from "./universe/renderer";


import {
    detectSelection
} from "./universe/renderer/select";


// =====================================================
// PROPS
// =====================================================

type Props = {

    data: AtlasData;

    colorMode: ColorMode;

    onSelect:
        (node: AtlasNode | null) => void;

};


// =====================================================
// COMPONENT
// =====================================================

export default function UniverseCanvas({

    data,

    colorMode,

    onSelect

}: Props) {


    const canvasRef =
        useRef<HTMLCanvasElement>(null);


    // Camera persists between renders.
    const cameraRef =
        useRef(createCamera());



    useEffect(() => {


        const canvas =
            canvasRef.current!;


        const ctx =
            canvas.getContext("2d")!;


        const camera =
            cameraRef.current;



        // =================================================
        // RESIZE
        // =================================================

        function resize() {

            const parent =
                canvas.parentElement;


            if (!parent)
                return;


            canvas.width =
                parent.clientWidth;


            canvas.height =
                parent.clientHeight;

        }



        resize();



        window.addEventListener(
            "resize",
            resize
        );



        // =================================================
        // MOUSE
        // =================================================

        const {
            mouse,
            cleanup
        } =

            setupMouse(

                canvas,

                camera,

                (x, y) => {


                    const selected =

                        detectSelection(

                            canvas,

                            camera,

                            data.atlas,

                            {
                                x,
                                y
                            }

                        );


                    onSelect(
                        selected
                    );

                }

            );



        // =================================================
        // ANIMATION LOOP
        // =================================================

        let animation: number;



        function frame() {


            renderUniverse(

                ctx,

                canvas,

                camera,

                data,

                mouse,

                colorMode

            );


            animation =

                requestAnimationFrame(
                    frame
                );

        }



        frame();



        // =================================================
        // CLEANUP
        // =================================================

        return () => {


            cancelAnimationFrame(
                animation
            );


            cleanup();


            window.removeEventListener(
                "resize",
                resize
            );

        };


    }, [

        data,

        onSelect,

        colorMode

    ]);



    return (

        <canvas

            ref={canvasRef}

            style={{

                width: "100%",

                height: "100%",

                display: "block",

                background: "#050816",

                cursor: "grab"

            }}

        />

    );

}