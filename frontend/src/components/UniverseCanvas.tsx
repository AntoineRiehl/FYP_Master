// frontend/src/components/UniverseCanvas.tsx

import {
    useEffect,
    useRef
} from "react";


import type {
    AtlasNode
} from "../types/atlas";


import {
    createCamera
} from "./universe/camera";


import {
    setupMouse
} from "./universe/mouse";


import {
    renderUniverse
} from "./universe/renderer";



type Props = {

    data: AtlasNode[];

    onHover:
        (node: AtlasNode | null) => void;

};



export default function UniverseCanvas({

    data,
    onHover

}:Props){


    const canvasRef =
        useRef<HTMLCanvasElement>(null);



    const hoveredRef =
        useRef<AtlasNode | null>(null);



    useEffect(()=>{


        const canvas =
            canvasRef.current!;


        const ctx =
            canvas.getContext("2d")!;



        function resize(){


            const parent =
                canvas.parentElement;


            if(!parent)
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



        const camera =
            createCamera();



        const {
            mouse,
            cleanup
        } =
            setupMouse(
                canvas,
                camera
            );



        let animation:number;



        function frame(){


            const hovered =
                renderUniverse(
                    ctx,
                    canvas,
                    camera,
                    data,
                    mouse
                );



            const previous =
                hoveredRef.current;



            if(
                previous?.id !== hovered?.id
            ){

                hoveredRef.current =
                    hovered;


                onHover(
                    hovered
                );

            }



            animation =
                requestAnimationFrame(
                    frame
                );

        }



        frame();



        return ()=>{


            cancelAnimationFrame(
                animation
            );


            cleanup();



            window.removeEventListener(
                "resize",
                resize
            );

        };


    },[
        data,
        onHover
    ]);



    return (

        <canvas

            ref={canvasRef}

            style={{

                width:"100%",

                height:"100%",

                display:"block",

                background:"#050816",

                cursor:"grab"

            }}

        />

    );

}