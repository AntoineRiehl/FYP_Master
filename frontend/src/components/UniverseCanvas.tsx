// frontend/src/components/UniverseCanvas.tsx

import {
    useEffect,
    useRef
} from "react";

import type {
    AtlasData,
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

import {
    detectSelection
} from "./universe/renderer/select";



type Props = {

    data:AtlasData;

    onSelect:
        (node:AtlasNode|null)=>void;

};



export default function UniverseCanvas({

    data,

    onSelect

}:Props){


    const canvasRef =
        useRef<HTMLCanvasElement>(null);

    // IMPORTANT:
    // Camera is created ONCE and survives re-renders.
    const cameraRef =
        useRef(createCamera());



    useEffect(()=>{

        const canvas =
            canvasRef.current!;

        const ctx =
            canvas.getContext("2d")!;

        const camera =
            cameraRef.current;



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



        const{
            mouse,
            cleanup
        }=
            setupMouse(

                canvas,

                camera,

                (x,y)=>{

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

                    onSelect(selected);

                }

            );



        let animation:number;



        function frame(){

            renderUniverse(

                ctx,

                canvas,

                camera,

                data,

                mouse

            );

            animation =
                requestAnimationFrame(
                    frame
                );

        }



        frame();



        return()=>{

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
        onSelect
    ]);



    return(

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