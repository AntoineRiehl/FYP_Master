//frontend/src/components/universe/renderer.ts

import type {
    AtlasData,
    AtlasNode
} from "../../types/atlas";


import type {
    Camera
} from "./camera";


import {
    getLOD
} from "./lod";


import {
    drawRegions
} from "./renderer/drawRegions";


import {
    drawNodes
} from "./renderer/drawNodes";


import {
    detectHover
} from "./renderer/hover";



export function renderUniverse(

    ctx:CanvasRenderingContext2D,

    canvas:HTMLCanvasElement,

    camera:Camera,

    data:AtlasData,

    mouse:{
        x:number;
        y:number;
    }

):AtlasNode|null{


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );



    const mode =
        getLOD(
            camera.zoom
        );



    if(mode===0){


        drawRegions(
            ctx,
            canvas,
            camera,
            data.regions
        );


    }
    else{


        drawNodes(
            ctx,
            canvas,
            camera,
            data.atlas,
            mode
        );


    }



    const hovered =
        detectHover(
            canvas,
            camera,
            data.atlas,
            mouse
        );



    if(hovered){


        ctx.fillStyle =
            "rgba(0,0,0,0.75)";


        ctx.fillRect(
            mouse.x+10,
            mouse.y+10,
            230,
            28
        );


        ctx.fillStyle =
            "white";


        ctx.font =
            "12px Arial";


        ctx.fillText(

            hovered.title,

            mouse.x+15,

            mouse.y+28

        );

    }



    return hovered;

}