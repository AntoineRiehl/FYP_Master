//frontend/src/components/universe/renderer/drawRegions.ts

import type {
    RegionNode
} from "../../../types/atlas";


import type {
    Camera
} from "../camera";


import {
    worldToScreen
} from "../camera";


import {
    clusterColor
} from "../colors";


import {
    sampleArray
} from "./utils";



export function drawRegions(

    ctx:CanvasRenderingContext2D,

    canvas:HTMLCanvasElement,

    camera:Camera,

    regions:RegionNode[]

){


    const visibleRegions =

        sampleArray(

            regions,

            50

        );



    for(const region of visibleRegions){



        const pos =

            worldToScreen(

                camera,

                canvas,

                region.x,

                region.y

            );



        const radius =

            Math.max(

                20,

                Math.sqrt(
                    region.size
                ) * 5

            );



        ctx.beginPath();



        ctx.arc(

            pos.x,

            pos.y,

            radius,

            0,

            Math.PI*2

        );



        ctx.fillStyle =

            region.color

            ??

            clusterColor(
                region.id
            );



        ctx.fill();



        ctx.fillStyle =
            "white";


        ctx.font =
            "13px Arial";



        ctx.fillText(

            region.label ?? "Region",

            pos.x+10,

            pos.y

        );

    }

}