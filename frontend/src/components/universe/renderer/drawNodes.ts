//frontend/src/components/universe/renderer/drawNodes.ts

import type {
    AtlasNode
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
    sampleArray,
    inView
} from "./utils";



export function drawNodes(

    ctx:CanvasRenderingContext2D,

    canvas:HTMLCanvasElement,

    camera:Camera,

    data:AtlasNode[],

    mode:number

){


    const visible =

        data.filter(

            node =>

                inView(

                    camera,

                    canvas,

                    node.position.x,

                    node.position.y

                )

        );



    const capped =

        mode===1

        ?

        sampleArray(
            visible,
            2500
        )

        :

        sampleArray(
            visible,
            6000
        );



    for(const node of capped){



        const pos =

            worldToScreen(

                camera,

                canvas,

                node.position.x,

                node.position.y

            );



        const radius =

            mode===1

            ?

            Math.sqrt(
                node.visual.size
            ) * 1.5


            :


            Math.max(

                1.5,

                Math.sqrt(
                    node.visual.size
                )

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

            clusterColor(

                node.visual.cluster ?? -1

            );



        ctx.fill();



        if(

            mode===2

            &&

            radius>4

        ){


            ctx.fillStyle =
                "white";


            ctx.font =
                "11px Arial";


            ctx.fillText(

                node.title,

                pos.x+5,

                pos.y

            );

        }

    }

}