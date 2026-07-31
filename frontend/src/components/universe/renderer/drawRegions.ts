//frontend/src/components/universe/renderer/drawRegions.ts

import type { AtlasNode } from "../../../types/atlas";

import type { Camera } from "../camera";

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
    data:AtlasNode[]
){


    const regions =
        sampleArray(
            [
                ...new Map(
                    data.map(
                        d=>[
                            d.cluster,
                            d
                        ]
                    )
                ).values()
            ],
            12
        );



    for(const r of regions){


        const pos =
            worldToScreen(
                camera,
                canvas,
                r.umap_x,
                r.umap_y
            );


        const radius=28;


        ctx.beginPath();

        ctx.arc(
            pos.x,
            pos.y,
            radius,
            0,
            Math.PI*2
        );


        ctx.fillStyle =
            clusterColor(r.cluster);


        ctx.fill();



        ctx.fillStyle="white";

        ctx.font="13px Arial";


        ctx.fillText(
            r.cluster_label ?? "Region",
            pos.x+10,
            pos.y
        );

    }

}