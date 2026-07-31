//frontend/src/components/universe/renderer.ts
//MIGHT NEED TO SEPARATE INTO RENDERER FOLDER

import type { AtlasNode } from "../../types/atlas";

import type { Camera } from "./camera";

import {
    worldToScreen
} from "./camera";

import {
    clusterColor
} from "./colors";

import {
    getLOD
} from "./lod";


function sampleArray<T>(
    arr:T[],
    max:number
){

    if(arr.length <= max)
        return arr;


    const step =
        Math.floor(arr.length/max);


    const output:T[]=[];


    for(
        let i=0;
        i<arr.length;
        i+=step
    ){

        output.push(arr[i]);

        if(output.length>=max)
            break;
    }


    return output;
}



function distance(
    x1:number,
    y1:number,
    x2:number,
    y2:number
){

    return Math.hypot(
        x1-x2,
        y1-y2
    );
}



function inView(
    camera:Camera,
    canvas:HTMLCanvasElement,
    x:number,
    y:number
){

    const p =
        worldToScreen(
            camera,
            canvas,
            x,
            y
        );


    return (
        p.x>-100 &&
        p.x<canvas.width+100 &&
        p.y>-100 &&
        p.y<canvas.height+100
    );
}



export function renderUniverse(
    ctx:CanvasRenderingContext2D,
    canvas:HTMLCanvasElement,
    camera:Camera,
    data:AtlasNode[],
    mouse:{
        x:number,
        y:number
    }
){


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    const mode =
        getLOD(camera.zoom);



    let hoverText:string|null=null;

    let hoverDist=Infinity;



    if(mode===0){


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



            const d =
                distance(
                    mouse.x,
                    mouse.y,
                    pos.x,
                    pos.y
                );


            if(
                d<radius &&
                d<hoverDist
            ){

                hoverText =
                    r.cluster_label ??
                    "Region";

                hoverDist=d;
            }
        }

    }


    else {


        const visible =
            data.filter(
                d =>
                    inView(
                        camera,
                        canvas,
                        d.umap_x,
                        d.umap_y
                    )
            );


        const capped =
            mode===1
            ?
            sampleArray(visible,2500)
            :
            sampleArray(visible,6000);



        for(const m of capped){


            const pos =
                worldToScreen(
                    camera,
                    canvas,
                    m.umap_x,
                    m.umap_y
                );


            const radius =
                mode===1
                ?
                Math.sqrt(m.visual_size)*1.5
                :
                Math.max(
                    1.2,
                    Math.sqrt(m.visual_size)*0.8
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
                clusterColor(m.cluster);

            ctx.fill();



            const d =
                distance(
                    mouse.x,
                    mouse.y,
                    pos.x,
                    pos.y
                );


            if(
                d<radius &&
                d<hoverDist
            ){

                hoverText =
                    m.label;

                hoverDist=d;
            }
        }
    }



    if(hoverText){

        ctx.fillStyle=
            "rgba(0,0,0,0.75)";


        ctx.fillRect(
            mouse.x+10,
            mouse.y+10,
            220,
            26
        );


        ctx.fillStyle="white";

        ctx.font="12px Arial";


        ctx.fillText(
            hoverText,
            mouse.x+15,
            mouse.y+28
        );
    }
}