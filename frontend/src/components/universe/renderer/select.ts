//frontend/src/components/universe/renderer/select.ts

import type { AtlasNode } from "../../../types/atlas";
import type { Camera } from "../camera";

import {
    worldToScreen
} from "../camera";

import {
    distance
} from "./utils";


export function detectSelection(

    canvas:HTMLCanvasElement,

    camera:Camera,

    nodes:AtlasNode[],

    mouse:{
        x:number;
        y:number;
    }

):AtlasNode|null{


    let closest = Infinity;

    let selected:AtlasNode|null=null;


    for(const node of nodes){


        const pos =
            worldToScreen(
                camera,
                canvas,
                node.position.x,
                node.position.y
            );


        const d =
            distance(
                mouse.x,
                mouse.y,
                pos.x,
                pos.y
            );


        if(
            d < 10 &&
            d < closest
        ){

            closest=d;
            selected=node;

        }

    }


    return selected;

}