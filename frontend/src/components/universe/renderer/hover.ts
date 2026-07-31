//frontend/src/components/universe/renderer/hover.ts

import type { AtlasNode } from "../../../types/atlas";

import type { Camera } from "../camera";

import {
    worldToScreen
} from "../camera";

import {
    distance
} from "./utils";


export function detectHover(

    canvas: HTMLCanvasElement,

    camera: Camera,

    nodes: AtlasNode[],

    mouse: {
        x:number;
        y:number;
    }

): AtlasNode | null {


    let hovered: AtlasNode | null = null;

    let closest = Infinity;



    for (const node of nodes) {


        const pos =
            worldToScreen(
                camera,
                canvas,
                node.umap_x,
                node.umap_y
            );


        const radius =
            Math.max(
                8,
                Math.sqrt(node.visual_size)
            );


        const d =
            distance(
                mouse.x,
                mouse.y,
                pos.x,
                pos.y
            );


        if (
            d < radius &&
            d < closest
        ) {

            closest = d;

            hovered = node;

        }

    }


    return hovered;

}