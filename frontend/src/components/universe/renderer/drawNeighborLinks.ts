// frontend/src/components/universe/renderer/drawNeighborLinks.ts

import type {
    AtlasNode
} from "../../../types/atlas";


import type {
    Camera
} from "../camera";


import {
    worldToScreen
} from "../camera";


import type {
    NeighborResult
} from "../neighbors";



// =====================================================
// DRAW NEIGHBOUR LINKS
// =====================================================

export function drawNeighborLinks(

    ctx: CanvasRenderingContext2D,

    canvas: HTMLCanvasElement,

    camera: Camera,

    selectedNode: AtlasNode,

    neighbors: NeighborResult[]

) {


    if (

        neighbors.length === 0

    ) {

        return;

    }



    // =================================================
    // SELECTED POSITION
    // =================================================

    const selectedPosition =

        worldToScreen(

            camera,

            canvas,

            selectedNode.position.x,

            selectedNode.position.y

        );



    ctx.save();



    // =================================================
    // LINE STYLE
    // =================================================

    /*
     * Links should explain neighbourhood structure
     * without turning the atlas into a network graph.
     */

    ctx.lineWidth =
        1;


    ctx.strokeStyle =
        "rgba(255, 255, 255, 0.22)";


    ctx.lineCap =
        "round";



    // =================================================
    // DRAW EACH LINK
    // =================================================

    for (const neighbor of neighbors) {


        const neighborPosition =

            worldToScreen(

                camera,

                canvas,

                neighbor.node.position.x,

                neighbor.node.position.y

            );



        ctx.beginPath();


        ctx.moveTo(

            selectedPosition.x,

            selectedPosition.y

        );


        ctx.lineTo(

            neighborPosition.x,

            neighborPosition.y

        );


        ctx.stroke();

    }



    ctx.restore();

}