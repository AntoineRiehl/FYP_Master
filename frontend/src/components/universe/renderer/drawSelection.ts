// frontend/src/components/universe/renderer/drawSelection.ts

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
    getNodeColor
} from "../colors";


import type {
    ColorMode
} from "../../../config/colorMode";



// =====================================================
// DRAW SELECTED NODE
// =====================================================

export function drawSelection(

    ctx: CanvasRenderingContext2D,

    canvas: HTMLCanvasElement,

    camera: Camera,

    node: AtlasNode,

    mode: number,

    colorMode: ColorMode

) {


    // =================================================
    // SCREEN POSITION
    // =================================================

    const pos =

        worldToScreen(

            camera,

            canvas,

            node.position.x,

            node.position.y

        );



    // =================================================
    // NORMAL NODE RADIUS
    // =================================================

    /*
     * Match the same radius rules used by drawNodes.
     */

    const baseRadius =

        mode === 1

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



    // =================================================
    // SELECTED RADIUS
    // =================================================

    /*
     * Selection enlargement is purely visual.
     *
     * node.visual.size itself is NOT modified.
     */

    const selectedRadius =

        Math.max(

            5,

            baseRadius * 1.4

        );



    ctx.save();



    // =================================================
    // SOFT HALO
    // =================================================

    ctx.beginPath();


    ctx.arc(

        pos.x,

        pos.y,

        selectedRadius + 7,

        0,

        Math.PI * 2

    );


    ctx.fillStyle =
        "rgba(255, 255, 255, 0.12)";


    ctx.fill();



    // =================================================
    // OUTER WHITE RING
    // =================================================

    ctx.beginPath();


    ctx.arc(

        pos.x,

        pos.y,

        selectedRadius + 3,

        0,

        Math.PI * 2

    );


    ctx.strokeStyle =
        "rgba(255, 255, 255, 0.95)";


    ctx.lineWidth =
        2;


    ctx.stroke();



    // =================================================
    // SELECTED NODE
    // =================================================

    ctx.beginPath();


    ctx.arc(

        pos.x,

        pos.y,

        selectedRadius,

        0,

        Math.PI * 2

    );


    ctx.fillStyle =

        getNodeColor(

            node,

            colorMode

        );


    ctx.fill();



    // =================================================
    // INNER HIGHLIGHT
    // =================================================

    /*
     * Thin white border keeps the selected node readable
     * even when its cluster/domain colour is dark.
     */

    ctx.strokeStyle =
        "white";


    ctx.lineWidth =
        1.5;


    ctx.stroke();



    // =================================================
    // ALWAYS-VISIBLE LABEL
    // =================================================

    const labelX =

        pos.x
        +
        selectedRadius
        +
        8;


    const labelY =

        pos.y;


    ctx.font =
        "bold 12px Arial";


    ctx.textBaseline =
        "middle";


    /*
     * Dark text halo.
     */

    ctx.strokeStyle =
        "rgba(5, 8, 22, 0.95)";


    ctx.lineWidth =
        4;


    ctx.strokeText(

        node.title,

        labelX,

        labelY

    );


    ctx.fillStyle =
        "white";


    ctx.fillText(

        node.title,

        labelX,

        labelY

    );



    ctx.restore();

}